import copy
import csv
import logging
import math
import os
import time
from typing import Any, Dict, List

import torch
from torch.utils.tensorboard import SummaryWriter

from model.runtime_modes import (
    RUNTIME_MODE_HOST_ONLY,
    RUNTIME_MODE_JOINT_TRAINING,
    normalize_runtime_mode,
    resolve_runtime_mode_from_args,
)
from solver import (
    build_lr_scheduler,
    build_optimizer,
    summarize_optimizer_param_groups,
    summarize_scheduler_effective_lrs,
)
from utils.comm import get_rank, synchronize
from utils.experiment import ExperimentTracker
from utils.freeze_schedule import (
    LOGICAL_TO_OPTIMIZER_GROUPS,
    apply_loss_weight_overrides,
    apply_optimizer_lr_overrides,
    apply_phase_trainability,
    get_group_trainability_snapshot,
    get_active_phase,
    parse_freeze_schedule_config,
)
from utils.metric_logging import (
    TRACKED_SCALAR_KEYS,
    TRAIN_LOSS_KEYS,
    RoutingCoverageTracker,
    build_train_metrics_from_scalars,
    build_validation_metrics,
    collect_scalar_metrics,
)
from utils.meter import AverageMeter
from utils.metrics import Evaluator, collect_monitored_eval_rows, summarize_epoch_monitor
from utils.metrics_prototype import (
    collect_prototype_debug_metrics,
    resolve_prototype_bank_tensor,
)
from utils.precision import build_autocast_context, build_grad_scaler, canonicalize_amp_dtype, is_amp_enabled, is_cuda_device


METER_KEYS = ('loss_total',) + tuple(key for key in TRACKED_SCALAR_KEYS if key != 'loss_total')
_CONSOLE_LOSS_SKIP_KEYS = {
    # Keep canonical naming in console logs.
    'loss_proto_total',  # alias of loss_proto in current prototype path
}
CONSOLE_LOSS_LOG_KEYS = tuple(
    key for key in TRAIN_LOSS_KEYS if key in METER_KEYS and key not in _CONSOLE_LOSS_SKIP_KEYS
)

PAIRWISE_HARD_EXPORT_COLUMNS = (
    'epoch',
    'step',
    'anchor_id',
    'negative_id',
    'is_positive',
    'rank_position',
    's_pos_host',
    's_neg_host',
    'margin_host',
    'margin_global',
    'margin_local',
    'gate_host',
    'gate_global',
    'gate_local',
    'proto_pair_signal',
    'proto_gate',
    'omega',
    'routing_entropy',
    'routing_top1_top2_gap',
    'diag_cos_full',
    'split',
    'method_variant',
)

_HBR_CONTROL_MODE_TO_METHOD_VARIANT = {
    'host_only_weight': 'hbr_host_only_weight',
    'proto_weight': 'hbr_proto_weight',
    'none': 'hbr_proto_weight',
    'proto_weight_shuffled': 'hbr_proto_weight_shuffled',
    'random_matched_weight': 'hbr_random_matched_weight',
    'proto_adaptive_margin': 'hbr_proto_adaptive_margin',
}


def _to_python_float(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (float, int, bool)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().reshape(-1)[0].cpu().item())
    return float(default)


def _resolve_method_variant(args, runtime_mode: str) -> str:
    mode = normalize_runtime_mode(runtime_mode)
    use_prototype_branch = bool(getattr(args, 'use_prototype_branch', False))
    use_hbr = bool(getattr(args, 'use_hbr', False))
    if mode == RUNTIME_MODE_HOST_ONLY or not use_prototype_branch:
        return 'host_only_baseline'
    if not use_hbr:
        return 'host_plus_diag_only'
    control_mode = str(getattr(args, 'hbr_control_mode', 'none') or 'none').lower()
    return _HBR_CONTROL_MODE_TO_METHOD_VARIANT.get(control_mode, f'hbr_{control_mode}')


def _append_hbr_pairwise_rows(
    *,
    rows: List[Dict[str, object]],
    outputs: Dict[str, object],
    batch: Dict[str, torch.Tensor],
    epoch: int,
    step: int,
    max_rows: int,
    method_variant: str,
) -> None:
    if max_rows <= 0 or len(rows) >= max_rows:
        return
    export = outputs.get('hbr_pairwise_export')
    if not isinstance(export, dict):
        return

    anchor_index = export.get('anchor_index')
    negative_index = export.get('negative_index')
    rank_position = export.get('rank_position')
    if not all(isinstance(tensor, torch.Tensor) for tensor in (anchor_index, negative_index, rank_position)):
        return

    anchor_index = anchor_index.detach().long().reshape(-1).cpu()
    negative_index = negative_index.detach().long().reshape(-1).cpu()
    rank_position = rank_position.detach().long().reshape(-1).cpu()
    pair_count = int(min(anchor_index.numel(), negative_index.numel(), rank_position.numel()))
    if pair_count <= 0:
        return

    image_ids = batch.get('image_ids')
    if isinstance(image_ids, torch.Tensor):
        image_ids = image_ids.detach().long().reshape(-1).cpu()
    else:
        image_ids = None

    debug_dict = outputs.get('debug', {}) if isinstance(outputs.get('debug', {}), dict) else {}
    diag_cos_full = _to_python_float(debug_dict.get('diag_cos_full', 0.0), default=0.0)

    flat_tensors: Dict[str, torch.Tensor] = {}
    for key in (
        's_pos_host',
        's_neg_host',
        'margin_host',
        'margin_global',
        'margin_local',
        'gate_host',
        'gate_global',
        'gate_local',
        'proto_pair_signal',
        'proto_gate',
        'omega',
        'routing_entropy',
        'routing_top1_top2_gap',
    ):
        value = export.get(key)
        if isinstance(value, torch.Tensor):
            flat_tensors[key] = value.detach().reshape(-1).cpu()

    order = torch.arange(pair_count, dtype=torch.long)
    omega_values = flat_tensors.get('omega')
    if isinstance(omega_values, torch.Tensor) and omega_values.numel() >= pair_count:
        order = torch.argsort(omega_values[:pair_count].float(), descending=True)

    remaining = int(max_rows - len(rows))
    selected = order[:remaining]

    def _value_at(key: str, index: int, default: float = 0.0) -> float:
        tensor = flat_tensors.get(key)
        if not isinstance(tensor, torch.Tensor) or index >= tensor.numel():
            return float(default)
        return float(tensor[index].float().item())

    for selected_index in selected.tolist():
        anchor = int(anchor_index[selected_index].item())
        negative = int(negative_index[selected_index].item())

        if isinstance(image_ids, torch.Tensor) and 0 <= anchor < image_ids.numel():
            anchor_id = int(image_ids[anchor].item())
        else:
            anchor_id = anchor
        if isinstance(image_ids, torch.Tensor) and 0 <= negative < image_ids.numel():
            negative_id = int(image_ids[negative].item())
        else:
            negative_id = negative

        row = {
            'epoch': int(epoch),
            'step': int(step),
            'anchor_id': int(anchor_id),
            'negative_id': int(negative_id),
            'is_positive': int(anchor_id == negative_id),
            'rank_position': int(rank_position[selected_index].item()),
            's_pos_host': _value_at('s_pos_host', selected_index),
            's_neg_host': _value_at('s_neg_host', selected_index),
            'margin_host': _value_at('margin_host', selected_index),
            'margin_global': _value_at('margin_global', selected_index),
            'margin_local': _value_at('margin_local', selected_index),
            'gate_host': _value_at('gate_host', selected_index),
            'gate_global': _value_at('gate_global', selected_index),
            'gate_local': _value_at('gate_local', selected_index),
            'proto_pair_signal': _value_at('proto_pair_signal', selected_index),
            'proto_gate': _value_at('proto_gate', selected_index),
            'omega': _value_at('omega', selected_index),
            'routing_entropy': _value_at('routing_entropy', selected_index),
            'routing_top1_top2_gap': _value_at('routing_top1_top2_gap', selected_index),
            'diag_cos_full': float(diag_cos_full),
            'split': 'train',
            'method_variant': str(method_variant),
        }
        rows.append(row)


def _write_pairwise_hard_rows(
    *,
    output_dir: str,
    epoch: int,
    rows: List[Dict[str, object]],
    logger,
) -> None:
    if not rows:
        return
    export_dir = os.path.join(str(output_dir), 'pairwise_hard_samples')
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, f'pairwise_hard_samples_epoch_{int(epoch):04d}.csv')
    with open(export_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PAIRWISE_HARD_EXPORT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    logger.info('Exported %d pairwise hard samples to %s', len(rows), export_path)


def _format_ratio(trainable: int, total: int):
    percentage = (100.0 * float(trainable) / max(int(total), 1)) if int(total) > 0 else 0.0
    return f'{int(trainable)}/{int(total)} ({percentage:.2f}%)'


def _summarize_effective_lrs_by_logical_group(optimizer_group_summaries, logical_groups):
    lr_by_optimizer_group = {
        str(summary.get('name', '')): float(summary.get('lr', 0.0))
        for summary in optimizer_group_summaries
        if int(summary.get('tensor_count', 0)) > 0
    }
    formatted = []
    for logical_group in logical_groups:
        optimizer_group_names = LOGICAL_TO_OPTIMIZER_GROUPS.get(logical_group, tuple())
        matches = [
            (group_name, lr_by_optimizer_group[group_name])
            for group_name in optimizer_group_names
            if group_name in lr_by_optimizer_group
        ]
        if not matches:
            continue
        if len(matches) == 1:
            formatted.append(f'{logical_group}={matches[0][1]:.2e}')
            continue
        parts = ','.join(f'{name}:{lr:.2e}' for name, lr in matches)
        formatted.append(f'{logical_group}=[{parts}]')
    return ', '.join(formatted) if formatted else 'none'


def _resolve_display_lr(optimizer, scheduler) -> float:
    """Return a stable, host-relevant LR for logging dashboards and console output.

    `scheduler.get_lr()[0]` is tied to optimizer group order, which can start with
    prototype groups in joint training. For comparability with host-only runs, prefer
    host/backbone group LR when available, then fall back to group-0 LR.
    """
    lrs = list(scheduler.get_lr())
    if not lrs:
        return 0.0
    preferred_group_names = ('image_backbone', 'text_backbone', 'host_projectors', 'other')
    named_lrs = []
    for index, group in enumerate(optimizer.param_groups):
        if index >= len(lrs):
            break
        group_name = str(group.get('name', ''))
        named_lrs.append((group_name, float(lrs[index])))
    for preferred_name in preferred_group_names:
        for group_name, lr_value in named_lrs:
            if group_name == preferred_name:
                return lr_value
    return float(lrs[0])


def _log_scheduler_section_c(logger, optimizer, scheduler, epoch_label):
    logger.info('=== Section C: Scheduler-effective LR Snapshot (epoch=%s) ===', epoch_label)
    logger.info(
        'Display Lr selection policy: host-preferred scheduler-selected view (image_backbone -> text_backbone -> '
        'host_projectors -> other -> group_00 fallback).'
    )
    logger.info('display_lr=%.6g', _resolve_display_lr(optimizer, scheduler))
    for row in summarize_scheduler_effective_lrs(optimizer=optimizer, scheduler=scheduler):
        logger.info(
            '%s logical=%s scheduler_effective_lr=%.6g optimizer_initial_lr=%.6g optimizer_current_lr=%.6g',
            row.get('group_id'),
            row.get('name'),
            float(row.get('scheduler_effective_lr', 0.0)),
            float(row.get('optimizer_initial_lr', 0.0)),
            float(row.get('optimizer_current_lr', 0.0)),
        )


def _optimizer_has_fp16_trainable_params(optimizer) -> bool:
    for group in optimizer.param_groups:
        for parameter in group.get('params', []):
            if not isinstance(parameter, torch.Tensor):
                continue
            if not bool(parameter.requires_grad):
                continue
            if parameter.dtype == torch.float16:
                return True
    return False


def _maybe_disable_grad_scaler_for_fp16_params(scaler, optimizer, logger):
    if not scaler.is_enabled():
        return scaler
    if not _optimizer_has_fp16_trainable_params(optimizer):
        return scaler
    logger.warning(
        'Detected FP16 trainable optimizer parameters while AMP GradScaler is enabled. '
        'Disabling GradScaler to avoid "Attempting to unscale FP16 gradients".'
    )
    return torch.cuda.amp.GradScaler(enabled=False)


def _compute_eval_loss_metrics(model, val_loss_loader, args):
    if val_loss_loader is None:
        return {}
    metrics = {}
    counts = {}
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in val_loss_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                with build_autocast_context(args, device):
                    outputs = model(batch, return_debug=False, disable_proxy_losses=True)
                batch_size = int(batch['images'].shape[0])
                scalar_metrics = collect_scalar_metrics(
                    outputs,
                    include_debug_metrics=bool(getattr(args, 'log_debug_metrics', True)),
                )
                for key, value in scalar_metrics.items():
                    metrics[key] = metrics.get(key, 0.0) + (float(value) * batch_size)
                    counts[key] = counts.get(key, 0) + batch_size
    finally:
        if was_training:
            model.train()
    averaged = {}
    for key, total in metrics.items():
        count = counts.get(key, 0)
        if count > 0:
            averaged[key] = total / float(count)
    return averaged

def _make_meters():
    return {key: AverageMeter() for key in METER_KEYS}


def _meter_averages(meters):
    return {key: float(meter.avg) for key, meter in meters.items() if meter.count > 0}



def _named_parameters_for_logging(model):
    if hasattr(model, 'module'):
        return model.module.named_parameters()
    return model.named_parameters()



def _parameter_group_grad_norm(named_parameters, prefixes):
    total = 0.0
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        grad_norm = parameter.grad.detach().float().norm(2).item()
        total += grad_norm * grad_norm
    return total ** 0.5



def _collect_gradient_metrics(model):
    named_parameters = list(_named_parameters_for_logging(model))
    metrics = {
        'grad_norm_class_proxies': _parameter_group_grad_norm(
            named_parameters,
            ('prototype_head.losses.class_proxies', 'prototype_plugin.prototype_head.losses.class_proxies'),
        ),
        'grad_norm_image_projector': _parameter_group_grad_norm(
            named_parameters,
            (
                'prototype_head.image_projector',
                'prototype_head.proto_query_proj',
                'prototype_head.local_routing_adapter',
                'prototype_head.image_adapter',
                'prototype_plugin.prototype_head.image_projector',
                'prototype_plugin.prototype_head.proto_query_proj',
                'prototype_plugin.prototype_head.local_routing_adapter',
                'prototype_plugin.prototype_head.image_adapter',
                'host_head',
                'host_core.host_head',
            ),
        ),
        'grad_norm_text_projector': _parameter_group_grad_norm(
            named_parameters,
            (
                'prototype_head.text_projector',
                'prototype_head.text_adapter',
                'prototype_plugin.prototype_head.text_projector',
                'prototype_plugin.prototype_head.text_adapter',
                'host_head',
                'host_core.host_head',
            ),
        ),
        'grad_norm_prototype_bank': _parameter_group_grad_norm(
            named_parameters,
            ('prototype_head.prototype_bank', 'prototype_plugin.prototype_head.prototype_bank'),
        ),
        'grad_norm_image_backbone': _parameter_group_grad_norm(
            named_parameters,
            ('base_model.visual', 'host_core.base_model.visual'),
        ),
        'grad_norm_text_backbone': _parameter_group_grad_norm(
            named_parameters,
            (
                'base_model.transformer',
                'base_model.token_embedding',
                'base_model.positional_embedding',
                'base_model.ln_final',
                'base_model.text_projection',
                'host_core.base_model.transformer',
                'host_core.base_model.token_embedding',
                'host_core.base_model.positional_embedding',
                'host_core.base_model.ln_final',
                'host_core.base_model.text_projection',
            ),
        ),
    }
    total = 0.0
    for _, parameter in named_parameters:
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().float().norm(2).item()
        total += grad_norm * grad_norm
    metrics['grad_norm_total'] = total ** 0.5
    return metrics



def _collect_output_gradient_metrics(outputs, scale: float = 1.0):
    metrics = {}
    safe_scale = float(scale) if scale and scale > 0 else 1.0
    for output_key, metric_key in (
        ('z_v', 'grad_norm_image_projected_output'),
        ('z_t_hat_diag', 'grad_norm_surrogate_text_projected_output'),
        ('z_t_exact_diag', 'grad_norm_exact_text_projected_output'),
        ('surrogate_pairwise_logits', 'surrogate_retrieval_grad_norm'),
    ):
        tensor = outputs.get(output_key)
        if not isinstance(tensor, torch.Tensor) or tensor.grad is None:
            metrics[metric_key] = 0.0
            continue
        metrics[metric_key] = tensor.grad.detach().float().norm(2).item() / safe_scale
    return metrics


def _compute_grad_norm_from_loss(loss: torch.Tensor, parameters) -> float:
    if not isinstance(loss, torch.Tensor) or not bool(loss.requires_grad):
        return 0.0
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    total_sq = 0.0
    for grad in grads:
        if grad is None:
            continue
        grad_tensor = grad.detach()
        if grad_tensor.is_sparse:
            grad_tensor = grad_tensor.coalesce().values()
        total_sq += float(grad_tensor.float().pow(2).sum().item())
    return math.sqrt(total_sq)


def _collect_loss_grad_norm_metrics(outputs, parameters):
    metrics = {
        'train/grad_loss_norm/host': 0.0,
        'train/grad_loss_norm/diag': 0.0,
        'train/grad_loss_norm/hbr': 0.0,
        'train/grad_loss_norm/semantic_pbt': 0.0,
        'train/grad_loss_norm/semantic_hardneg_margin': 0.0,
        'train/grad_loss_norm/semantic_hosthard_weighted': 0.0,
    }
    if not parameters:
        return metrics

    loss_key_preferences = (
        ('train/grad_loss_norm/host', ('loss_host_weighted', 'loss_host')),
        ('train/grad_loss_norm/diag', ('loss_diag_weighted', 'loss_diag')),
        ('train/grad_loss_norm/hbr', ('loss_hbr_weighted', 'loss_hbr')),
        ('train/grad_loss_norm/semantic_pbt', ('loss_semantic_pbt_weighted', 'loss_semantic_pbt')),
        (
            'train/grad_loss_norm/semantic_hardneg_margin',
            ('loss_semantic_hardneg_margin_weighted', 'loss_semantic_hardneg_margin'),
        ),
        (
            'train/grad_loss_norm/semantic_hosthard_weighted',
            ('loss_semantic_hosthard_weighted_weighted', 'loss_semantic_hosthard_weighted'),
        ),
    )
    for metric_key, candidate_loss_keys in loss_key_preferences:
        selected_loss = None
        for loss_key in candidate_loss_keys:
            candidate = outputs.get(loss_key)
            if isinstance(candidate, torch.Tensor):
                selected_loss = candidate
                break
        if selected_loss is None:
            continue
        metrics[metric_key] = _compute_grad_norm_from_loss(selected_loss, parameters)
    return metrics


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _copy_optimizer_state(old_optimizer, new_optimizer):
    if old_optimizer is None:
        return
    old_state = old_optimizer.state_dict()
    old_param_groups = old_state.get('param_groups', [])
    old_states = old_state.get('state', {})
    if not old_param_groups:
        return

    old_param_order = []
    for group in old_param_groups:
        old_param_order.extend(group.get('params', []))

    old_param_tensors = []
    for group in old_optimizer.param_groups:
        old_param_tensors.extend(group.get('params', []))
    if len(old_param_tensors) != len(old_param_order):
        return

    index_to_param = {}
    for param_index, parameter in zip(old_param_order, old_param_tensors):
        index_to_param[param_index] = parameter

    tensor_state_by_id = {}
    for param_index, state_value in old_states.items():
        tensor = index_to_param.get(param_index)
        if tensor is None:
            continue
        tensor_state_by_id[id(tensor)] = state_value

    for group in new_optimizer.param_groups:
        for parameter in group.get('params', []):
            source_state = tensor_state_by_id.get(id(parameter))
            if source_state is None:
                continue
            new_optimizer.state[parameter] = source_state


def _rewind_scheduler_to_epoch(scheduler, completed_epochs: int):
    completed = max(int(completed_epochs), 0)
    if completed == 0:
        return
    scheduler.step(completed)


def _set_requires_grad_by_prefixes(model, prefixes, requires_grad: bool) -> int:
    touched = 0
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad = bool(requires_grad)
            touched += 1
    return touched


def _apply_runtime_mode_trainability(model, runtime_mode: str, logger) -> None:
    mode = normalize_runtime_mode(runtime_mode)
    host_prefixes = (
        'base_model.',
        'host_head.',
        'host_core.base_model.',
        'host_core.host_head.',
    )
    prototype_prefixes = (
        'prototype_head.',
        'prototype_plugin.prototype_head.',
    )
    legacy_fusion_prefixes = ('fusion_module.',)
    if mode == RUNTIME_MODE_JOINT_TRAINING:
        return
    if mode == RUNTIME_MODE_HOST_ONLY:
        frozen = _set_requires_grad_by_prefixes(model, prototype_prefixes + legacy_fusion_prefixes, False)
        logger.info('Runtime mode `%s`: froze prototype tensors=%d.', mode, frozen)
        return
    raise ValueError(f'Unsupported runtime mode after refactor: {mode!r}')


def _resolve_early_stopping_config(args) -> Dict[str, Any]:
    mode = str(getattr(args, 'early_stopping_mode', 'max') or 'max').strip().lower()
    if mode not in {'max', 'min'}:
        raise ValueError(
            f'early_stopping_mode must be one of ["max", "min"]. Got {mode!r}.'
        )
    patience = int(getattr(args, 'early_stopping_patience', 5))
    if patience <= 0:
        raise ValueError(f'early_stopping_patience must be a positive integer. Got {patience}.')
    start_epoch = int(getattr(args, 'early_stopping_start_epoch', 1))
    if start_epoch < 1:
        raise ValueError(f'early_stopping_start_epoch must be >= 1. Got {start_epoch}.')
    monitored_bucket_raw = getattr(args, 'early_stopping_monitored_bucket', 'host')
    monitored_bucket = None
    if monitored_bucket_raw is not None:
        monitored_bucket = str(monitored_bucket_raw).strip().lower()
        if monitored_bucket == '':
            monitored_bucket = None
    task_pattern_raw = getattr(args, 'early_stopping_monitored_task_pattern', None)
    monitored_task_pattern = None
    if task_pattern_raw is not None:
        monitored_task_pattern = str(task_pattern_raw).strip()
        if monitored_task_pattern == '':
            monitored_task_pattern = None
    return {
        'enabled': bool(getattr(args, 'early_stopping_enabled', False)),
        'metric': str(getattr(args, 'early_stopping_metric', 'R1') or 'R1'),
        'mode': mode,
        'patience': patience,
        'min_delta': float(getattr(args, 'early_stopping_min_delta', 0.0) or 0.0),
        'start_epoch': start_epoch,
        'monitored_bucket': monitored_bucket if monitored_bucket is not None else None,
        'monitored_task_pattern': monitored_task_pattern,
        'stop_on_nan': bool(getattr(args, 'early_stopping_stop_on_nan', False)),
    }


def _initialize_early_stopping_state() -> Dict[str, Any]:
    return {
        'best_value': None,
        'best_epoch': None,
        'best_row_name': None,
        'best_row_metadata': None,
        'bad_epochs': 0,
        'should_stop': False,
        'stop_reason': None,
    }


def _update_early_stopping_monitor(
    logger,
    eval_epoch: int,
    eval_result: Any,
    config: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    if not bool(config.get('enabled', False)):
        return state

    if int(eval_epoch) < int(config['start_epoch']):
        logger.info(
            'early_stop skipped: epoch=%d < start_epoch=%d',
            int(eval_epoch),
            int(config['start_epoch']),
        )
        return state

    monitored_rows = collect_monitored_eval_rows(
        eval_result,
        monitored_bucket=config.get('monitored_bucket', 'host'),
        monitored_task_pattern=config.get('monitored_task_pattern'),
    )
    summary = summarize_epoch_monitor(
        monitored_rows,
        metric_name=config.get('metric', 'R1'),
        mode=config.get('mode', 'max'),
    )

    improved = False
    epoch_best_value = summary.get('best_value')
    epoch_best_row_name = summary.get('best_row_name')
    num_rows_considered = int(summary.get('num_rows_considered', 0))
    num_rows_total = int(summary.get('num_rows_total', 0))
    num_invalid_rows = int(summary.get('num_invalid_rows', 0))

    if num_rows_considered <= 0:
        logger.warning(
            'early_stop no valid monitored rows at epoch=%d (matched_rows=%d invalid_rows=%d metric=%s bucket=%s pattern=%s).',
            int(eval_epoch),
            num_rows_total,
            num_invalid_rows,
            config.get('metric', 'R1'),
            config.get('monitored_bucket', 'host'),
            config.get('monitored_task_pattern'),
        )
        all_rows_invalid = num_rows_total > 0 and num_invalid_rows == num_rows_total
        if bool(config.get('stop_on_nan', False)) and all_rows_invalid:
            state['should_stop'] = True
            state['stop_reason'] = (
                f'All monitored rows are invalid for metric={config.get("metric", "R1")} at epoch={int(eval_epoch)} '
                f'and early_stopping_stop_on_nan=true.'
            )
        else:
            state['bad_epochs'] = int(state.get('bad_epochs', 0)) + 1
    else:
        best_value = state.get('best_value')
        if best_value is None:
            improved = True
        elif config.get('mode', 'max') == 'max':
            improved = float(epoch_best_value) > float(best_value) + float(config.get('min_delta', 0.0))
        else:
            improved = float(epoch_best_value) < float(best_value) - float(config.get('min_delta', 0.0))

        if improved:
            state['best_value'] = float(epoch_best_value)
            state['best_epoch'] = int(eval_epoch)
            state['best_row_name'] = str(epoch_best_row_name or '')
            state['best_row_metadata'] = summary.get('best_row_full_metadata')
            state['bad_epochs'] = 0
        else:
            state['bad_epochs'] = int(state.get('bad_epochs', 0)) + 1

    if not bool(state.get('should_stop', False)) and int(state.get('bad_epochs', 0)) >= int(config['patience']):
        state['should_stop'] = True
        state['stop_reason'] = (
            f'no improvement in best monitored {config.get("metric", "R1")} for '
            f'{int(state.get("bad_epochs", 0))} consecutive validation epochs'
        )

    logger.info(
        (
            'early_stop metric=%s mode=%s patience=%d min_delta=%.6f start_epoch=%d '
            'monitored_bucket=%s monitored_task_pattern=%s monitored_rows_count=%d '
            'epoch_best_value=%s epoch_best_row=%s global_best_value=%s global_best_epoch=%s '
            'bad_epochs=%d/%d improved=%s'
        ),
        config.get('metric', 'R1'),
        config.get('mode', 'max'),
        int(config['patience']),
        float(config.get('min_delta', 0.0)),
        int(config['start_epoch']),
        config.get('monitored_bucket', 'host'),
        config.get('monitored_task_pattern'),
        num_rows_considered,
        'None' if epoch_best_value is None else f'{float(epoch_best_value):.4f}',
        epoch_best_row_name,
        'None' if state.get('best_value') is None else f'{float(state["best_value"]):.4f}',
        state.get('best_epoch'),
        int(state.get('bad_epochs', 0)),
        int(config['patience']),
        bool(improved),
    )

    if bool(state.get('should_stop', False)):
        logger.warning(
            'Early stopping triggered at epoch %d: %s. Best %s=%.4f, best epoch=%s, best row=%s',
            int(eval_epoch),
            state.get('stop_reason', 'stopping condition reached'),
            config.get('metric', 'R1'),
            float(state['best_value']) if state.get('best_value') is not None else float('nan'),
            state.get('best_epoch'),
            state.get('best_row_name'),
        )
    return state


def _do_train_runtime(
    start_epoch,
    args,
    model,
    train_loader,
    evaluator,
    optimizer,
    scheduler,
    checkpointer,
    experiment_tracker: ExperimentTracker = None,
    eval_loss_loader=None,
    modular_checkpoint_manager=None,
    runtime_mode: str = None,
    resume_state: Dict[str, Any] = None,
):
    log_period = args.log_period
    eval_period = args.eval_period
    grad_clip = float(getattr(args, 'grad_clip', 0.0) or 0.0)
    device = getattr(args, 'device', 'cuda')
    num_epoch = args.num_epoch

    logger = logging.getLogger('pas.train')
    logger.info('start training')
    resolved_runtime_mode = normalize_runtime_mode(
        runtime_mode or resolve_runtime_mode_from_args(args, for_training=True)
    )
    logger.info('Trainer runtime_mode=%s', resolved_runtime_mode)
    runtime_model = _unwrap_model(model)
    if hasattr(runtime_model, 'set_runtime_mode'):
        runtime_model.set_runtime_mode(resolved_runtime_mode)
    _apply_runtime_mode_trainability(runtime_model, resolved_runtime_mode, logger)
    resume_bundle = resume_state if isinstance(resume_state, dict) else {}
    resume_optimizer_restored = bool(resume_bundle.get('optimizer_restored', False))
    resume_scheduler_restored = bool(resume_bundle.get('scheduler_restored', False))
    if resolved_runtime_mode != RUNTIME_MODE_JOINT_TRAINING and not resume_bundle:
        optimizer = build_optimizer(args, runtime_model)
        scheduler = build_lr_scheduler(args, optimizer)
        _rewind_scheduler_to_epoch(scheduler, start_epoch - 1)
        logger.info(
            'Rebuilt optimizer/scheduler for runtime_mode=%s to enforce explicit trainable boundary.',
            resolved_runtime_mode,
        )
    elif resolved_runtime_mode != RUNTIME_MODE_JOINT_TRAINING and resume_bundle:
        if not resume_optimizer_restored:
            optimizer = build_optimizer(args, runtime_model)
            logger.warning(
                'Resume checkpoint did not restore optimizer state; rebuilt optimizer from config for runtime_mode=%s.',
                resolved_runtime_mode,
            )
        if not resume_scheduler_restored:
            scheduler = build_lr_scheduler(args, optimizer)
            _rewind_scheduler_to_epoch(scheduler, start_epoch - 1)
            logger.warning(
                (
                    'Resume checkpoint did not restore scheduler state; rebuilt and rewound scheduler to '
                    'completed_epochs=%d for runtime_mode=%s.'
                ),
                int(start_epoch - 1),
                resolved_runtime_mode,
            )
        logger.info(
            (
                'Resume requested in runtime_mode=%s; keeping currently loaded optimizer/scheduler objects '
                '(optimizer_restored=%s scheduler_restored=%s).'
            ),
            resolved_runtime_mode,
            resume_optimizer_restored,
            resume_scheduler_restored,
        )
    checkpointer.optimizer = optimizer
    checkpointer.scheduler = scheduler
    _log_scheduler_section_c(logger, optimizer, scheduler, epoch_label='train_start')
    if getattr(args, 'prototype_selection_metric', None):
        logger.warning(
            'training.prototype_selection_metric is deprecated by checkpointing.metric/checkpointing.save.*. '
            'Use checkpointing for per-group best/latest artifacts.'
        )
    if bool(getattr(args, 'amp', False)) and not is_cuda_device(device):
        raise ValueError('training.amp=true requires a CUDA device.')
    scaler = build_grad_scaler(args, device)
    scaler = _maybe_disable_grad_scaler_for_fp16_params(scaler, optimizer, logger)
    resume_scaler_state = resume_bundle.get('scaler_state_dict')
    if isinstance(resume_scaler_state, dict):
        if scaler is not None and hasattr(scaler, 'load_state_dict') and scaler.is_enabled():
            try:
                scaler.load_state_dict(resume_scaler_state)
                logger.info('Restored AMP GradScaler state from resume checkpoint.')
            except Exception as exc:
                logger.warning('Failed to restore AMP GradScaler state from resume checkpoint: %s', exc)
        else:
            logger.warning(
                'Resume checkpoint contains AMP scaler state, but active scaler is disabled; skipping scaler restore.'
            )
    logger.info(
        'Precision config: backbone_precision=%s, prototype_precision=%s, amp=%s, amp_dtype=%s',
        getattr(args, 'backbone_precision', 'fp16'),
        getattr(args, 'prototype_precision', 'fp32'),
        is_amp_enabled(args, device),
        canonicalize_amp_dtype(getattr(args, 'amp_dtype', 'fp16')),
    )
    meters = _make_meters()
    wandb_interval_meters = _make_meters()
    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    current_steps = 0
    best_epoch = start_epoch
    last_epoch = start_epoch - 1
    metric_name_for_ckpt = (
        str(getattr(modular_checkpoint_manager, 'metric_name', 'R1')) if modular_checkpoint_manager is not None else 'R1'
    )
    metric_mode_for_ckpt = (
        str(getattr(modular_checkpoint_manager, 'metric_mode', 'max')) if modular_checkpoint_manager is not None else 'max'
    )
    best_metric_state_current = {
        'name': metric_name_for_ckpt,
        'mode': metric_mode_for_ckpt,
        'value': float(best_top1),
        'best_epoch': int(best_epoch),
        'selected_row': None,
        'source_row': None,
        'display_row': None,
        'authority_bucket': None,
        'selection_reason': None,
    }
    latest_metric_state_current = {
        'name': metric_name_for_ckpt,
        'mode': metric_mode_for_ckpt,
        'value': None,
        'epoch': None,
        'selected_row': None,
        'source_row': None,
        'display_row': None,
        'authority_bucket': None,
        'selection_reason': None,
    }
    early_stopping_cfg = _resolve_early_stopping_config(args)
    early_stopping_state = _initialize_early_stopping_state()
    resume_training_state = resume_bundle.get('training_state', {}) if isinstance(resume_bundle, dict) else {}
    if isinstance(resume_training_state, dict) and resume_training_state:
        current_steps = int(resume_training_state.get('global_step', resume_bundle.get('global_step', 0) or 0))
        epoch_completed = int(
            resume_training_state.get(
                'epoch_completed',
                resume_training_state.get('epoch', max(start_epoch - 1, 0)),
            )
            or 0
        )
        last_epoch = max(epoch_completed, start_epoch - 1)
        best_metric_payload = resume_training_state.get('best_metric_state')
        if isinstance(best_metric_payload, dict):
            best_metric_state_current = copy.deepcopy(best_metric_payload)
            best_top1 = float(best_metric_payload.get('value', best_top1) or best_top1)
            best_epoch = int(best_metric_payload.get('best_epoch', best_epoch) or best_epoch)
        elif resume_training_state.get('best_top1') is not None:
            best_top1 = float(resume_training_state.get('best_top1'))
            best_epoch = int(resume_training_state.get('best_epoch', best_epoch) or best_epoch)
        latest_metric_payload = resume_training_state.get('latest_metric_state')
        if isinstance(latest_metric_payload, dict):
            latest_metric_state_current = copy.deepcopy(latest_metric_payload)
        early_stopping_payload = resume_training_state.get('early_stopping_state')
        if isinstance(early_stopping_payload, dict):
            merged_state = _initialize_early_stopping_state()
            merged_state.update(copy.deepcopy(early_stopping_payload))
            early_stopping_state = merged_state
        modular_best_payload = resume_training_state.get('modular_best_metric_value_by_group')
        if (
            modular_checkpoint_manager is not None
            and isinstance(modular_best_payload, dict)
            and hasattr(modular_checkpoint_manager, 'best_metric_value_by_group')
        ):
            modular_checkpoint_manager.best_metric_value_by_group = {
                str(key): float(value) for key, value in modular_best_payload.items()
            }
        logger.info(
            (
                'Applied training resume state: start_epoch=%d resumed_global_step=%d resumed_epoch_completed=%d '
                'best_%s=%.4f best_epoch=%d early_stop_bad_epochs=%d'
            ),
            int(start_epoch),
            int(current_steps),
            int(epoch_completed),
            metric_name_for_ckpt,
            float(best_top1),
            int(best_epoch),
            int(early_stopping_state.get('bad_epochs', 0)),
        )
    elif resume_bundle:
        logger.warning(
            'Resume requested but checkpoint has no structured training_state. '
            'Falling back to start_epoch=%d and global_step=%d.',
            int(start_epoch),
            int(resume_bundle.get('global_step', 0) or 0),
        )
        current_steps = int(resume_bundle.get('global_step', 0) or 0)
    if early_stopping_cfg['enabled']:
        logger.info(
            'Early stopping enabled: metric=%s mode=%s patience=%d min_delta=%.6f start_epoch=%d bucket=%s pattern=%s stop_on_nan=%s',
            early_stopping_cfg['metric'],
            early_stopping_cfg['mode'],
            int(early_stopping_cfg['patience']),
            float(early_stopping_cfg['min_delta']),
            int(early_stopping_cfg['start_epoch']),
            early_stopping_cfg['monitored_bucket'],
            early_stopping_cfg['monitored_task_pattern'],
            bool(early_stopping_cfg['stop_on_nan']),
        )
    else:
        logger.info('Early stopping disabled.')
    log_debug_metrics = bool(getattr(args, 'log_debug_metrics', True))
    export_pairwise_hard_samples = bool(getattr(args, 'export_pairwise_hard_samples', False)) and get_rank() == 0
    pairwise_export_max_rows_per_epoch = max(int(getattr(args, 'pairwise_export_max_rows_per_epoch', 2000)), 0)
    method_variant = _resolve_method_variant(args, resolved_runtime_mode)
    coverage_tracker = RoutingCoverageTracker() if log_debug_metrics else None
    if resolved_runtime_mode == RUNTIME_MODE_JOINT_TRAINING:
        freeze_schedule_phases = parse_freeze_schedule_config(
            getattr(args, 'freeze_schedule', None),
            num_epoch=num_epoch,
        )
    else:
        if getattr(args, 'freeze_schedule', None):
            logger.warning(
                'Ignoring training.freeze_schedule in runtime_mode=%s; mode routing defines semantics first.',
                resolved_runtime_mode,
            )
        freeze_schedule_phases = []
    active_freeze_phase_name = None
    missing_phase_warning_emitted = False
    if freeze_schedule_phases:
        schedule_preview = ', '.join(
            f'{phase.name}[{phase.epoch_start}-{phase.epoch_end}]' for phase in freeze_schedule_phases
        )
        logger.info('Using training.freeze_schedule phases: %s', schedule_preview)
    total_training_steps = max(int(num_epoch) * max(len(train_loader), 1), 1)
    wandb_log_interval = max(getattr(args, 'wandb_log_interval', log_period), 1)

    def _save_training_resume_checkpoint(name: str, checkpoint_kind: str, epoch_completed: int, iteration_in_epoch: int = 0):
        if get_rank() != 0:
            return
        config_snapshot = getattr(args, 'config_data', None)
        modular_best_values = {}
        if modular_checkpoint_manager is not None and hasattr(modular_checkpoint_manager, 'best_metric_value_by_group'):
            modular_best_values = dict(modular_checkpoint_manager.best_metric_value_by_group)
        additional_state = {
            'last_epoch': int(epoch_completed),
            'runtime_mode': str(resolved_runtime_mode),
            'resume_source_path': str(getattr(args, 'resume_ckpt_file', '') or ''),
        }
        checkpointer.save_training_checkpoint(
            name=name,
            epoch=int(epoch_completed),
            global_step=int(current_steps),
            iteration_in_epoch=int(iteration_in_epoch),
            checkpoint_kind=str(checkpoint_kind),
            metric_name=str(metric_name_for_ckpt),
            metric_mode=str(metric_mode_for_ckpt),
            best_metric_state=copy.deepcopy(best_metric_state_current),
            latest_metric_state=copy.deepcopy(latest_metric_state_current),
            early_stopping_state=copy.deepcopy(early_stopping_state),
            modular_best_metric_value_by_group=modular_best_values,
            scaler=scaler,
            config_snapshot=config_snapshot if isinstance(config_snapshot, dict) else None,
            include_rng_state=True,
            additional_training_state=additional_state,
        )

    def _run_validation(eval_epoch: int) -> float:
        nonlocal latest_metric_state_current
        logger.info('Validation Results - Epoch: {}'.format(eval_epoch))
        eval_loss_metrics = _compute_eval_loss_metrics(
            model.module if args.distributed else model,
            eval_loss_loader,
            args,
        )
        eval_loss_total = eval_loss_metrics.get('loss_total')
        if eval_loss_total is not None:
            logger.info('Eval total loss: %.4f', eval_loss_total)
        if args.distributed:
            top1_score = evaluator.eval(model.module.eval())
        else:
            top1_score = evaluator.eval(model.eval())
        torch.cuda.empty_cache()
        authority_context = dict(getattr(evaluator, 'latest_authority', {}) or {})
        selected_display_row = str(
            evaluator.latest_metrics.get(
                'val/top1_display_row',
                evaluator.latest_metrics.get('val/top1_row', ''),
            )
            or ''
        ) or None
        selected_source_row = str(evaluator.latest_metrics.get('val/top1_source_row', '') or '') or None
        selected_metric_row = (
            str(authority_context.get('source_row', '') or '').strip()
            or selected_source_row
            or selected_display_row
        )
        authority_bucket = authority_context.get('selected_source_role')
        latest_metric_state_current = {
            'name': metric_name_for_ckpt,
            'mode': metric_mode_for_ckpt,
            'value': float(top1_score),
            'epoch': int(eval_epoch),
            'selected_row': selected_metric_row,
            'source_row': selected_source_row,
            'display_row': selected_display_row,
            'authority_bucket': authority_bucket,
            'selection_reason': 'authority_source_row' if authority_context.get('source_row') else 'display_row_fallback',
        }
        if experiment_tracker is not None:
            validation_metrics = build_validation_metrics(eval_epoch, evaluator=evaluator, loss_metrics=eval_loss_metrics)
            experiment_tracker.log(validation_metrics)
        if modular_checkpoint_manager is not None:
            model_for_group_ckpt = model.module if hasattr(model, 'module') else model
            if (
                selected_display_row is not None
                and selected_source_row is not None
                and selected_display_row != selected_source_row
            ):
                logger.warning(
                    'Checkpoint row provenance mismatch: display_row=%s source_row=%s (authority row uses source).',
                    selected_display_row,
                    selected_source_row,
                )
            modular_checkpoint_manager.save_latest(
                model=model_for_group_ckpt,
                epoch=eval_epoch,
                global_step=current_steps,
                metric_value=float(top1_score),
                metric_row=selected_metric_row,
                metric_display_row=selected_display_row,
                metric_source_row=selected_source_row,
                authority_context=authority_context,
            )
            modular_checkpoint_manager.save_best_if_improved(
                model=model_for_group_ckpt,
                epoch=eval_epoch,
                global_step=current_steps,
                metric_value=float(top1_score),
                metric_row=selected_metric_row,
                metric_display_row=selected_display_row,
                metric_source_row=selected_source_row,
                authority_context=authority_context,
            )
        return float(top1_score)

    # Match the original ITSELF workflow: one validation pass before any updates.
    if get_rank() == 0 and start_epoch <= 1:
        top1_epoch0 = _run_validation(eval_epoch=0)
        if best_top1 < top1_epoch0:
            best_top1 = top1_epoch0
            best_epoch = 0
            best_metric_state_current = {
                **copy.deepcopy(latest_metric_state_current),
                'name': metric_name_for_ckpt,
                'mode': metric_mode_for_ckpt,
                'value': float(best_top1),
                'best_epoch': int(best_epoch),
            }

    for epoch in range(start_epoch, num_epoch + 1):
        _log_scheduler_section_c(logger, optimizer, scheduler, epoch_label=f'epoch_start_{epoch}')
        active_phase = get_active_phase(freeze_schedule_phases, epoch)
        if active_phase is not None and active_phase.name != active_freeze_phase_name:
            runtime_model = _unwrap_model(model)
            apply_phase_trainability(runtime_model, active_phase)
            applied_loss_weights = apply_loss_weight_overrides(runtime_model, args, active_phase.loss_weights)

            previous_optimizer = optimizer
            optimizer = build_optimizer(args, runtime_model)
            _copy_optimizer_state(previous_optimizer, optimizer)
            lr_override_summary = apply_optimizer_lr_overrides(optimizer, active_phase.lr_overrides)
            scaler = _maybe_disable_grad_scaler_for_fp16_params(scaler, optimizer, logger)

            scheduler = build_lr_scheduler(args, optimizer)
            _rewind_scheduler_to_epoch(scheduler, epoch - 1)
            checkpointer.optimizer = optimizer
            checkpointer.scheduler = scheduler
            _log_scheduler_section_c(logger, optimizer, scheduler, epoch_label=f'phase_activate_{epoch}')

            optimizer_groups = summarize_optimizer_param_groups(optimizer)
            logger.info('Activated freeze phase `%s` at epoch %d.', active_phase.name, epoch)
            logger.info(
                'Phase groups: trainable=%s frozen=%s loss_weights=%s',
                list(active_phase.trainable_groups),
                list(active_phase.frozen_groups),
                applied_loss_weights,
            )
            phase_groups = tuple(sorted(set(active_phase.trainable_groups + active_phase.frozen_groups)))
            model_total_params = sum(int(parameter.numel()) for parameter in runtime_model.parameters())
            model_trainable_params = sum(
                int(parameter.numel()) for parameter in runtime_model.parameters() if parameter.requires_grad
            )
            if phase_groups:
                snapshot = get_group_trainability_snapshot(runtime_model, groups=phase_groups)
                portion_entries = []
                for group_name in phase_groups:
                    group_stats = snapshot.get(group_name, {})
                    group_total = int(group_stats.get('total_params', 0))
                    group_trainable = int(group_stats.get('trainable_params', 0))
                    if group_total > 0:
                        portion_entries.append(f'{group_name}={_format_ratio(group_trainable, group_total)}')
                logger.info(
                    'Phase trainability portions: global=%s groups={%s}',
                    _format_ratio(model_trainable_params, model_total_params),
                    ', '.join(portion_entries) if portion_entries else 'none',
                )
                for group_name in active_phase.frozen_groups:
                    group_stats = snapshot.get(group_name, {})
                    if int(group_stats.get('trainable_tensors', 0)) > 0:
                        logger.warning(
                            'Freeze schedule requested frozen group `%s`, but it still has trainable tensors: %s',
                            group_name,
                            group_stats,
                        )
            if active_phase.lr_overrides:
                missing_override_hits = sorted(
                    group_name for group_name, hit_count in lr_override_summary.items() if int(hit_count) <= 0
                )
                if missing_override_hits:
                    logger.warning(
                        'Phase `%s` lr_overrides did not hit optimizer groups for: %s',
                        active_phase.name,
                        missing_override_hits,
                    )
            lr_scope_groups = phase_groups if phase_groups else tuple(LOGICAL_TO_OPTIMIZER_GROUPS.keys())
            logger.info(
                'Phase effective lrs: %s',
                _summarize_effective_lrs_by_logical_group(optimizer_groups, lr_scope_groups),
            )
            active_freeze_phase_name = active_phase.name
        elif freeze_schedule_phases and active_phase is None and not missing_phase_warning_emitted:
            logger.warning(
                'No freeze-schedule phase covers epoch %d. Training will continue with the previous trainability/lr state.',
                epoch,
            )
            missing_phase_warning_emitted = True

        last_epoch = epoch
        start_time = time.time()
        pairwise_hard_rows: List[Dict[str, object]] = []
        for meter in meters.values():
            meter.reset()

        model.train()
        for n_iter, batch in enumerate(train_loader):
            current_steps += 1
            should_log_wandb_step = (
                experiment_tracker is not None
                and get_rank() == 0
                and current_steps % wandb_log_interval == 0
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with build_autocast_context(args, device):
                outputs = model(
                    batch,
                    epoch=epoch,
                    current_step=current_steps,
                    total_steps=total_training_steps,
                    disable_proxy_losses=False,
                )
                total_loss = outputs['loss_total']
            grad_loss_norm_metrics = None
            if should_log_wandb_step:
                trainable_parameters = tuple(
                    parameter for parameter in model.parameters() if isinstance(parameter, torch.Tensor) and parameter.requires_grad
                )
                grad_loss_norm_metrics = _collect_loss_grad_norm_metrics(outputs, trainable_parameters)

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                if log_debug_metrics and isinstance(outputs.get('debug'), dict):
                    outputs['debug'].update(_collect_gradient_metrics(model))
                    outputs['debug'].update(_collect_output_gradient_metrics(outputs, scale=scaler.get_scale()))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if log_debug_metrics and isinstance(outputs.get('debug'), dict):
                    outputs['debug'].update(_collect_gradient_metrics(model))
                    outputs['debug'].update(_collect_output_gradient_metrics(outputs))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            synchronize()

            if coverage_tracker is not None:
                if not isinstance(outputs.get('debug'), dict):
                    outputs['debug'] = {}
                alpha = outputs.get('alpha')
                if isinstance(alpha, torch.Tensor) and alpha.ndim == 2 and alpha.size(1) > 0:
                    # Cross-batch coverage uses the existing routing weights and never feeds back into training.
                    coverage_tracker.update(alpha)
                    outputs['debug'].update(coverage_tracker.get_debug_metrics())

            if log_debug_metrics:
                core_model = model.module if hasattr(model, 'module') else model
                prototype_enabled = bool(getattr(core_model, 'use_prototype_branch', False))
                debug_dict = outputs.get('debug') if isinstance(outputs.get('debug'), dict) else {}
                prototype_active = bool(float(debug_dict.get('prototype_branch_active', 1.0)) > 0.5)
                labels = batch.get('pids')
                routing_probs = outputs.get('alpha')
                prototype_bank = resolve_prototype_bank_tensor(core_model)
                if (
                    prototype_enabled
                    and prototype_active
                    and isinstance(debug_dict, dict)
                    and isinstance(labels, torch.Tensor)
                    and labels.ndim == 1
                    and isinstance(routing_probs, torch.Tensor)
                    and routing_probs.ndim == 2
                    and routing_probs.size(0) == labels.size(0)
                    and routing_probs.size(1) > 0
                    and isinstance(prototype_bank, torch.Tensor)
                    and prototype_bank.ndim == 2
                    and prototype_bank.size(0) > 0
                ):
                    with torch.no_grad():
                        proto_ids = routing_probs.detach().argmax(dim=1)
                        prototype_metrics, _prototype_sanity = collect_prototype_debug_metrics(
                            proto_ids=proto_ids,
                            labels=labels,
                            routing_probs=routing_probs,
                            prototype_bank=prototype_bank,
                        )
                        debug_dict.update(prototype_metrics)

            if export_pairwise_hard_samples and pairwise_export_max_rows_per_epoch > 0:
                _append_hbr_pairwise_rows(
                    rows=pairwise_hard_rows,
                    outputs=outputs,
                    batch=batch,
                    epoch=epoch,
                    step=current_steps,
                    max_rows=pairwise_export_max_rows_per_epoch,
                    method_variant=method_variant,
                )

            scalar_metrics = collect_scalar_metrics(outputs, include_debug_metrics=log_debug_metrics)
            batch_size = batch['images'].shape[0]
            for key, meter in meters.items():
                value = scalar_metrics.get(key)
                if value is not None:
                    meter.update(value, batch_size)
                    wandb_interval_meters[key].update(value, batch_size)

            if should_log_wandb_step:
                averaged_metrics = _meter_averages(wandb_interval_meters)
                display_lr = _resolve_display_lr(optimizer, scheduler)
                train_metrics = build_train_metrics_from_scalars(
                    epoch=epoch,
                    step=current_steps,
                    scalar_metrics=averaged_metrics,
                    lr=display_lr,
                )
                if grad_loss_norm_metrics is not None:
                    train_metrics.update(grad_loss_norm_metrics)
                experiment_tracker.log(train_metrics)
                for meter in wandb_interval_meters.values():
                    meter.reset()

            if (n_iter + 1) % log_period == 0:
                info = [f'Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]']
                for key in CONSOLE_LOSS_LOG_KEYS:
                    meter = meters[key]
                    if meter.count > 0:
                        info.append(f'{key}: {meter.avg:.4f}')
                display_lr = _resolve_display_lr(optimizer, scheduler)
                info.append(f'Display Lr: {display_lr:.2e}')
                logger.info(', '.join(info))

        display_lr = _resolve_display_lr(optimizer, scheduler)
        tb_writer.add_scalar('lr', display_lr, epoch)
        for key, meter in meters.items():
            if meter.count > 0:
                tb_writer.add_scalar(key, meter.avg, epoch)

        if experiment_tracker is not None and get_rank() == 0:
            averaged_metrics = _meter_averages(meters)
            experiment_tracker.log(
                build_train_metrics_from_scalars(
                    epoch=epoch,
                    step=current_steps,
                    scalar_metrics=averaged_metrics,
                    lr=display_lr,
                )
            )
        if coverage_tracker is not None:
            coverage_tracker.reset_epoch()
        if export_pairwise_hard_samples and pairwise_export_max_rows_per_epoch > 0:
            _write_pairwise_hard_rows(
                output_dir=args.output_dir,
                epoch=epoch,
                rows=pairwise_hard_rows,
                logger=logger,
            )

        scheduler.step()
        if get_rank() == 0:
            elapsed = time.time() - start_time
            time_per_batch = elapsed / max(n_iter + 1, 1)
            logger.info(
                'Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'.format(
                    epoch,
                    time_per_batch,
                    train_loader.batch_size / max(time_per_batch, 1e-12),
                )
            )

        if epoch % eval_period == 0 and get_rank() == 0:
            top1 = _run_validation(eval_epoch=epoch)
            improved = False
            if best_top1 < top1:
                best_top1 = top1
                best_epoch = epoch
                improved = True
                best_metric_state_current = {
                    **copy.deepcopy(latest_metric_state_current),
                    'name': metric_name_for_ckpt,
                    'mode': metric_mode_for_ckpt,
                    'value': float(best_top1),
                    'best_epoch': int(best_epoch),
                }
            early_stopping_state = _update_early_stopping_monitor(
                logger=logger,
                eval_epoch=epoch,
                eval_result=evaluator,
                config=early_stopping_cfg,
                state=early_stopping_state,
            )
            _save_training_resume_checkpoint(
                name='checkpoint_training_latest',
                checkpoint_kind='latest',
                epoch_completed=epoch,
                iteration_in_epoch=0,
            )
            _save_training_resume_checkpoint(
                name='last',
                checkpoint_kind='latest',
                epoch_completed=epoch,
                iteration_in_epoch=0,
            )
            if improved:
                _save_training_resume_checkpoint(
                    name='checkpoint_training_best',
                    checkpoint_kind='best',
                    epoch_completed=epoch,
                    iteration_in_epoch=0,
                )
                _save_training_resume_checkpoint(
                    name='best',
                    checkpoint_kind='best',
                    epoch_completed=epoch,
                    iteration_in_epoch=0,
                )
            if early_stopping_state.get('should_stop', False):
                break

    if get_rank() == 0:
        if last_epoch >= start_epoch:
            _save_training_resume_checkpoint(
                name='checkpoint_training_latest',
                checkpoint_kind='latest',
                epoch_completed=last_epoch,
                iteration_in_epoch=0,
            )
            _save_training_resume_checkpoint(
                name='last',
                checkpoint_kind='last',
                epoch_completed=last_epoch,
                iteration_in_epoch=0,
            )
        logger.info(f'best R1: {best_top1} at epoch {best_epoch}')

    tb_writer.close()


def train_host_core(*args, **kwargs):
    kwargs['runtime_mode'] = RUNTIME_MODE_HOST_ONLY
    return _do_train_runtime(*args, **kwargs)


def train_prototype_external(*args, **kwargs):
    kwargs['runtime_mode'] = RUNTIME_MODE_JOINT_TRAINING
    return _do_train_runtime(*args, **kwargs)


def train_joint(*args, **kwargs):
    kwargs['runtime_mode'] = RUNTIME_MODE_JOINT_TRAINING
    return _do_train_runtime(*args, **kwargs)


def do_train(
    start_epoch,
    args,
    model,
    train_loader,
    evaluator,
    optimizer,
    scheduler,
    checkpointer,
    experiment_tracker: ExperimentTracker = None,
    eval_loss_loader=None,
    modular_checkpoint_manager=None,
    resume_state: Dict[str, Any] = None,
):
    resolved_runtime_mode = resolve_runtime_mode_from_args(args, for_training=True)
    if resolved_runtime_mode == RUNTIME_MODE_HOST_ONLY:
        return train_host_core(
            start_epoch,
            args,
            model,
            train_loader,
            evaluator,
            optimizer,
            scheduler,
            checkpointer,
            experiment_tracker=experiment_tracker,
            eval_loss_loader=eval_loss_loader,
            modular_checkpoint_manager=modular_checkpoint_manager,
            resume_state=resume_state,
        )
    return train_joint(
        start_epoch,
        args,
        model,
        train_loader,
        evaluator,
        optimizer,
        scheduler,
        checkpointer,
        experiment_tracker=experiment_tracker,
        eval_loss_loader=eval_loss_loader,
        modular_checkpoint_manager=modular_checkpoint_manager,
        resume_state=resume_state,
    )


def do_inference(model, test_img_loader, test_txt_loader, args):
    logger = logging.getLogger('pas.eval')
    logger.info('Enter inferencing')

    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    _ = evaluator.eval(model.eval())
