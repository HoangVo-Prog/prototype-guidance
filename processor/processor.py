import copy
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
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
from utils.precision import build_autocast_context, build_grad_scaler, is_cuda_device
from utils.repro import array_hash, tensor_hash


METER_KEYS = ('loss_total',) + tuple(key for key in TRACKED_SCALAR_KEYS if key != 'loss_total')
_CONSOLE_LOSS_SKIP_KEYS = {
    # Keep canonical naming in console logs.
    'loss_proto_total',  # alias of loss_proto in current prototype path
}
CONSOLE_LOSS_LOG_KEYS = tuple(
    key for key in TRAIN_LOSS_KEYS if key in METER_KEYS and key not in _CONSOLE_LOSS_SKIP_KEYS
)


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
        'train/grad_loss_norm/semantic_pbt': 0.0,
        'train/grad_loss_norm/semantic_hardneg_margin': 0.0,
        'train/grad_loss_norm/semantic_hosthard_weighted': 0.0,
    }
    if not parameters:
        return metrics

    loss_key_preferences = (
        ('train/grad_loss_norm/host', ('loss_host_weighted', 'loss_host')),
        ('train/grad_loss_norm/diag', ('loss_diag_weighted', 'loss_diag')),
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


def _set_epoch_on_loader_sampler(train_loader, epoch, logger=None, enabled: bool = True):
    if not enabled:
        return
    sampler = getattr(train_loader, 'sampler', None)
    if sampler is not None and hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(int(epoch))
        if logger is not None:
            logger.debug('REPRO_DEBUG sampler.set_epoch epoch=%d sampler=%s', int(epoch), type(sampler).__name__)
        return
    batch_sampler = getattr(train_loader, 'batch_sampler', None)
    nested_sampler = getattr(batch_sampler, 'sampler', None) if batch_sampler is not None else None
    if nested_sampler is not None and hasattr(nested_sampler, 'set_epoch'):
        nested_sampler.set_epoch(int(epoch))
        if logger is not None:
            logger.debug('REPRO_DEBUG batch_sampler.sampler.set_epoch epoch=%d sampler=%s', int(epoch), type(nested_sampler).__name__)


def _maybe_recompute_prototypes_from_deterministic_cache(model, train_loader, args, epoch, current_step, logger):
    if not (bool(getattr(args, 'repro_enabled', False)) and bool(getattr(args, 'repro_proto_deterministic_recompute', True))):
        return
    proto_loader = getattr(train_loader, 'proto_recompute_loader', None)
    if proto_loader is None:
        return
    runtime_model = _unwrap_model(model)
    proto_head = getattr(runtime_model, 'prototype_head', None)
    if proto_head is None:
        return
    device = next(runtime_model.parameters()).device
    was_training = runtime_model.training
    if bool(getattr(args, 'repro_proto_eval_mode_recompute', True)):
        runtime_model.eval()
    ids, feats = [], []
    def _extract_projected_pooled_images(images_tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(runtime_model, 'extract_image_features'):
            image_out = runtime_model.extract_image_features(images_tensor)
            return image_out.projected_pooled
        if hasattr(runtime_model, 'host_core') and hasattr(runtime_model.host_core, 'extract_image_features'):
            image_out = runtime_model.host_core.extract_image_features(images_tensor)
            return image_out.projected_pooled
        raise AttributeError(f'{type(runtime_model).__name__} has no supported image-feature extractor.')

    def _cast_proto_dtype(x: torch.Tensor) -> torch.Tensor:
        if hasattr(runtime_model, '_cast_to_prototype_dtype'):
            return runtime_model._cast_to_prototype_dtype(x)
        return x

    with torch.no_grad():
        for batch in proto_loader:
            images = batch['images'].to(device)
            projected_pooled = _extract_projected_pooled_images(images)
            image_emb = _cast_proto_dtype(projected_pooled)
            semantic_features = proto_head.image_adapter(image_emb.detach()).detach().float().cpu()
            sample_ids = batch.get('index', batch.get('image_ids'))
            ids.append(sample_ids.detach().cpu().reshape(-1))
            feats.append(semantic_features)
    if was_training:
        runtime_model.train()
    if not feats:
        return
    ids_tensor = torch.cat(ids, dim=0)
    feat_tensor = torch.cat(feats, dim=0)
    if bool(getattr(args, 'repro_proto_sort_cache_by_id', True)):
        order = torch.argsort(ids_tensor, stable=True)
        ids_tensor = ids_tensor.index_select(0, order)
        feat_tensor = feat_tensor.index_select(0, order)
    proto_head.get_prototype_context(
        return_debug=False,
        epoch=epoch,
        current_step=current_step,
        semantic_recompute_features=feat_tensor.to(device=device),
        semantic_recompute_ids=ids_tensor.to(device=device),
    )
    if bool(getattr(args, 'repro_proto_hash_logging', True)):
        logger.info(
            'REPRO_DEBUG proto_cache step=%d n=%d ids_hash=%s feat_hash=%s first10_ids=%s',
            int(current_step),
            int(ids_tensor.numel()),
            tensor_hash(ids_tensor, name='proto_ids'),
            tensor_hash(feat_tensor, name='proto_feats'),
            ids_tensor[:10].tolist(),
        )


def _normalize_adaptive_k_candidates(raw_candidates) -> List[int]:
    if raw_candidates is None:
        return [16, 32, 64]
    if not isinstance(raw_candidates, (list, tuple)):
        raw_candidates = [raw_candidates]
    candidates: List[int] = []
    for raw_candidate in raw_candidates:
        candidate = int(raw_candidate)
        if candidate <= 0:
            continue
        if candidate in candidates:
            continue
        candidates.append(candidate)
    return candidates


def _resolve_adaptive_k_config(args) -> Dict[str, Any]:
    semantic_schedule = str(getattr(args, 'semantic_recompute_schedule', 'epoch') or 'epoch').lower()
    semantic_interval = max(int(getattr(args, 'semantic_recompute_interval', 1)), 1)
    recompute_schedule = str(getattr(args, 'adaptive_k_recompute_schedule', 'semantic') or 'semantic').lower()
    if recompute_schedule == 'semantic':
        recompute_schedule = semantic_schedule
    recompute_interval_raw = getattr(args, 'adaptive_k_recompute_interval', None)
    recompute_interval = semantic_interval if recompute_interval_raw in (None, '') else max(int(recompute_interval_raw), 1)
    return {
        'enabled': bool(getattr(args, 'adaptive_k_enabled', False)),
        'method': str(getattr(args, 'adaptive_k_method', 'spb') or 'spb').lower(),
        'select_once': bool(getattr(args, 'adaptive_k_select_once', True)),
        'recompute_schedule': recompute_schedule,
        'recompute_interval': recompute_interval,
        'recompute_start_epoch': max(int(getattr(args, 'adaptive_k_recompute_start_epoch', 0)), 0),
        'recompute_start_step': max(int(getattr(args, 'adaptive_k_recompute_start_step', 0)), 0),
        'candidates': _normalize_adaptive_k_candidates(getattr(args, 'adaptive_k_candidates', [16, 32, 64])),
        'usage_threshold': float(getattr(args, 'adaptive_k_usage_threshold', 0.5)),
        'min_p10_cluster_size': float(getattr(args, 'adaptive_k_min_p10_cluster_size', 4.0)),
        'max_calib_batches': max(int(getattr(args, 'adaptive_k_max_calib_batches', 8)), 1),
        'use_one_standard_error_rule': bool(getattr(args, 'adaptive_k_use_one_standard_error_rule', True)),
        'fallback_to_current_k': bool(getattr(args, 'adaptive_k_fallback_to_current_k', True)),
        'log_candidate_metrics': bool(getattr(args, 'adaptive_k_log_candidate_metrics', True)),
}


def _adaptive_k_schedule_started(adaptive_cfg: Dict[str, Any], *, epoch: int, current_step: int) -> bool:
    if int(epoch) < int(adaptive_cfg.get('recompute_start_epoch', 0)):
        return False
    if int(current_step) < int(adaptive_cfg.get('recompute_start_step', 0)):
        return False
    return True


def _should_run_adaptive_k_recompute(adaptive_cfg: Dict[str, Any], adaptive_k_state: Dict[str, Any], *, epoch: int, current_step: int) -> bool:
    if not _adaptive_k_schedule_started(adaptive_cfg, epoch=epoch, current_step=current_step):
        return False
    if bool(adaptive_k_state.get('selection_done', False)) and bool(adaptive_cfg.get('select_once', True)):
        return False
    if not bool(adaptive_k_state.get('selection_done', False)):
        return True

    schedule = str(adaptive_cfg.get('recompute_schedule', 'epoch')).lower()
    interval = max(int(adaptive_cfg.get('recompute_interval', 1)), 1)
    last_epoch = adaptive_k_state.get('last_selection_epoch')
    last_step = adaptive_k_state.get('last_selection_step')
    if schedule in {'epoch', 'stage'}:
        if last_epoch is None:
            return True
        return (int(epoch) - int(last_epoch)) >= interval
    if schedule == 'steps':
        if last_step is None:
            return True
        return (int(current_step) - int(last_step)) >= interval
    return False


def _collect_calibration_batches(train_loader, device: str, max_batches: int) -> List[Dict[str, Any]]:
    calibration_batches: List[Dict[str, Any]] = []
    for batch_index, batch in enumerate(train_loader):
        if batch_index >= int(max_batches):
            break
        moved = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        calibration_batches.append(moved)
    return calibration_batches


def _collect_semantic_recompute_features(runtime_model, prototype_head, calibration_batches, args, device: str) -> Optional[torch.Tensor]:
    feature_batches: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in calibration_batches:
            with build_autocast_context(args, device):
                image_output = runtime_model.extract_image_features(batch['images'])
            image_embeddings = runtime_model._cast_to_prototype_dtype(image_output.projected_pooled)
            semantic_features = prototype_head.image_adapter(image_embeddings.detach())
            feature_batches.append(semantic_features.detach().float())
    if not feature_batches:
        return None
    return torch.cat(feature_batches, dim=0)


def _compute_p10_cluster_size(cluster_counts: Optional[torch.Tensor]) -> float:
    if not isinstance(cluster_counts, torch.Tensor) or cluster_counts.numel() == 0:
        return float('nan')
    return float(torch.quantile(cluster_counts.float(), 0.10).item())


def _evaluate_spb_k_candidates(
    *,
    model,
    runtime_model,
    prototype_head,
    calibration_batches: List[Dict[str, Any]],
    semantic_recompute_features: torch.Tensor,
    epoch: int,
    current_step: int,
    total_training_steps: int,
    args,
    adaptive_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_metrics: Dict[int, Dict[str, float]] = {}
    saved_state = prototype_head.export_adaptive_k_runtime_state()
    previous_mode = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            num_semantic_samples = int(semantic_recompute_features.size(0))
            for candidate_k in adaptive_cfg['candidates']:
                metrics: Dict[str, float] = {
                    'diag_cal_loss_mean': float('inf'),
                    'diag_cal_loss_se': 0.0,
                    'routing_usage': float('nan'),
                    'active_basis_count': float('nan'),
                    'p10_cluster_size_img': float('nan'),
                    'p10_cluster_size_txt': float('nan'),
                    'feasible': 0.0,
                    'num_calib_batches': float(len(calibration_batches)),
                    'num_calib_samples': 0.0,
                }
                prototype_head.restore_adaptive_k_runtime_state(saved_state)
                if int(candidate_k) > num_semantic_samples:
                    metrics['reason_infeasible_too_few_samples'] = 1.0
                    candidate_metrics[int(candidate_k)] = metrics
                    continue

                recomputed = prototype_head._recompute_kmeans_anchors(
                    features=semantic_recompute_features,
                    num_clusters=int(candidate_k),
                    max_iters=max(int(getattr(prototype_head.prototype_bank, 'init_max_iters', 15)), 1),
                )
                centers = recomputed['centers'].detach()
                counts_img = recomputed['counts'].detach()
                prototype_head.install_semantic_anchor_cache(centers=centers, counts=counts_img, mark_initialized=True)

                diag_losses: List[float] = []
                alpha_sum: Optional[torch.Tensor] = None
                num_alpha_samples = 0
                counts_txt: Optional[torch.Tensor] = None

                for batch in calibration_batches:
                    with build_autocast_context(args, getattr(args, 'device', 'cuda')):
                        outputs = model(
                            batch,
                            epoch=epoch,
                            current_step=current_step,
                            total_steps=total_training_steps,
                            disable_proxy_losses=True,
                        )
                    alpha = outputs.get('alpha')
                    if isinstance(alpha, torch.Tensor) and alpha.ndim == 2 and alpha.size(1) == int(candidate_k):
                        detached_alpha = alpha.detach().float()
                        alpha_sum = detached_alpha.sum(dim=0) if alpha_sum is None else (alpha_sum + detached_alpha.sum(dim=0))
                        num_alpha_samples += int(detached_alpha.size(0))

                    surrogate_diag = outputs.get('z_t_hat_diag')
                    exact_diag = outputs.get('z_t_exact_diag')
                    if isinstance(surrogate_diag, torch.Tensor) and isinstance(exact_diag, torch.Tensor):
                        diag_loss = prototype_head.losses.diagonal_fidelity_loss(surrogate_diag, exact_diag)
                        diag_losses.append(float(diag_loss.detach().float().item()))

                        anchors_txt, _ = prototype_head._prepare_semantic_anchor_features(
                            centers.to(device=exact_diag.device, dtype=exact_diag.dtype),
                            target_feature_dim=exact_diag.size(-1),
                        )
                        normalized_text = F.normalize(exact_diag.detach().float(), dim=-1)
                        normalized_anchors = F.normalize(anchors_txt.detach().float(), dim=-1)
                        text_assignments = (normalized_text @ normalized_anchors.t()).argmax(dim=-1)
                        text_counts = torch.bincount(text_assignments, minlength=int(candidate_k)).float()
                        counts_txt = text_counts if counts_txt is None else (counts_txt + text_counts)

                if diag_losses:
                    diag_tensor = torch.tensor(diag_losses, dtype=torch.float32)
                    metrics['diag_cal_loss_mean'] = float(diag_tensor.mean().item())
                    if diag_tensor.numel() > 1:
                        metrics['diag_cal_loss_se'] = float(diag_tensor.std(unbiased=True).item() / math.sqrt(float(diag_tensor.numel())))
                    else:
                        metrics['diag_cal_loss_se'] = 0.0
                else:
                    metrics['reason_infeasible_no_diag_batches'] = 1.0

                metrics['num_calib_samples'] = float(num_alpha_samples)
                metrics['p10_cluster_size_img'] = _compute_p10_cluster_size(counts_img)
                metrics['p10_cluster_size_txt'] = _compute_p10_cluster_size(counts_txt)
                if alpha_sum is not None and num_alpha_samples > 0:
                    mean_alpha = (alpha_sum / float(num_alpha_samples)).clamp_min(1e-12)
                    entropy = -(mean_alpha * mean_alpha.log()).sum()
                    active_basis_count = entropy.exp()
                    routing_usage = active_basis_count / float(int(candidate_k))
                    metrics['active_basis_count'] = float(active_basis_count.item())
                    metrics['routing_usage'] = float(routing_usage.item())
                else:
                    metrics['reason_infeasible_no_routing'] = 1.0

                feasible = True
                if not math.isfinite(metrics['diag_cal_loss_mean']):
                    feasible = False
                if not math.isfinite(metrics['routing_usage']) or metrics['routing_usage'] < float(adaptive_cfg['usage_threshold']):
                    feasible = False
                if not math.isfinite(metrics['p10_cluster_size_img']) or metrics['p10_cluster_size_img'] < float(adaptive_cfg['min_p10_cluster_size']):
                    feasible = False
                if math.isfinite(metrics['p10_cluster_size_txt']):
                    if metrics['p10_cluster_size_txt'] < float(adaptive_cfg['min_p10_cluster_size']):
                        feasible = False
                metrics['feasible'] = 1.0 if feasible else 0.0
                candidate_metrics[int(candidate_k)] = metrics
    finally:
        prototype_head.restore_adaptive_k_runtime_state(saved_state)
        if previous_mode:
            model.train()
    return {
        'candidate_metrics': candidate_metrics,
    }


def _select_spb_k_from_metrics(candidate_metrics: Dict[int, Dict[str, float]], adaptive_cfg: Dict[str, Any], current_k: int) -> Tuple[int, bool, bool]:
    feasible_candidates = []
    for candidate_k, metrics in candidate_metrics.items():
        if float(metrics.get('feasible', 0.0)) >= 0.5 and math.isfinite(float(metrics.get('diag_cal_loss_mean', float('inf')))):
            feasible_candidates.append(int(candidate_k))

    if not feasible_candidates:
        return int(current_k), True, True

    best_k = min(
        feasible_candidates,
        key=lambda candidate: float(candidate_metrics[candidate]['diag_cal_loss_mean']),
    )
    selected_k = int(best_k)
    if bool(adaptive_cfg.get('use_one_standard_error_rule', True)):
        threshold = (
            float(candidate_metrics[best_k]['diag_cal_loss_mean'])
            + float(candidate_metrics[best_k].get('diag_cal_loss_se', 0.0))
        )
        selected_k = min(
            candidate
            for candidate in feasible_candidates
            if float(candidate_metrics[candidate]['diag_cal_loss_mean']) <= threshold
        )
    return int(selected_k), False, True


def _maybe_run_adaptive_k_selection(
    *,
    args,
    model,
    train_loader,
    epoch: int,
    current_step: int,
    total_training_steps: int,
    logger,
) -> None:
    # SPB-K adapts the sufficient semantic basis width of the prototype-mediated
    # surrogate construction module. It does not adapt the capacity of a retrieval expert.
    runtime_model = _unwrap_model(model)
    prototype_head = getattr(runtime_model, 'prototype_head', None)
    if prototype_head is None:
        return

    adaptive_cfg = _resolve_adaptive_k_config(args)
    if not bool(adaptive_cfg['enabled']):
        return
    if str(adaptive_cfg['method']).lower() != 'spb':
        return
    if not getattr(runtime_model, 'use_prototype_branch', False):
        return
    if not hasattr(prototype_head, 'adaptive_k_state'):
        return
    if not _should_run_adaptive_k_recompute(
        adaptive_cfg,
        prototype_head.adaptive_k_state,
        epoch=epoch,
        current_step=current_step,
    ):
        return
    if str(getattr(prototype_head, 'prototype_bank_source', '')).lower() != 'recomputed_kmeans':
        if get_rank() == 0:
            logger.warning(
                'adaptive_k enabled but prototype.bank_source=%s; skipping SPB-K because detached recompute anchors are unavailable.',
                getattr(prototype_head, 'prototype_bank_source', None),
            )
        return
    if not bool(getattr(prototype_head, '_semantic_mode_enabled', lambda: False)()):
        if get_rank() == 0:
            logger.warning('adaptive_k enabled but semantic structure mode is inactive; skipping SPB-K.')
        return

    device = getattr(args, 'device', 'cuda')
    calibration_batches = _collect_calibration_batches(
        train_loader=train_loader,
        device=device,
        max_batches=int(adaptive_cfg['max_calib_batches']),
    )
    if not calibration_batches:
        if get_rank() == 0:
            logger.warning('adaptive_k selection skipped because no calibration batches were available.')
        return

    semantic_recompute_features = _collect_semantic_recompute_features(
        runtime_model=runtime_model,
        prototype_head=prototype_head,
        calibration_batches=calibration_batches,
        args=args,
        device=device,
    )
    if not isinstance(semantic_recompute_features, torch.Tensor) or semantic_recompute_features.numel() == 0:
        if get_rank() == 0:
            logger.warning('adaptive_k selection skipped because semantic recompute features were unavailable.')
        return

    eval_result = _evaluate_spb_k_candidates(
        model=model,
        runtime_model=runtime_model,
        prototype_head=prototype_head,
        calibration_batches=calibration_batches,
        semantic_recompute_features=semantic_recompute_features,
        epoch=epoch,
        current_step=current_step,
        total_training_steps=total_training_steps,
        args=args,
        adaptive_cfg=adaptive_cfg,
    )
    candidate_metrics = eval_result.get('candidate_metrics', {})
    current_k = int(prototype_head.get_active_num_prototypes())
    selected_k_local, fallback_used_local, selection_done_local = _select_spb_k_from_metrics(
        candidate_metrics=candidate_metrics,
        adaptive_cfg=adaptive_cfg,
        current_k=current_k,
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        selection_tensor = torch.tensor(
            [selected_k_local, int(fallback_used_local), int(selection_done_local)],
            device=semantic_recompute_features.device,
            dtype=torch.long,
        )
        torch.distributed.broadcast(selection_tensor, src=0)
        selected_k = int(selection_tensor[0].item())
        fallback_used = bool(int(selection_tensor[1].item()))
        selection_done = bool(int(selection_tensor[2].item()))
    else:
        selected_k = int(selected_k_local)
        fallback_used = bool(fallback_used_local)
        selection_done = bool(selection_done_local)

    if int(selected_k) <= int(semantic_recompute_features.size(0)):
        recomputed_selected = prototype_head._recompute_kmeans_anchors(
            features=semantic_recompute_features,
            num_clusters=int(selected_k),
            max_iters=max(int(getattr(prototype_head.prototype_bank, 'init_max_iters', 15)), 1),
        )
        prototype_head.install_semantic_anchor_cache(
            centers=recomputed_selected['centers'].detach(),
            counts=recomputed_selected['counts'].detach(),
            mark_initialized=True,
        )
    else:
        prototype_head._ensure_semantic_cache_shape(int(selected_k))
        prototype_head._semantic_cache_initialized = False

    prototype_head.mark_adaptive_k_selection(
        selected_k=int(selected_k),
        fallback_used=bool(fallback_used),
        candidate_metrics=candidate_metrics,
        selection_done=bool(selection_done),
        epoch=epoch,
        current_step=current_step,
    )

    if get_rank() == 0:
        if bool(adaptive_cfg['log_candidate_metrics']) and isinstance(candidate_metrics, dict):
            for candidate_k, metrics in sorted(candidate_metrics.items(), key=lambda item: int(item[0])):
                logger.info('adaptive_k/candidate_%s/diag_cal_loss_mean=%.6f', candidate_k, float(metrics.get('diag_cal_loss_mean', float('nan'))))
                logger.info('adaptive_k/candidate_%s/diag_cal_loss_se=%.6f', candidate_k, float(metrics.get('diag_cal_loss_se', float('nan'))))
                logger.info('adaptive_k/candidate_%s/routing_usage=%.6f', candidate_k, float(metrics.get('routing_usage', float('nan'))))
                logger.info('adaptive_k/candidate_%s/active_basis_count=%.6f', candidate_k, float(metrics.get('active_basis_count', float('nan'))))
                logger.info('adaptive_k/candidate_%s/p10_cluster_size_img=%.6f', candidate_k, float(metrics.get('p10_cluster_size_img', float('nan'))))
                logger.info('adaptive_k/candidate_%s/p10_cluster_size_txt=%.6f', candidate_k, float(metrics.get('p10_cluster_size_txt', float('nan'))))
                logger.info('adaptive_k/candidate_%s/feasible=%.1f', candidate_k, float(metrics.get('feasible', 0.0)))

        logger.info('adaptive_k/enabled=%.1f', float(bool(adaptive_cfg['enabled'])))
        logger.info('adaptive_k/selected_k=%d', int(selected_k))
        logger.info('adaptive_k/current_k=%d', int(prototype_head.get_active_num_prototypes()))
        logger.info('adaptive_k/selection_done=%.1f', float(bool(prototype_head.adaptive_k_state.get('selection_done', False))))
        logger.info('adaptive_k/fallback_used=%.1f', float(bool(fallback_used)))
        if bool(fallback_used):
            logger.warning(
                'adaptive_k fallback to current K=%d because no feasible SPB-K candidate satisfied routing/support constraints.',
                int(selected_k),
            )


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
    return_epoch_records: bool = False,
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
    if not is_cuda_device(device):
        logger.warning('CUDA is unavailable; fp16-only execution will run without CUDA autocast.')
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
        'Precision policy: forced_fp16=true scaler_enabled=%s',
        bool(getattr(scaler, 'is_enabled', lambda: False)()),
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
    last_validation_snapshot: Dict[str, Any] = {}
    epoch_records: List[Dict[str, Any]] = []
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
        prototype_runtime_state = None
        if bool(getattr(args, 'repro_enabled', False)) and bool(getattr(args, 'repro_save_proto_runtime_state', True)):
            runtime_model = _unwrap_model(model)
            if hasattr(runtime_model, 'get_prototype_runtime_state'):
                prototype_runtime_state = runtime_model.get_prototype_runtime_state()
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
            prototype_runtime_state=prototype_runtime_state,
        )

    def _run_validation(eval_epoch: int) -> float:
        nonlocal latest_metric_state_current, last_validation_snapshot
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
        last_validation_snapshot = {
            'epoch': int(eval_epoch),
            'top1': float(top1_score),
            'loss_metrics': copy.deepcopy(eval_loss_metrics),
            'latest_metrics': copy.deepcopy(getattr(evaluator, 'latest_metrics', {}) or {}),
            'latest_eval_rows': copy.deepcopy(getattr(evaluator, 'latest_eval_rows', []) or []),
            'latest_authority': copy.deepcopy(getattr(evaluator, 'latest_authority', {}) or {}),
        }
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
        _set_epoch_on_loader_sampler(
            train_loader,
            epoch,
            logger=logger,
            enabled=bool(getattr(args, 'repro_enabled', False)) and bool(getattr(args, 'repro_sampler_set_epoch', True)),
        )
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
        for meter in meters.values():
            meter.reset()

        _maybe_run_adaptive_k_selection(
            args=args,
            model=model,
            train_loader=train_loader,
            epoch=epoch,
            current_step=current_steps,
            total_training_steps=total_training_steps,
            logger=logger,
        )
        _maybe_recompute_prototypes_from_deterministic_cache(
            model=model,
            train_loader=train_loader,
            args=args,
            epoch=epoch,
            current_step=current_steps,
            logger=logger,
        )

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
        else:
            averaged_metrics = _meter_averages(meters)
        if coverage_tracker is not None:
            coverage_tracker.reset_epoch()

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
            elif math.isclose(float(top1), float(best_top1), rel_tol=0.0, abs_tol=1e-8) and int(epoch) > int(best_epoch):
                # Reproducible tie policy for logging/state: keep the most recent epoch for equal best metric values.
                best_epoch = int(epoch)
                if isinstance(best_metric_state_current, dict):
                    best_metric_state_current['best_epoch'] = int(best_epoch)
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
                if return_epoch_records:
                    epoch_records.append(
                        {
                            'epoch': int(epoch),
                            'train_scalar_metrics': copy.deepcopy(averaged_metrics),
                            'validation': copy.deepcopy(last_validation_snapshot),
                        }
                    )
                break
        if return_epoch_records:
            epoch_records.append(
                {
                    'epoch': int(epoch),
                    'train_scalar_metrics': copy.deepcopy(averaged_metrics),
                    'validation': copy.deepcopy(last_validation_snapshot),
                }
            )

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
    if return_epoch_records:
        return {
            'best_top1': float(best_top1),
            'best_epoch': int(best_epoch),
            'epoch_records': epoch_records,
            'latest_metric_state': copy.deepcopy(latest_metric_state_current),
            'best_metric_state': copy.deepcopy(best_metric_state_current),
        }
    return None


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
    return_epoch_records: bool = False,
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
            return_epoch_records=return_epoch_records,
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
        return_epoch_records=return_epoch_records,
    )


def do_inference(model, test_img_loader, test_txt_loader, args):
    logger = logging.getLogger('pas.eval')
    logger.info('Enter inferencing')

    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    _ = evaluator.eval(model.eval())
