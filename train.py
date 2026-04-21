import os
import os.path as op
import random
import sys
import warnings
import logging

import numpy as np
import torch

from datasets import build_dataloader
from model import build_model
from model.hosts import (
    get_original_itself_module_paths,
    get_original_itself_training_components,
    prepare_itself_legacy_args,
    should_use_original_itself_runtime,
)
from model.prototype import init_mode_requires_data
from processor.processor import do_train as do_train_pas
from solver import (
    build_lr_scheduler,
    build_optimizer,
    summarize_optimizer_param_groups,
    summarize_optimizer_param_groups_observability,
    summarize_scheduler_effective_lrs,
    summarize_config_declared_optimizer_settings,
)
from utils.checkpoint import Checkpointer
from utils.comm import get_rank, synchronize
from utils.env import load_runtime_environment
from utils.experiment import ExperimentTracker
from utils.freeze_schedule import get_group_trainability_snapshot
from utils.iotools import save_train_configs
from utils.launch import build_nohup_log_path, build_run_name, get_effective_wandb_run_name, launch_with_nohup
from utils.logger import setup_logger
from utils.metrics import Evaluator as PASEvaluator
from utils.modular_checkpoint import ModularCheckpointManager
from utils.options import get_args

warnings.filterwarnings('ignore')


_ITSELF_LEGACY_KEY_PREFIXES = (
    # Legacy typo/field-name variants seen in original ITSELF checkpoints.
    ('visul_emb_layer.linear.', 'host_head.visual_embedding_layer.fc.'),
    ('visual_emb_layer.linear.', 'host_head.visual_embedding_layer.fc.'),
    ('classifier_bge.', 'host_head.classifier_global.'),
    ('classifier_id_bge.', 'host_head.classifier_id_global.'),
    ('mlp_bge.', 'host_head.mlp_global.'),
    ('classifier_tse.', 'host_head.classifier_grab.'),
    ('classifier_id_tse.', 'host_head.classifier_id_grab.'),
    ('mlp_tse.', 'host_head.mlp_grab.'),
    ('visul_emb_layer.', 'host_head.visual_embedding_layer.'),
    ('visual_emb_layer.', 'host_head.visual_embedding_layer.'),
    ('texual_emb_layer.', 'host_head.textual_embedding_layer.'),
    ('textual_emb_layer.', 'host_head.textual_embedding_layer.'),
)


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _count_parameters(parameters):
    total = 0
    trainable = 0
    for parameter in parameters:
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return total, trainable


def _format_ratio(trainable: int, total: int):
    percentage = (100.0 * float(trainable) / max(int(total), 1)) if int(total) > 0 else 0.0
    return f'{int(trainable)}/{int(total)} ({percentage:.2f}%)'


def _summarize_optimizer_lrs(optimizer_group_summaries):
    lr_parts = []
    for group_summary in optimizer_group_summaries:
        if int(group_summary.get('tensor_count', 0)) <= 0:
            continue
        lr_parts.append(f"{group_summary['name']}={group_summary['lr']:.2e}")
    return ', '.join(lr_parts) if lr_parts else 'none'


def _log_config_declared_optimizer_summary(logger, args):
    summary = summarize_config_declared_optimizer_settings(args)
    logger.info('=== Section A: Config-declared LR/Optimizer Summary ===')
    logger.info(
        'optimizer.type=%s optimizer_eps=%s lr_factor=%s bias_lr_factor=%s',
        summary.get('optimizer_type'),
        summary.get('optimizer_eps'),
        summary.get('lr_factor'),
        summary.get('bias_lr_factor'),
    )
    lr_parts = ', '.join(f'{key}={value}' for key, value in summary.get('declared_lrs', {}).items())
    wd_parts = ', '.join(f'{key}={value}' for key, value in summary.get('declared_weight_decays', {}).items())
    logger.info('declared_lrs={%s}', lr_parts if lr_parts else 'none')
    logger.info('declared_weight_decays={%s}', wd_parts if wd_parts else 'none')
    schedule_overrides = summary.get('freeze_schedule_lr_overrides', [])
    if schedule_overrides:
        override_parts = []
        for item in schedule_overrides:
            phase = item.get('phase')
            overrides = item.get('lr_overrides', {})
            override_parts.append(f'{phase}:{overrides}')
        logger.info('freeze_schedule.lr_overrides={%s}', '; '.join(override_parts))
    else:
        logger.info('freeze_schedule.lr_overrides={none}')


def _log_optimizer_observability_ground_truth(logger, args, model, optimizer):
    logger.info('=== Section B: Optimizer Param-group Ground Truth ===')
    rows = summarize_optimizer_param_groups_observability(args=args, model=model, optimizer=optimizer)
    if not rows:
        logger.info('No observability rows reconstructed (model has no named_optimizer_groups).')
        return
    for row in rows:
        logger.info(
            (
                '%s logical=%s bucket=%s ownership=%s lr_source=%s base_lr_declared=%s '
                'multiplier_chain=%s final_optimizer_initial_lr=%.6g weight_decay=%.6g tensors=%d params=%d '
                'prefixes=%s samples=%s'
            ),
            row.get('group_id'),
            row.get('logical_group_name', row.get('name')),
            row.get('derived_bucket_label', 'direct'),
            row.get('ownership_tag', 'mixed_or_ambiguous'),
            row.get('lr_source_key', 'unknown'),
            row.get('base_lr_declared'),
            row.get('multiplier_chain'),
            float(row.get('lr', 0.0)),
            float(row.get('weight_decay', 0.0)),
            int(row.get('tensor_count', 0)),
            int(row.get('parameter_count', 0)),
            row.get('prefix_summary', []),
            row.get('parameter_name_samples', []),
        )


def _log_scheduler_effective_lr_snapshot(logger, optimizer, scheduler, epoch_label):
    logger.info('=== Section C: Scheduler-effective LR Snapshot (epoch=%s) ===', epoch_label)
    logger.info(
        'Display Lr selection policy: host-preferred scheduler-selected view (image_backbone -> text_backbone -> '
        'host_projectors -> other -> group_00 fallback).'
    )
    lrs = list(scheduler.get_lr())
    preferred_group_names = ('image_backbone', 'text_backbone', 'host_projectors', 'other')
    named_lrs = []
    for index, group in enumerate(optimizer.param_groups):
        if index >= len(lrs):
            break
        group_name = str(group.get('name', ''))
        named_lrs.append((group_name, float(lrs[index])))
    display_lr = float(lrs[0]) if lrs else 0.0
    for preferred_name in preferred_group_names:
        matched = False
        for group_name, lr_value in named_lrs:
            if group_name == preferred_name:
                display_lr = lr_value
                matched = True
                break
        if matched:
            break
    logger.info('display_lr=%.6g', display_lr)
    for row in summarize_scheduler_effective_lrs(optimizer=optimizer, scheduler=scheduler):
        logger.info(
            '%s logical=%s scheduler_effective_lr=%.6g optimizer_initial_lr=%.6g optimizer_current_lr=%.6g',
            row.get('group_id'),
            row.get('name'),
            float(row.get('scheduler_effective_lr', 0.0)),
            float(row.get('optimizer_initial_lr', 0.0)),
            float(row.get('optimizer_current_lr', 0.0)),
        )


def _bridge_original_itself_loggers(base_logger):
    """Route original ITSELF logger namespaces to the active PAS handlers."""
    bridge_names = ('ITSELF', 'ITSELF.train', 'ITSELF.eval', 'ITSELF.test', 'IRRA', 'IRRA.dataset')
    for logger_name in bridge_names:
        target_logger = logging.getLogger(logger_name)
        target_logger.setLevel(base_logger.level)
        target_logger.propagate = False
        target_logger.handlers.clear()
        for handler in base_logger.handlers:
            target_logger.addHandler(handler)


def _extract_finetune_model_state_dict(checkpoint_payload):
    if isinstance(checkpoint_payload, dict):
        model_state = checkpoint_payload.get('model')
        if isinstance(model_state, dict):
            return model_state
        state_dict = checkpoint_payload.get('state_dict')
        if isinstance(state_dict, dict):
            return state_dict
        if checkpoint_payload and all(isinstance(key, str) for key in checkpoint_payload.keys()):
            return checkpoint_payload
    raise ValueError(
        'Finetune checkpoint must contain a model state dict under `model` or `state_dict`, '
        'or be a plain state_dict mapping.'
    )


def _remap_legacy_itself_key(key):
    for legacy_prefix, current_prefix in _ITSELF_LEGACY_KEY_PREFIXES:
        if key.startswith(legacy_prefix):
            return current_prefix + key[len(legacy_prefix):], True
    return key, False


def _prepare_finetune_state_dict(raw_state_dict, model, args):
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())
    host_type = str(getattr(args, 'host_type', 'clip')).lower()
    prepared_state = {}
    stats = {
        'legacy_renamed': 0,
        'host_head_prefixed': 0,
        'loaded_keys': 0,
        'skipped_missing': 0,
        'skipped_shape_mismatch': 0,
        'skipped_shape_mismatch_keys': [],
        'skipped_missing_keys': [],
    }

    for original_key, value in raw_state_dict.items():
        key = str(original_key)
        if key.startswith('module.'):
            key = key[len('module.'):]
        if key.startswith('model.') and key[len('model.'):] in model_keys:
            key = key[len('model.'):]

        if host_type == 'itself':
            key, renamed = _remap_legacy_itself_key(key)
            if renamed:
                stats['legacy_renamed'] += 1
            if (
                not key.startswith(('host_head.', 'base_model.', 'prototype_head.'))
                and ('host_head.' + key) in model_keys
            ):
                key = 'host_head.' + key
                stats['host_head_prefixed'] += 1

        if key not in model_state_dict:
            stats['skipped_missing'] += 1
            stats['skipped_missing_keys'].append(str(original_key))
            continue

        model_tensor = model_state_dict[key]
        if tuple(model_tensor.shape) != tuple(value.shape):
            stats['skipped_shape_mismatch'] += 1
            stats['skipped_shape_mismatch_keys'].append(
                f'{key}: ckpt{tuple(value.shape)} != model{tuple(model_tensor.shape)}'
            )
            continue

        prepared_state[key] = value.detach().clone()
        stats['loaded_keys'] += 1

    return prepared_state, stats


def log_parameter_trainability(logger, model, args):
    debug_mode = bool(getattr(args, 'log_debug_metrics', False))
    if getattr(args, 'freeze_schedule', None):
        logger.info(
            'Initial parameter trainability snapshot (before epoch-based freeze_schedule activation at epoch start).'
        )
    total_params, trainable_params = _count_parameters(model.parameters())
    snapshot = get_group_trainability_snapshot(model)
    trainable_groups = []
    frozen_groups = []
    parameterless_groups = []
    group_portions = []
    for group_name in sorted(snapshot.keys()):
        group_stats = snapshot[group_name]
        group_total = int(group_stats.get('total_params', 0))
        group_trainable = int(group_stats.get('trainable_params', 0))
        if group_total == 0:
            parameterless_groups.append(group_name)
            continue
        group_portions.append(f'{group_name}={_format_ratio(group_trainable, group_total)}')
        if group_trainable > 0:
            trainable_groups.append(group_name)
        else:
            frozen_groups.append(group_name)

    logger.info(
        'Trainability summary: global=%s trainable_groups=%s frozen_groups=%s parameterless_groups=%s',
        _format_ratio(trainable_params, total_params),
        trainable_groups,
        frozen_groups,
        parameterless_groups,
    )
    if group_portions:
        logger.info('Trainability portions by logical group: %s', ', '.join(group_portions))
    if debug_mode and hasattr(model, 'named_optimizer_groups'):
        for group_name, named_params in model.named_optimizer_groups().items():
            param_count = sum(parameter.numel() for _, parameter in named_params)
            tensor_count = len(named_params)
            logger.info(
                'Debug trainable group %-24s tensors=%d params=%d',
                group_name,
                tensor_count,
                param_count,
            )


def _resolve_resume_checkpoint_path(args) -> str:
    configured_path = str(getattr(args, 'resume_ckpt_file', '') or '').strip()
    if configured_path:
        return configured_path
    candidate_paths = (
        op.join(args.output_dir, 'checkpoint_training_latest.pth'),
        op.join(args.output_dir, 'last.pth'),
    )
    for candidate in candidate_paths:
        if op.exists(candidate):
            return candidate
    return ''


if __name__ == '__main__':
    load_runtime_environment()
    args = get_args()
    use_original_itself = should_use_original_itself_runtime(args)
    if use_original_itself:
        prepare_itself_legacy_args(args)
    args.run_name = build_run_name(args)
    if not args.wandb_run_name:
        args.wandb_run_name = args.run_name

    if args.nohup:
        log_path = build_nohup_log_path(args)
        pid = launch_with_nohup(
            sys.argv,
            log_path,
            run_name_override=args.run_name,
            cwd=os.getcwd(),
        )
        print(f'Launched PAS training in background with PID {pid}. Log: {log_path}')
        raise SystemExit(0)

    set_seed(args.seed + get_rank())

    num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    device = args.device
    args.output_dir = op.join(args.output_dir, args.dataset_name, args.run_name)

    logger = setup_logger('pas', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info('Using %s GPUs', num_gpus)
    logger.info('W&B/log run name: %s', get_effective_wandb_run_name(args))
    logger.info(str(args).replace(',', '\n'))
    deprecated_freeze_controls = list(getattr(args, 'deprecated_freeze_controls', []) or [])
    if deprecated_freeze_controls:
        logger.warning(
            'Deprecated freeze controls were used as compatibility fallbacks: %s. '
            'Use training.freeze_host_retrieval/training.freeze_fusion/training.freeze_prototype_bank/'
            'training.freeze_prototype_projector/training.freeze_routing (and optional training.freeze_host_backbone).',
            sorted(set(deprecated_freeze_controls)),
        )
    modular_checkpoint_manager = ModularCheckpointManager(args=args, save_dir=args.output_dir, logger=logger)
    finetune_path = str(getattr(args, 'finetune', '') or '').strip()
    if finetune_path:
        logger.info('Finetuning enabled (training.finetune): %s', finetune_path)
    if use_original_itself:
        _bridge_original_itself_loggers(logger)
        module_paths = get_original_itself_module_paths()
        logger.info(
            'Original ITSELF adapter modules active: model=%s solver=%s processor=%s metrics=%s',
            module_paths['model_build'],
            module_paths['solver_build'],
            module_paths['processor'],
            module_paths['metrics'],
        )

    experiment_tracker = None
    if not use_original_itself:
        experiment_tracker = ExperimentTracker(args, args.output_dir, distributed_rank=get_rank())

    if bool(getattr(args, 'use_prototype_bank', False)) and modular_checkpoint_manager.has_group_loading_enabled('prototype_bank'):
        init_mode = str(getattr(args, 'prototype_init', 'normalized_random'))
        init_path = str(getattr(args, 'prototype_init_path', '') or '').strip()
        if init_mode_requires_data(init_mode) and not init_path:
            source_cfg = modular_checkpoint_manager.get_group_load_source('prototype_bank')
            source_path = str(source_cfg.get('path', '') or '').strip()
            logger.info(
                'Prototype-bank checkpoint load is enabled (path=%s); overriding prototype_init from %s to normalized_random '
                'to skip automatic train-image-embedding fallback before checkpoint restore.',
                source_path,
                init_mode,
            )
            args.prototype_init = 'normalized_random'

    save_train_configs(args.output_dir, args)
    os.makedirs(op.join(args.output_dir, 'img'), exist_ok=True)

    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    eval_loss_loader = getattr(train_loader, 'eval_loss_loader', None)
    model = build_model(args, num_classes, train_loader=train_loader)
    if modular_checkpoint_manager.has_enabled_group_loading():
        if finetune_path:
            logger.warning(
                'Ignoring deprecated training.finetune=%s because checkpointing.load.enabled sources are active.',
                finetune_path,
            )
        modular_checkpoint_manager.load_configured_groups(model)
    elif finetune_path:
        logger.warning(
            'training.finetune is deprecated as a primary load path. '
            'Prefer checkpointing.load.sources.<group>.path for modular loads.'
        )
        logger.info('Loading finetune checkpoint from %s', finetune_path)
        checkpoint_payload = torch.load(finetune_path, map_location='cpu')
        param_dict = _extract_finetune_model_state_dict(checkpoint_payload)
        refined, remap_stats = _prepare_finetune_state_dict(param_dict, model, args)
        logger.info(
            'Finetune key remap summary: loaded=%d legacy_renamed=%d host_head_prefixed=%d skipped_missing=%d skipped_shape_mismatch=%d',
            remap_stats['loaded_keys'],
            remap_stats['legacy_renamed'],
            remap_stats['host_head_prefixed'],
            remap_stats['skipped_missing'],
            remap_stats['skipped_shape_mismatch'],
        )
        incompatible = model.load_state_dict(refined, strict=False)
        missing_keys = list(getattr(incompatible, 'missing_keys', []))
        unexpected_keys = list(getattr(incompatible, 'unexpected_keys', []))
        logger.info(
            'Finetune load summary: missing_keys=%s unexpected_keys=%s',
            len(missing_keys),
            len(unexpected_keys),
        )
        if missing_keys:
            logger.info('Finetune missing model keys (all): %s', missing_keys)
        if remap_stats['skipped_shape_mismatch_keys']:
            logger.warning(
                'Finetune skipped shape-mismatched keys (all): %s',
                remap_stats['skipped_shape_mismatch_keys'],
            )
        if remap_stats['skipped_missing_keys']:
            logger.info(
                'Finetune ignored unmatched checkpoint keys (all): %s',
                remap_stats['skipped_missing_keys'],
            )
        critical_prefixes = (
            'prototype_head.prototype_bank',
            'prototype_head.image_projector',
            'prototype_head.text_projector',
            'prototype_head.image_adapter',
            'prototype_head.text_adapter',
        )
        critical_missing = [key for key in missing_keys if any(key.startswith(prefix) for prefix in critical_prefixes)]
        if critical_missing:
            loaded_prototype_keys = any(key.startswith('prototype_head.') for key in refined.keys())
            if loaded_prototype_keys:
                logger.warning(
                    'Finetune checkpoint is missing critical PAS modules: %s',
                    critical_missing[:20],
                )
            else:
                logger.info(
                    'Finetune checkpoint has no prototype_head weights; missing PAS prototype modules are expected when loading host-only checkpoints.'
                )
        if unexpected_keys:
            logger.warning('Finetune checkpoint has unexpected keys: %s', unexpected_keys[:20])
    logger.info('Total params: %2.fM', sum(p.numel() for p in model.parameters()) / 1000000.0)
    log_parameter_trainability(logger, model, args)
    model.to(device)
    _log_config_declared_optimizer_summary(logger, args)
    optimizer = build_optimizer(args, model)
    optimizer_group_summaries = summarize_optimizer_param_groups(optimizer)
    _log_optimizer_observability_ground_truth(logger, args, model, optimizer)
    debug_mode = bool(getattr(args, 'log_debug_metrics', False))
    if use_original_itself:
        total_groups = len(optimizer_group_summaries)
        total_tensors = sum(summary['tensor_count'] for summary in optimizer_group_summaries)
        total_params = sum(summary['parameter_count'] for summary in optimizer_group_summaries)
        logger.info(
            'Original ITSELF optimizer summary: groups=%d tensors=%d params=%d effective_lrs={%s}',
            total_groups,
            total_tensors,
            total_params,
            _summarize_optimizer_lrs(optimizer_group_summaries),
        )
        if debug_mode:
            preview_count = min(8, total_groups)
            for group_summary in optimizer_group_summaries[:preview_count]:
                logger.info(
                    'Debug optimizer preview %-24s lr=%.6g weight_decay=%.6g tensors=%d params=%d',
                    group_summary['name'],
                    group_summary['lr'],
                    group_summary['weight_decay'],
                    group_summary['tensor_count'],
                    group_summary['parameter_count'],
                )
            if total_groups > preview_count:
                logger.info('... %d additional optimizer groups omitted from debug preview.', total_groups - preview_count)
    else:
        total_groups = len(optimizer_group_summaries)
        total_tensors = sum(summary['tensor_count'] for summary in optimizer_group_summaries)
        total_params = sum(summary['parameter_count'] for summary in optimizer_group_summaries)
        logger.info(
            'Optimizer summary: groups=%d tensors=%d params=%d effective_lrs={%s}',
            total_groups,
            total_tensors,
            total_params,
            _summarize_optimizer_lrs(optimizer_group_summaries),
        )
        if debug_mode:
            for group_summary in optimizer_group_summaries:
                logger.info(
                    'Debug optimizer group %-28s lr=%.6g weight_decay=%.6g tensors=%d params=%d',
                    group_summary['name'],
                    group_summary['lr'],
                    group_summary['weight_decay'],
                    group_summary['tensor_count'],
                    group_summary['parameter_count'],
                )
    scheduler = build_lr_scheduler(args, optimizer)
    _log_scheduler_effective_lr_snapshot(logger, optimizer, scheduler, epoch_label='init')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)

    do_train_fn = do_train_pas
    evaluator_class = PASEvaluator
    if use_original_itself:
        do_train_fn, evaluator_class = get_original_itself_training_components(args)
    evaluator = evaluator_class(val_img_loader, val_txt_loader, args)

    start_epoch = 1
    resume_state = None
    if args.resume:
        resume_path = _resolve_resume_checkpoint_path(args)
        if not resume_path:
            raise FileNotFoundError(
                'training.resume=true but no resume checkpoint path was provided and no default '
                '`checkpoint_training_latest.pth` / `last.pth` was found in output_dir.'
            )
        restore_rng_on_this_rank = bool(getattr(args, 'resume_restore_rng', True))
        if args.distributed and get_rank() != 0 and restore_rng_on_this_rank:
            logger.warning(
                'Skipping RNG restore on distributed rank=%d because checkpoint stores a single-process RNG snapshot '
                '(saved by rank0).',
                get_rank(),
            )
            restore_rng_on_this_rank = False
        resume_state = checkpointer.resume_training(
            resume_path,
            strict=bool(getattr(args, 'resume_strict', False)),
            restore_rng=restore_rng_on_this_rank,
            scaler=None,
        )
        start_epoch = int(resume_state.get('start_epoch', 1) or 1)
        training_state = resume_state.get('training_state', {}) if isinstance(resume_state, dict) else {}
        best_metric_state = training_state.get('best_metric_state', {}) if isinstance(training_state, dict) else {}
        logger.info(
            (
                'Resumed checkpoint path=%s start_epoch=%d global_step=%d best_%s=%s best_row=%s '
                'optimizer_restored=%s scheduler_restored=%s scaler_restored=%s rng_restored=%s strict=%s'
            ),
            resume_path,
            start_epoch,
            int(resume_state.get('global_step', 0) or 0),
            str(best_metric_state.get('name', 'R1')),
            best_metric_state.get('value'),
            best_metric_state.get('selected_row'),
            bool(resume_state.get('optimizer_restored', False)),
            bool(resume_state.get('scheduler_restored', False)),
            bool(resume_state.get('scaler_restored', False)),
            bool(resume_state.get('rng_restored', False)),
            bool(getattr(args, 'resume_strict', False)),
        )
        if resume_state.get('warnings'):
            logger.warning('Resume warnings for %s: %s', resume_path, resume_state.get('warnings'))

    try:
        if use_original_itself:
            if resume_state is not None:
                logger.warning(
                    'Full resume state propagation is only implemented for the PAS runtime path. '
                    'Original ITSELF adapter runtime will resume using loaded model/optimizer/scheduler/start_epoch only.'
                )
            do_train_fn(
                start_epoch,
                args,
                model,
                train_loader,
                evaluator,
                optimizer,
                scheduler,
                checkpointer,
            )
        else:
            do_train_fn(
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
    finally:
        if experiment_tracker is not None:
            experiment_tracker.finish()

