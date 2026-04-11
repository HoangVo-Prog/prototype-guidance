import logging
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from solver import build_lr_scheduler, build_optimizer, summarize_optimizer_param_groups
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
    collect_loss_metrics,
    collect_scalar_metrics,
)
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.precision import build_autocast_context, build_grad_scaler, canonicalize_amp_dtype, is_amp_enabled, is_cuda_device


METER_KEYS = ('loss_total',) + tuple(key for key in TRACKED_SCALAR_KEYS if key != 'loss_total')
CONSOLE_LOSS_LOG_KEYS = tuple(key for key in TRAIN_LOSS_KEYS if key in METER_KEYS)


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
                for key, value in collect_loss_metrics(outputs).items():
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
        'grad_norm_class_proxies': _parameter_group_grad_norm(named_parameters, ('prototype_head.losses.class_proxies',)),
        'grad_norm_image_projector': _parameter_group_grad_norm(named_parameters, ('prototype_head.image_projector', 'prototype_head.image_adapter', 'host_head')),
        'grad_norm_text_projector': _parameter_group_grad_norm(named_parameters, ('prototype_head.text_projector', 'prototype_head.text_adapter', 'host_head')),
        'grad_norm_prototype_bank': _parameter_group_grad_norm(named_parameters, ('prototype_head.prototype_bank',)),
        'grad_norm_image_backbone': _parameter_group_grad_norm(named_parameters, ('base_model.visual',)),
        'grad_norm_text_backbone': _parameter_group_grad_norm(named_parameters, ('base_model.transformer', 'base_model.token_embedding', 'base_model.positional_embedding', 'base_model.ln_final', 'base_model.text_projection')),
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
):
    log_period = args.log_period
    eval_period = args.eval_period
    grad_clip = float(getattr(args, 'grad_clip', 0.0) or 0.0)
    device = getattr(args, 'device', 'cuda')
    num_epoch = args.num_epoch
    arguments = {'num_epoch': num_epoch, 'iteration': 0}

    logger = logging.getLogger('pas.train')
    logger.info('start training')
    if getattr(args, 'prototype_selection_metric', None):
        logger.warning(
            'training.prototype_selection_metric is deprecated by checkpointing.metric/checkpointing.save.*. '
            'Use checkpointing for per-group best/latest artifacts.'
        )
    if bool(getattr(args, 'amp', False)) and not is_cuda_device(device):
        raise ValueError('training.amp=true requires a CUDA device.')
    scaler = build_grad_scaler(args, device)
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
    log_debug_metrics = bool(getattr(args, 'log_debug_metrics', True))
    coverage_tracker = RoutingCoverageTracker() if log_debug_metrics else None
    freeze_schedule_phases = parse_freeze_schedule_config(
        getattr(args, 'freeze_schedule', None),
        num_epoch=num_epoch,
    )
    active_freeze_phase_name = None
    missing_phase_warning_emitted = False
    if freeze_schedule_phases:
        schedule_preview = ', '.join(
            f'{phase.name}[{phase.epoch_start}-{phase.epoch_end}]' for phase in freeze_schedule_phases
        )
        logger.info('Using training.freeze_schedule phases: %s', schedule_preview)

    for epoch in range(start_epoch, num_epoch + 1):
        active_phase = get_active_phase(freeze_schedule_phases, epoch)
        if active_phase is not None and active_phase.name != active_freeze_phase_name:
            runtime_model = _unwrap_model(model)
            apply_phase_trainability(runtime_model, active_phase)
            applied_loss_weights = apply_loss_weight_overrides(runtime_model, args, active_phase.loss_weights)

            previous_optimizer = optimizer
            optimizer = build_optimizer(args, runtime_model)
            _copy_optimizer_state(previous_optimizer, optimizer)
            lr_override_summary = apply_optimizer_lr_overrides(optimizer, active_phase.lr_overrides)

            scheduler = build_lr_scheduler(args, optimizer)
            _rewind_scheduler_to_epoch(scheduler, epoch - 1)

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

        model.train()
        for n_iter, batch in enumerate(train_loader):
            current_steps += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with build_autocast_context(args, device):
                outputs = model(batch, current_step=current_steps)
                total_loss = outputs['loss_total']

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

            scalar_metrics = collect_scalar_metrics(outputs, include_debug_metrics=log_debug_metrics)
            batch_size = batch['images'].shape[0]
            for key, meter in meters.items():
                value = scalar_metrics.get(key)
                if value is not None:
                    meter.update(value, batch_size)
                    wandb_interval_meters[key].update(value, batch_size)

            if experiment_tracker is not None and get_rank() == 0 and current_steps % max(getattr(args, 'wandb_log_interval', log_period), 1) == 0:
                averaged_metrics = _meter_averages(wandb_interval_meters)
                experiment_tracker.log(
                    build_train_metrics_from_scalars(
                        epoch=epoch,
                        step=current_steps,
                        scalar_metrics=averaged_metrics,
                        lr=scheduler.get_lr()[0],
                    )
                )
                for meter in wandb_interval_meters.values():
                    meter.reset()

            if (n_iter + 1) % log_period == 0:
                info = [f'Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]']
                for key in CONSOLE_LOSS_LOG_KEYS:
                    meter = meters[key]
                    if meter.count > 0:
                        info.append(f'{key}: {meter.avg:.4f}')
                info.append(f'Base Lr: {scheduler.get_lr()[0]:.2e}')
                logger.info(', '.join(info))

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
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
                    lr=scheduler.get_lr()[0],
                )
            )
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
            logger.info('Validation Results - Epoch: {}'.format(epoch))
            eval_loss_metrics = _compute_eval_loss_metrics(
                model.module if args.distributed else model,
                eval_loss_loader,
                args,
            )
            eval_loss_total = eval_loss_metrics.get('loss_total')
            if eval_loss_total is not None:
                logger.info('Selected eval split loss (proxy-disabled): %.4f', eval_loss_total)
            if args.distributed:
                top1 = evaluator.eval(model.module.eval())
            else:
                top1 = evaluator.eval(model.eval())
            torch.cuda.empty_cache()
            if experiment_tracker is not None:
                validation_metrics = build_validation_metrics(epoch, evaluator=evaluator, loss_metrics=eval_loss_metrics)
                experiment_tracker.log(validation_metrics)
            if modular_checkpoint_manager is not None:
                model_for_group_ckpt = model.module if hasattr(model, 'module') else model
                modular_checkpoint_manager.save_latest(
                    model=model_for_group_ckpt,
                    epoch=epoch,
                    global_step=current_steps,
                    metric_value=float(top1),
                )
                modular_checkpoint_manager.save_best_if_improved(
                    model=model_for_group_ckpt,
                    epoch=epoch,
                    global_step=current_steps,
                    metric_value=float(top1),
                )
            if best_top1 < top1:
                best_top1 = top1
                best_epoch = epoch
                arguments['epoch'] = epoch
                checkpointer.save('best', **arguments)

    if get_rank() == 0:
        if last_epoch >= start_epoch:
            arguments['epoch'] = last_epoch
            checkpointer.save('last', **arguments)
        logger.info(f'best R1: {best_top1} at epoch {best_epoch}')

    tb_writer.close()



def do_inference(model, test_img_loader, test_txt_loader, args):
    logger = logging.getLogger('pas.eval')
    logger.info('Enter inferencing')

    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    _ = evaluator.eval(model.eval())









