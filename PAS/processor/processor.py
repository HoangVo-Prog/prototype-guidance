import logging
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.comm import get_rank, synchronize
from utils.experiment import ExperimentTracker
from utils.metric_logging import TRACKED_SCALAR_KEYS, RoutingCoverageTracker, build_train_metrics, build_validation_metrics, collect_loss_metrics, collect_scalar_metrics
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.precision import build_autocast_context, build_grad_scaler, canonicalize_amp_dtype, is_amp_enabled, is_cuda_device


METER_KEYS = ('loss_total',) + tuple(key for key in TRACKED_SCALAR_KEYS if key != 'loss_total')





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
        'grad_norm_image_projector': _parameter_group_grad_norm(named_parameters, ('prototype_head.image_projector', 'prototype_head.image_adapter', 'host_head.image_projector')),
        'grad_norm_text_projector': _parameter_group_grad_norm(named_parameters, ('prototype_head.text_projector', 'prototype_head.text_adapter', 'host_head.text_projector')),
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


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer, experiment_tracker: ExperimentTracker = None, eval_loss_loader=None):
    log_period = args.log_period
    eval_period = args.eval_period
    save_interval = int(getattr(args, 'save_interval', 0) or 0)
    grad_clip = float(getattr(args, 'grad_clip', 0.0) or 0.0)
    device = getattr(args, 'device', 'cuda')
    num_epoch = args.num_epoch
    arguments = {'num_epoch': num_epoch, 'iteration': 0}

    logger = logging.getLogger('pas.train')
    logger.info('start training')
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
    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    current_steps = 0
    best_epoch = start_epoch
    log_debug_metrics = bool(getattr(args, 'log_debug_metrics', True))
    coverage_tracker = RoutingCoverageTracker() if log_debug_metrics else None

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        model.train()
        for n_iter, batch in enumerate(train_loader):
            current_steps += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with build_autocast_context(args, device):
                outputs = model(batch)
                total_loss = outputs['loss_total']

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                if isinstance(outputs.get('debug'), dict):
                    outputs['debug'].update(_collect_gradient_metrics(model))
                    outputs['debug'].update(_collect_output_gradient_metrics(outputs, scale=scaler.get_scale()))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if isinstance(outputs.get('debug'), dict):
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

            if experiment_tracker is not None and get_rank() == 0 and current_steps % max(getattr(args, 'wandb_log_interval', log_period), 1) == 0:
                experiment_tracker.log(
                    build_train_metrics(
                        epoch,
                        current_steps,
                        outputs,
                        scheduler.get_lr()[0],
                        include_debug_metrics=log_debug_metrics,
                    )
                )

            if (n_iter + 1) % log_period == 0:
                info = [f'Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]']
                for key, meter in meters.items():
                    if meter.count > 0:
                        info.append(f'{key}: {meter.avg:.4f}')
                info.append(f'Base Lr: {scheduler.get_lr()[0]:.2e}')
                logger.info(', '.join(info))

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        for key, meter in meters.items():
            if meter.count > 0:
                tb_writer.add_scalar(key, meter.avg, epoch)

        if experiment_tracker is not None and get_rank() == 0:
            experiment_tracker.log(
                build_train_metrics(
                    epoch,
                    current_steps,
                    outputs,
                    scheduler.get_lr()[0],
                    include_debug_metrics=log_debug_metrics,
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

        if save_interval > 0 and epoch % save_interval == 0:
            arguments['epoch'] = epoch
            checkpointer.save(f'epoch_{epoch}', **arguments)

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
            if best_top1 < top1:
                best_top1 = top1
                best_epoch = epoch
                arguments['epoch'] = epoch
                checkpointer.save('best', **arguments)

    if get_rank() == 0:
        logger.info(f'best R1: {best_top1} at epoch {best_epoch}')

    tb_writer.close()



def do_inference(model, test_img_loader, test_txt_loader, args):
    logger = logging.getLogger('pas.eval')
    logger.info('Enter inferencing')
    if bool(getattr(args, 'amp', False)) and not is_cuda_device(getattr(args, 'device', 'cuda')):
        raise ValueError('training.amp=true requires a CUDA device.')

    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    _ = evaluator.eval(model.eval())









