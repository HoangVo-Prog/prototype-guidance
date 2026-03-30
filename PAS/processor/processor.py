import logging
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.comm import get_rank, synchronize
from utils.experiment import ExperimentTracker
from utils.metric_logging import TRACKED_SCALAR_KEYS, build_train_metrics, build_validation_metrics, collect_scalar_metrics
from utils.meter import AverageMeter
from utils.metrics import Evaluator


METER_KEYS = ('loss_total',) + tuple(key for key in TRACKED_SCALAR_KEYS if key != 'loss_total')



def _make_meters():
    return {key: AverageMeter() for key in METER_KEYS}



def do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer, experiment_tracker: ExperimentTracker = None):
    log_period = args.log_period
    eval_period = args.eval_period
    save_interval = int(getattr(args, 'save_interval', 0) or 0)
    grad_clip = float(getattr(args, 'grad_clip', 0.0) or 0.0)
    device = getattr(args, 'device', 'cuda')
    num_epoch = args.num_epoch
    arguments = {'num_epoch': num_epoch, 'iteration': 0}

    logger = logging.getLogger('pas.train')
    logger.info('start training')
    meters = _make_meters()
    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    current_steps = 0
    best_epoch = start_epoch

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        model.train()
        for n_iter, batch in enumerate(train_loader):
            current_steps += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch)
            total_loss = outputs['loss_total']

            optimizer.zero_grad()
            total_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            synchronize()

            scalar_metrics = collect_scalar_metrics(outputs, include_debug_metrics=getattr(args, 'log_debug_metrics', True))
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
                        include_debug_metrics=getattr(args, 'log_debug_metrics', True),
                    ),
                    step=current_steps,
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
                    None,
                    outputs,
                    scheduler.get_lr()[0],
                    include_debug_metrics=getattr(args, 'log_debug_metrics', True),
                ),
                step=current_steps,
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

        if save_interval > 0 and epoch % save_interval == 0:
            arguments['epoch'] = epoch
            checkpointer.save(f'epoch_{epoch}', **arguments)

        if epoch % eval_period == 0 and get_rank() == 0:
            logger.info('Validation Results - Epoch: {}'.format(epoch))
            if args.distributed:
                top1 = evaluator.eval(model.module.eval())
            else:
                top1 = evaluator.eval(model.eval())
            torch.cuda.empty_cache()
            if experiment_tracker is not None:
                experiment_tracker.log(build_validation_metrics(epoch, evaluator=evaluator), step=current_steps)
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

    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    _ = evaluator.eval(model.eval())

