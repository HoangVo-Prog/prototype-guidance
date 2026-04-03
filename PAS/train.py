import os
import os.path as op
import random
import sys
import warnings

import numpy as np
import torch

from datasets import build_dataloader
from model import build_model
from processor.processor import do_train
from solver import build_lr_scheduler, build_optimizer
from utils.checkpoint import Checkpointer
from utils.comm import get_rank, synchronize
from utils.env import load_runtime_environment
from utils.experiment import ExperimentTracker
from utils.iotools import save_train_configs
from utils.launch import build_nohup_log_path, build_run_name, get_effective_wandb_run_name, launch_with_nohup
from utils.logger import setup_logger
from utils.metrics import Evaluator
from utils.options import get_args

warnings.filterwarnings('ignore')


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    load_runtime_environment()
    args = get_args()
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

    save_train_configs(args.output_dir, args)
    os.makedirs(op.join(args.output_dir, 'img'), exist_ok=True)

    experiment_tracker = ExperimentTracker(args, args.output_dir, distributed_rank=get_rank())

    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes, train_loader=train_loader)
    logger.info('Total params: %2.fM', sum(p.numel() for p in model.parameters()) / 1000000.0)
    model.to(device)

    if args.finetune:
        logger.info('Loading finetune checkpoint from %s', args.finetune)
        param_dict = torch.load(args.finetune, map_location='cpu')['model']
        refined = {}
        for key, value in param_dict.items():
            refined[key.replace('module.', '')] = value.detach().clone()
        model.load_state_dict(refined, strict=False)

    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader, args)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint.get('epoch', 1)
        logger.info('Resuming from epoch %s', start_epoch)

    try:
        do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer, experiment_tracker=experiment_tracker)
    finally:
        experiment_tracker.finish()

