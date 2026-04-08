import os
import os.path as op
import torch
import numpy as np
import random
import time
from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize
from utils.env import load_runtime_environment
from utils.launch import  build_nohup_log_path, build_run_name, get_effective_wandb_run_name, launch_with_nohup
import warnings
warnings.filterwarnings("ignore")

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

def log_parameter_trainability(logger, model, args):
    total_params, trainable_params = _count_parameters(model.parameters())
    logger.info(
        'Parameter trainability: trainable=%d / total=%d (%.2f%%)',
        trainable_params,
        total_params,
        100.0 * trainable_params / max(total_params, 1),
    )
    if hasattr(model, 'named_optimizer_groups'):
        for group_name, named_params in model.named_optimizer_groups().items():
            param_count = sum(parameter.numel() for _, parameter in named_params)
            tensor_count = len(named_params)
            logger.info('Trainable group %-16s tensors=%d params=%d', group_name, tensor_count, param_count)

    image_total, image_trainable = _count_parameters(model.base_model.visual.parameters())
    text_parameters = list(model.base_model.transformer.parameters())
    text_parameters.extend(model.base_model.token_embedding.parameters())
    text_parameters.extend([model.base_model.positional_embedding, model.base_model.ln_final.weight, model.base_model.ln_final.bias, model.base_model.text_projection])
    text_total, text_trainable = _count_parameters(text_parameters)
    prototype_bank_module = getattr(model.prototype_head, 'prototype_bank', None)
    prototype_total, prototype_trainable = _count_parameters(prototype_bank_module.parameters()) if prototype_bank_module is not None else (0, 0)

    projector_params = []
    for module_name in ('image_projector', 'text_projector', 'image_adapter', 'text_adapter'):
        module = getattr(model.prototype_head, module_name, None)
        if module is None:
            continue
        projector_params.extend(list(module.parameters()))
    projector_total, projector_trainable = _count_parameters(projector_params)

    class_proxies = getattr(getattr(model.prototype_head, 'losses', None), 'class_proxies', None)
    proxy_total, proxy_trainable = _count_parameters([class_proxies]) if class_proxies is not None else (0, 0)
    logger.info(
        'Freeze status: stage=%s image_backbone=%s text_backbone=%s prototype_side=%s projectors=%s',
        str(getattr(args, 'training_stage', 'stage1')),
        'frozen' if bool(getattr(args, 'freeze_image_backbone', True)) else 'trainable',
        'frozen' if bool(getattr(args, 'freeze_text_backbone', True)) else 'trainable',
        'frozen' if bool(getattr(args, 'freeze_prototype_side', False)) else 'trainable',
        'frozen' if projector_trainable == 0 else 'trainable',
    )
    logger.info(
        'Module params: image_backbone=%d/%d text_backbone=%d/%d prototype_bank=%d/%d projectors=%d/%d class_proxies=%d/%d',
        image_trainable,
        image_total,
        text_trainable,
        text_total,
        prototype_trainable,
        prototype_total,
        projector_trainable,
        projector_total,
        proxy_trainable,
        proxy_total,
    )


if __name__ == '__main__':
    env = load_runtime_environment()
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
    name = "noname"

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = args.device
    args.output_dir = op.join(args.output_dir, args.dataset_name, args.run_name)

    logger = setup_logger('pas', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info('Using %s GPUs', num_gpus)
    logger.info('W&B/log run name: %s', get_effective_wandb_run_name(args))
    logger.info(str(args).replace(',', '\n'))

    save_train_configs(args.output_dir, args)
    os.makedirs(op.join(args.output_dir, 'img'), exist_ok=True)

        
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    log_parameter_trainability(logger, model, args)

    model.to(device)
    if args.finetune:
        logger.info("loading {} model".format(args.finetune))
        param_dict = torch.load(args.finetune,map_location='cpu')['model']
        for k in list(param_dict.keys()):
            refine_k = k.replace('module.','')
            param_dict[refine_k] = param_dict[k].detach().clone()
            del param_dict[k]
        model.load_state_dict(param_dict, False)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)


    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader, args)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']
        logger.info(f"===================>start {start_epoch}")

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
