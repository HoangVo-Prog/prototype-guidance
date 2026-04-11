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
from processor.processor import do_train as do_train_pas
from solver import build_lr_scheduler, build_optimizer, summarize_optimizer_param_groups
from utils.checkpoint import Checkpointer
from utils.comm import get_rank, synchronize
from utils.env import load_runtime_environment
from utils.experiment import ExperimentTracker
from utils.iotools import save_train_configs
from utils.launch import build_nohup_log_path, build_run_name, get_effective_wandb_run_name, launch_with_nohup
from utils.logger import setup_logger
from utils.metrics import Evaluator as PASEvaluator
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

    if should_use_original_itself_runtime(args):
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'visual'):
            image_total, image_trainable = _count_parameters(model.base_model.visual.parameters())
        else:
            image_total, image_trainable = 0, 0

        text_parameters = []
        if hasattr(model, 'base_model'):
            base_model = model.base_model
            for attribute in ('transformer', 'token_embedding'):
                module = getattr(base_model, attribute, None)
                if module is not None:
                    text_parameters.extend(list(module.parameters()))
            for attribute in ('positional_embedding', 'text_projection'):
                tensor = getattr(base_model, attribute, None)
                if isinstance(tensor, torch.Tensor):
                    text_parameters.append(tensor)
            ln_final = getattr(base_model, 'ln_final', None)
            if ln_final is not None:
                for attribute in ('weight', 'bias'):
                    tensor = getattr(ln_final, attribute, None)
                    if isinstance(tensor, torch.Tensor):
                        text_parameters.append(tensor)

        text_total, text_trainable = _count_parameters(text_parameters)
        logger.info(
            'ITSELF module params: image_backbone=%d/%d text_backbone=%d/%d',
            image_trainable,
            image_total,
            text_trainable,
            text_total,
        )
        return

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
        'Freeze status: host_type=%s image_backbone=%s text_backbone=%s prototype_side=%s projectors=%s',
        str(getattr(args, 'host_type', 'clip')),
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

    save_train_configs(args.output_dir, args)
    os.makedirs(op.join(args.output_dir, 'img'), exist_ok=True)

    experiment_tracker = None
    if not use_original_itself:
        experiment_tracker = ExperimentTracker(args, args.output_dir, distributed_rank=get_rank())

    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    eval_loss_loader = getattr(train_loader, 'eval_loss_loader', None)
    model = build_model(args, num_classes, train_loader=train_loader)
    logger.info('Total params: %2.fM', sum(p.numel() for p in model.parameters()) / 1000000.0)
    log_parameter_trainability(logger, model, args)
    model.to(device)

    if finetune_path:
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
    optimizer = build_optimizer(args, model)
    optimizer_group_summaries = summarize_optimizer_param_groups(optimizer)
    if use_original_itself:
        total_groups = len(optimizer_group_summaries)
        total_tensors = sum(summary['tensor_count'] for summary in optimizer_group_summaries)
        total_params = sum(summary['parameter_count'] for summary in optimizer_group_summaries)
        logger.info(
            'Original ITSELF optimizer summary: groups=%d tensors=%d params=%d',
            total_groups,
            total_tensors,
            total_params,
        )
        preview_count = min(8, total_groups)
        for group_summary in optimizer_group_summaries[:preview_count]:
            logger.info(
                'Optimizer preview %-24s lr=%.6g weight_decay=%.6g tensors=%d params=%d',
                group_summary['name'],
                group_summary['lr'],
                group_summary['weight_decay'],
                group_summary['tensor_count'],
                group_summary['parameter_count'],
            )
        if total_groups > preview_count:
            logger.info('... %d additional optimizer groups omitted from log for brevity.', total_groups - preview_count)
    else:
        for group_summary in optimizer_group_summaries:
            logger.info(
                'Optimizer group %-28s lr=%.6g weight_decay=%.6g tensors=%d params=%d',
                group_summary['name'],
                group_summary['lr'],
                group_summary['weight_decay'],
                group_summary['tensor_count'],
                group_summary['parameter_count'],
            )
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

    do_train_fn = do_train_pas
    evaluator_class = PASEvaluator
    if use_original_itself:
        do_train_fn, evaluator_class = get_original_itself_training_components(args)
    evaluator = evaluator_class(val_img_loader, val_txt_loader, args)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint.get('epoch', 1)
        logger.info('Resuming from epoch %s', start_epoch)

    try:
        if use_original_itself:
            logger.info(
                'Entering original ITSELF training loop. Note: it performs a full validation pass before Epoch 1; this can take several minutes with no per-batch logs.'
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
            )
    finally:
        if experiment_tracker is not None:
            experiment_tracker.finish()






