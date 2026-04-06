import torch

from .lr_scheduler import LRSchedulerWithWarmup


GROUP_TO_LR_ATTR = {
    'prototype_bank': 'lr_prototype_bank',
    'prototype_projectors': 'lr_projectors',
    'host_projectors': 'lr_host_projectors',
    'class_proxies': 'lr_class_proxies',
    'image_backbone': 'lr_image_backbone',
    'text_backbone': 'lr_text_backbone',
    'other': 'lr',
}

GROUP_TO_WD_ATTR = {
    'prototype_bank': 'weight_decay_prototype_bank',
    'prototype_projectors': 'weight_decay_projectors',
    'host_projectors': 'weight_decay_host_projectors',
    'class_proxies': 'weight_decay_class_proxies',
    'image_backbone': 'weight_decay_image_backbone',
    'text_backbone': 'weight_decay_text_backbone',
    'other': 'weight_decay',
}


def _coerce_optional_float(value, attr_name: str):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == '' or stripped.lower() in {'none', 'null'}:
            return None
        try:
            return float(stripped)
        except ValueError as exc:
            raise TypeError(f'Optimizer setting `{attr_name}` must be numeric or null; got {value!r}.') from exc
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f'Optimizer setting `{attr_name}` must be numeric or null; got {value!r}.') from exc


def _group_lr(args, group_name: str) -> float:
    attr = GROUP_TO_LR_ATTR[group_name]
    value = _coerce_optional_float(getattr(args, attr, None), attr)
    if value is None or value < 0:
        return float(args.lr)
    return value


def _group_weight_decay(args, group_name: str) -> float:
    attr = GROUP_TO_WD_ATTR[group_name]
    value = _coerce_optional_float(getattr(args, attr, None), attr)
    if value is None:
        return float(args.weight_decay)
    return value


def _use_original_itself_stage0_optimizer(args, model) -> bool:
    del model
    return (
        str(getattr(args, 'host_type', 'clip')).lower() == 'itself'
        and str(getattr(args, 'training_stage', 'joint')).lower() == 'stage0'
        and not bool(getattr(args, 'use_prototype_branch', False))
    )


def _build_original_itself_stage0_optimizer(args, model):
    base_lr = float(getattr(args, 'lr', 1e-5))
    base_weight_decay = float(getattr(args, 'weight_decay', 4e-5))
    lr_factor = float(getattr(args, 'lr_factor', 5.0))
    bias_lr_factor = float(getattr(args, 'bias_lr_factor', 2.0))
    weight_decay_bias = float(getattr(args, 'weight_decay_bias', 0.0))
    optimizer_eps = float(getattr(args, 'optimizer_eps', 1e-3))

    param_groups = []
    seen_parameter_ids = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        seen_parameter_ids.add(id(parameter))
        lr = base_lr
        weight_decay = base_weight_decay

        if 'cross' in name:
            lr = base_lr * lr_factor
        if 'bias' in name:
            lr = base_lr * bias_lr_factor
            weight_decay = weight_decay_bias
        if 'classifier' in name or 'mlm_head' in name:
            lr = base_lr * lr_factor
        if 'visul_emb_layer' in name or 'visual_embedding_layer' in name:
            lr = 1e-3
        if 'texual_emb_layer' in name or 'textual_embedding_layer' in name:
            lr = 1e-3

        param_groups.append(
            {
                'params': [parameter],
                'lr': lr,
                'weight_decay': weight_decay,
                'name': name,
            }
        )

    trainable_parameter_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen_parameter_ids != trainable_parameter_ids:
        raise RuntimeError('Original ITSELF Stage 0 optimizer did not include all trainable parameters.')

    if args.optimizer == 'SGD':
        return torch.optim.SGD(param_groups, lr=base_lr, momentum=args.momentum)
    if args.optimizer == 'Adam':
        return torch.optim.Adam(param_groups, lr=base_lr, betas=(args.alpha, args.beta), eps=optimizer_eps)
    if args.optimizer == 'AdamW':
        return torch.optim.AdamW(param_groups, lr=base_lr, betas=(args.alpha, args.beta), eps=1e-8)
    raise NotImplementedError(f'Unsupported optimizer: {args.optimizer}')


def build_optimizer(args, model):
    if _use_original_itself_stage0_optimizer(args, model):
        return _build_original_itself_stage0_optimizer(args, model)

    if not hasattr(model, 'named_optimizer_groups'):
        raise AttributeError('Model must define named_optimizer_groups() for Phase E optimizer construction.')

    param_groups = []
    for group_name, named_params in model.named_optimizer_groups().items():
        if not named_params:
            continue
        parameters = [parameter for _, parameter in named_params]
        group = {
            'params': parameters,
            'lr': _group_lr(args, group_name),
            'weight_decay': _group_weight_decay(args, group_name),
            'name': group_name,
        }
        param_groups.append(group)

    if args.optimizer == 'SGD':
        return torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)
    if args.optimizer == 'Adam':
        return torch.optim.Adam(param_groups, lr=args.lr, betas=(args.alpha, args.beta), eps=1e-8)
    if args.optimizer == 'AdamW':
        return torch.optim.AdamW(param_groups, lr=args.lr, betas=(args.alpha, args.beta), eps=1e-8)
    raise NotImplementedError(f'Unsupported optimizer: {args.optimizer}')


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
