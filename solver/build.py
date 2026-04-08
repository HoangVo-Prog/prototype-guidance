import torch

from .lr_scheduler import LRSchedulerWithWarmup


GROUP_TO_LR_ATTR = {
    'prototype_bank': 'lr_prototype_bank',
    'prototype_projectors': 'lr_projectors',
    'prototype_routing': 'lr_prototype_routing',
    'prototype_pooling': 'lr_prototype_pooling',
    'prototype_contextualization': 'lr_prototype_contextualization',
    'host_projectors': 'lr_host_projectors',
    'class_proxies': 'lr_class_proxies',
    'image_backbone': 'lr_image_backbone',
    'text_backbone': 'lr_text_backbone',
    'other': 'lr',
}

GROUP_TO_WD_ATTR = {
    'prototype_bank': 'weight_decay_prototype_bank',
    'prototype_projectors': 'weight_decay_projectors',
    'prototype_routing': 'weight_decay_prototype_routing',
    'prototype_pooling': 'weight_decay_prototype_pooling',
    'prototype_contextualization': 'weight_decay_prototype_contextualization',
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


def _itself_stage0_group_spec(name: str, base_lr: float, base_weight_decay: float, lr_factor: float, bias_lr_factor: float, weight_decay_bias: float):
    lr = base_lr
    weight_decay = base_weight_decay
    label = 'base'

    if 'cross' in name:
        lr = base_lr * lr_factor
        label = 'cross'
    if 'bias' in name:
        lr = base_lr * bias_lr_factor
        weight_decay = weight_decay_bias
        label = 'bias'
    if 'classifier' in name or 'mlm_head' in name:
        lr = base_lr * lr_factor
        label = 'classifier_bias' if 'bias' in name else 'classifier'
    if 'visul_emb_layer' in name or 'visual_embedding_layer' in name:
        lr = base_lr * lr_factor
        label = 'visual_embedding_bias' if 'bias' in name else 'visual_embedding'
    if 'texual_emb_layer' in name or 'textual_embedding_layer' in name:
        lr = base_lr * lr_factor
        label = 'textual_embedding_bias' if 'bias' in name else 'textual_embedding'

    return label, lr, weight_decay


def _validate_param_group_assignment(model, named_group_entries):
    trainable_parameters = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    seen = {}
    duplicates = []

    for group_name, named_params in named_group_entries.items():
        local_ids = set()
        for name, parameter in named_params:
            parameter_id = id(parameter)
            if parameter_id in local_ids:
                duplicates.append((name, group_name, group_name))
                continue
            local_ids.add(parameter_id)
            if parameter_id in seen:
                duplicates.append((name, group_name, seen[parameter_id]))
            else:
                seen[parameter_id] = group_name

    if duplicates:
        formatted = ', '.join(f'{name} ({current} vs {previous})' for name, current, previous in duplicates[:10])
        raise RuntimeError(f'Optimizer parameter assignment contains duplicates: {formatted}')

    missing = [name for parameter_id, name in trainable_parameters.items() if parameter_id not in seen]
    if missing:
        raise RuntimeError(f'Optimizer parameter assignment is missing trainable parameters: {missing[:10]}')


def _validate_optimizer_groups_against_model(optimizer, model):
    trainable_parameters = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    seen = {}
    duplicates = []

    for group in optimizer.param_groups:
        group_name = group.get('name', '<unnamed>')
        for parameter in group.get('params', []):
            parameter_id = id(parameter)
            parameter_name = trainable_parameters.get(parameter_id, '<unknown>')
            if parameter_id in seen:
                duplicates.append((parameter_name, group_name, seen[parameter_id]))
            else:
                seen[parameter_id] = group_name

    if duplicates:
        formatted = ', '.join(f'{name} ({current} vs {previous})' for name, current, previous in duplicates[:10])
        raise RuntimeError(f'Optimizer constructed duplicate parameter groups: {formatted}')

    missing = [name for parameter_id, name in trainable_parameters.items() if parameter_id not in seen]
    if missing:
        raise RuntimeError(f'Optimizer is missing trainable parameters: {missing[:10]}')


def summarize_optimizer_param_groups(optimizer):
    summaries = []
    for index, group in enumerate(optimizer.param_groups):
        params = list(group.get('params', []))
        summaries.append({
            'index': index,
            'name': group.get('name', f'group_{index}'),
            'lr': float(group.get('lr', 0.0)),
            'weight_decay': float(group.get('weight_decay', 0.0)),
            'tensor_count': len(params),
            'parameter_count': int(sum(parameter.numel() for parameter in params)),
        })
    return summaries


def _build_original_itself_stage0_optimizer(args, model):
    base_lr = float(getattr(args, 'lr', 1e-5))
    base_weight_decay = float(getattr(args, 'weight_decay', 4e-5))
    lr_factor = float(getattr(args, 'lr_factor', 5.0))
    bias_lr_factor = float(getattr(args, 'bias_lr_factor', 2.0))
    weight_decay_bias = float(getattr(args, 'weight_decay_bias', 0.0))
    optimizer_eps = float(getattr(args, 'optimizer_eps', 1e-3))

    grouped_entries = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        label, lr, weight_decay = _itself_stage0_group_spec(
            name,
            base_lr=base_lr,
            base_weight_decay=base_weight_decay,
            lr_factor=lr_factor,
            bias_lr_factor=bias_lr_factor,
            weight_decay_bias=weight_decay_bias,
        )
        group_name = f'stage0_itself::{label}'
        key = (group_name, lr, weight_decay)
        entry = grouped_entries.setdefault(
            key,
            {
                'name': group_name,
                'params': [],
                'lr': lr,
                'weight_decay': weight_decay,
                'named_params': [],
            },
        )
        entry['params'].append(parameter)
        entry['named_params'].append((name, parameter))

    named_group_entries = {entry['name']: entry['named_params'] for entry in grouped_entries.values()}
    _validate_param_group_assignment(model, named_group_entries)

    param_groups = [
        {
            'params': entry['params'],
            'lr': entry['lr'],
            'weight_decay': entry['weight_decay'],
            'name': entry['name'],
        }
        for entry in grouped_entries.values()
    ]

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(param_groups, lr=base_lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(param_groups, lr=base_lr, betas=(args.alpha, args.beta), eps=optimizer_eps)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=(args.alpha, args.beta), eps=optimizer_eps)
    else:
        raise NotImplementedError(f'Unsupported optimizer: {args.optimizer}')

    _validate_optimizer_groups_against_model(optimizer, model)
    return optimizer


def build_optimizer(args, model):
    if _use_original_itself_stage0_optimizer(args, model):
        return _build_original_itself_stage0_optimizer(args, model)

    if not hasattr(model, 'named_optimizer_groups'):
        raise AttributeError('Model must define named_optimizer_groups() for Phase E optimizer construction.')

    named_groups = model.named_optimizer_groups()
    _validate_param_group_assignment(model, named_groups)

    param_groups = []
    for group_name, named_params in named_groups.items():
        if not named_params:
            continue
        parameters = [parameter for _, parameter in named_params]
        param_groups.append({
            'params': parameters,
            'lr': _group_lr(args, group_name),
            'weight_decay': _group_weight_decay(args, group_name),
            'name': group_name,
        })

    optimizer_eps = float(getattr(args, 'optimizer_eps', 1e-8))
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(param_groups, lr=args.lr, betas=(args.alpha, args.beta), eps=optimizer_eps)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(args.alpha, args.beta), eps=optimizer_eps)
    else:
        raise NotImplementedError(f'Unsupported optimizer: {args.optimizer}')

    _validate_optimizer_groups_against_model(optimizer, model)
    return optimizer


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
