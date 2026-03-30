import torch

from .lr_scheduler import LRSchedulerWithWarmup


GROUP_TO_LR_ATTR = {
    'prototype_bank': 'lr_prototype_bank',
    'contextualizer': 'lr_contextualizer',
    'projectors': 'lr_projectors',
    'logit_scale': 'lr_logit_scale',
    'image_backbone': 'lr_image_backbone',
    'text_backbone': 'lr_text_backbone',
    'other': 'lr',
}

GROUP_TO_WD_ATTR = {
    'prototype_bank': 'weight_decay_prototype_bank',
    'contextualizer': 'weight_decay_contextualizer',
    'projectors': 'weight_decay_projectors',
    'logit_scale': 'weight_decay_logit_scale',
    'image_backbone': 'weight_decay_image_backbone',
    'text_backbone': 'weight_decay_text_backbone',
    'other': 'weight_decay',
}


def _group_lr(args, group_name: str) -> float:
    attr = GROUP_TO_LR_ATTR[group_name]
    value = getattr(args, attr, 0.0)
    if value is None or value < 0:
        return float(args.lr)
    return float(value)


def _group_weight_decay(args, group_name: str) -> float:
    attr = GROUP_TO_WD_ATTR[group_name]
    value = getattr(args, attr, None)
    if value is None:
        return float(args.weight_decay)
    return float(value)


def build_optimizer(args, model):
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
