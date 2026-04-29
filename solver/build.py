import torch

from model.hosts import (
    build_original_itself_lr_scheduler,
    build_original_itself_optimizer,
    should_use_original_itself_runtime,
)
from .lr_scheduler import LRSchedulerWithWarmup


_OWNERSHIP_BY_GROUP = {
    'prototype_bank': 'prototype_added',
    'prototype_projectors': 'prototype_added',
    'prototype_routing': 'prototype_added',
    'prototype_pooling': 'prototype_added',
    'prototype_contextualization': 'prototype_added',
    'class_proxies': 'prototype_added',
    'host_projectors': 'itself_original',
    'image_backbone': 'itself_original',
    'text_backbone': 'itself_original',
    'other': 'pas_wrapper',
}


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

ITSELF_LEGACY_PARAMWISE_GROUPS = frozenset(
    {
        'host_projectors',
        'image_backbone',
        'text_backbone',
    }
)


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
    return should_use_original_itself_runtime(args)


def _itself_stage0_group_spec(
    name: str,
    base_lr: float,
    base_weight_decay: float,
    lr_factor: float,
    bias_lr_factor: float,
    weight_decay_bias: float,
    grab_embedding_lr: float = 1e-3,
):
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
        lr = float(grab_embedding_lr)
        label = 'visual_embedding_bias' if 'bias' in name else 'visual_embedding'
    if 'texual_emb_layer' in name or 'textual_embedding_layer' in name:
        lr = float(grab_embedding_lr)
        label = 'textual_embedding_bias' if 'bias' in name else 'textual_embedding'

    return label, lr, weight_decay


def _itself_stage0_group_spec_trace(
    name: str,
    base_lr: float,
    base_weight_decay: float,
    lr_factor: float,
    bias_lr_factor: float,
    weight_decay_bias: float,
    grab_embedding_lr: float = 1e-3,
):
    lr = base_lr
    weight_decay = base_weight_decay
    label = 'base'
    rule_hits = []
    multiplier_chain = []

    if 'cross' in name:
        lr = base_lr * lr_factor
        label = 'cross'
        rule_hits.append('matched_substring:cross')
        multiplier_chain.append(f'lr_factor={float(lr_factor):g}')
    if 'bias' in name:
        lr = base_lr * bias_lr_factor
        weight_decay = weight_decay_bias
        label = 'bias'
        rule_hits.append('matched_substring:bias')
        multiplier_chain.append(f'bias_lr_factor={float(bias_lr_factor):g}')
        multiplier_chain.append(f'weight_decay_bias={float(weight_decay_bias):g}')
    if 'classifier' in name or 'mlm_head' in name:
        lr = base_lr * lr_factor
        label = 'classifier_bias' if 'bias' in name else 'classifier'
        rule_hits.append('matched_substring:classifier_or_mlm_head')
        multiplier_chain.append(f'lr_factor={float(lr_factor):g}')
    if 'visul_emb_layer' in name or 'visual_embedding_layer' in name:
        lr = float(grab_embedding_lr)
        label = 'visual_embedding_bias' if 'bias' in name else 'visual_embedding'
        rule_hits.append('matched_substring:visual_embedding_layer')
        multiplier_chain.append(f'hardcoded_visual_embedding_lr={float(grab_embedding_lr):g}')
    if 'texual_emb_layer' in name or 'textual_embedding_layer' in name:
        lr = float(grab_embedding_lr)
        label = 'textual_embedding_bias' if 'bias' in name else 'textual_embedding'
        rule_hits.append('matched_substring:textual_embedding_layer')
        multiplier_chain.append(f'hardcoded_textual_embedding_lr={float(grab_embedding_lr):g}')
    if not rule_hits:
        rule_hits.append('direct_unsplit_base_rule')
    if not multiplier_chain:
        multiplier_chain.append('none')
    # Keep only first occurrence order.
    dedup_chain = list(dict.fromkeys(multiplier_chain))
    dedup_rules = list(dict.fromkeys(rule_hits))
    return {
        'label': label,
        'lr': float(lr),
        'weight_decay': float(weight_decay),
        'rule_hits': dedup_rules,
        'multiplier_chain': dedup_chain,
    }


def _should_apply_itself_legacy_paramwise_lr_policy(args, group_name: str) -> bool:
    return (
        str(group_name) in ITSELF_LEGACY_PARAMWISE_GROUPS
        and str(getattr(args, 'host_type', 'clip')).lower() == 'itself'
    )


def _build_itself_legacy_paramwise_groups(args, group_name: str, named_params):
    base_lr = _group_lr(args, group_name)
    base_weight_decay = _group_weight_decay(args, group_name)
    lr_factor = float(getattr(args, 'lr_factor', 5.0))
    bias_lr_factor = float(getattr(args, 'bias_lr_factor', 2.0))
    weight_decay_bias = float(getattr(args, 'weight_decay_bias', 0.0))
    grab_embedding_lr = float(getattr(args, 'itself_grab_embedding_lr', 1e-3))

    grouped_params = {}
    for name, parameter in named_params:
        _, lr, weight_decay = _itself_stage0_group_spec(
            name=name,
            base_lr=base_lr,
            base_weight_decay=base_weight_decay,
            lr_factor=lr_factor,
            bias_lr_factor=bias_lr_factor,
            weight_decay_bias=weight_decay_bias,
            grab_embedding_lr=grab_embedding_lr,
        )
        key = (float(lr), float(weight_decay))
        bucket = grouped_params.setdefault(key, [])
        bucket.append(parameter)

    param_groups = []
    for (lr, weight_decay), parameters in grouped_params.items():
        param_groups.append({
            'params': parameters,
            'lr': lr,
            'weight_decay': weight_decay,
            'name': group_name,
        })
    return param_groups


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


def _compress_parameter_prefixes(parameter_names, max_entries: int = 4):
    if not parameter_names:
        return []
    histogram = {}
    for name in parameter_names:
        parts = str(name).split('.')
        if len(parts) >= 3:
            prefix = '.'.join(parts[:3]) + '.*'
        elif len(parts) >= 2:
            prefix = '.'.join(parts[:2]) + '.*'
        else:
            prefix = parts[0]
        histogram[prefix] = int(histogram.get(prefix, 0)) + 1
    ranked = sorted(histogram.items(), key=lambda item: (-item[1], item[0]))
    return [f'{prefix}({count})' for prefix, count in ranked[:max_entries]]


def _group_ownership_tag(group_name: str, split_applied: bool):
    base = _OWNERSHIP_BY_GROUP.get(str(group_name), 'mixed_or_ambiguous')
    if split_applied and base != 'itself_original':
        return 'mixed_or_ambiguous'
    return base


def _declared_and_effective_base_lr(args, group_name: str):
    lr_attr = GROUP_TO_LR_ATTR[group_name]
    declared_raw = _coerce_optional_float(getattr(args, lr_attr, None), lr_attr)
    uses_global_fallback = declared_raw is None or float(declared_raw) < 0.0
    base_lr_declared = float(args.lr) if uses_global_fallback else float(declared_raw)
    source_key = f'optimizer.{lr_attr}'
    if uses_global_fallback:
        source_key = f'optimizer.{lr_attr} -> optimizer.lr'
    return source_key, base_lr_declared


def _reconstruct_optimizer_observability_rows(args, model):
    if not hasattr(model, 'named_optimizer_groups'):
        return []
    named_groups = model.named_optimizer_groups()
    rows = []
    for group_name, named_params in named_groups.items():
        if not named_params:
            continue
        source_key, base_lr_declared = _declared_and_effective_base_lr(args, group_name)
        if _should_apply_itself_legacy_paramwise_lr_policy(args, group_name):
            base_lr = _group_lr(args, group_name)
            base_weight_decay = _group_weight_decay(args, group_name)
            lr_factor = float(getattr(args, 'lr_factor', 5.0))
            bias_lr_factor = float(getattr(args, 'bias_lr_factor', 2.0))
            weight_decay_bias = float(getattr(args, 'weight_decay_bias', 0.0))
            grab_embedding_lr = float(getattr(args, 'itself_grab_embedding_lr', 1e-3))
            buckets = {}
            for parameter_name, parameter in named_params:
                trace = _itself_stage0_group_spec_trace(
                    name=parameter_name,
                    base_lr=base_lr,
                    base_weight_decay=base_weight_decay,
                    lr_factor=lr_factor,
                    bias_lr_factor=bias_lr_factor,
                    weight_decay_bias=weight_decay_bias,
                    grab_embedding_lr=grab_embedding_lr,
                )
                key = (float(trace['lr']), float(trace['weight_decay']))
                bucket = buckets.setdefault(
                    key,
                    {
                        'parameter_names': [],
                        'tensor_count': 0,
                        'parameter_count': 0,
                        'labels': [],
                        'rule_hits': [],
                        'multiplier_chain': [],
                    },
                )
                bucket['parameter_names'].append(str(parameter_name))
                bucket['tensor_count'] += 1
                bucket['parameter_count'] += int(parameter.numel())
                bucket['labels'].append(str(trace['label']))
                bucket['rule_hits'].extend(list(trace['rule_hits']))
                bucket['multiplier_chain'].extend(list(trace['multiplier_chain']))
            for (final_lr, final_wd), bucket in buckets.items():
                labels = list(dict.fromkeys(bucket['labels']))
                if len(labels) == 1:
                    bucket_label = labels[0]
                else:
                    bucket_label = 'mixed:' + '|'.join(sorted(labels))
                rows.append(
                    {
                        'name': str(group_name),
                        'derived_bucket_label': bucket_label,
                        'ownership_tag': _group_ownership_tag(group_name, split_applied=True),
                        'lr_source_key': source_key,
                        'base_lr_declared': float(base_lr_declared),
                        'multiplier_chain': list(dict.fromkeys(bucket['multiplier_chain'])),
                        'rule_hits': list(dict.fromkeys(bucket['rule_hits'])),
                        'lr': float(final_lr),
                        'weight_decay': float(final_wd),
                        'tensor_count': int(bucket['tensor_count']),
                        'parameter_count': int(bucket['parameter_count']),
                        'parameter_name_samples': bucket['parameter_names'][:6],
                        'prefix_summary': _compress_parameter_prefixes(bucket['parameter_names']),
                        'logical_group_name': str(group_name),
                        'split_applied': True,
                    }
                )
            continue

        parameter_names = [str(name) for name, _ in named_params]
        rows.append(
            {
                'name': str(group_name),
                'derived_bucket_label': 'direct',
                'ownership_tag': _group_ownership_tag(group_name, split_applied=False),
                'lr_source_key': source_key,
                'base_lr_declared': float(base_lr_declared),
                'multiplier_chain': ['none'],
                'rule_hits': ['direct_unsplit_group'],
                'lr': float(_group_lr(args, group_name)),
                'weight_decay': float(_group_weight_decay(args, group_name)),
                'tensor_count': int(len(named_params)),
                'parameter_count': int(sum(parameter.numel() for _, parameter in named_params)),
                'parameter_name_samples': parameter_names[:6],
                'prefix_summary': _compress_parameter_prefixes(parameter_names),
                'logical_group_name': str(group_name),
                'split_applied': False,
            }
        )
    return rows


def summarize_optimizer_param_groups_observability(args, model, optimizer):
    summaries = summarize_optimizer_param_groups(optimizer)
    reconstructed = _reconstruct_optimizer_observability_rows(args=args, model=model)
    for index, group_summary in enumerate(summaries):
        if index < len(reconstructed):
            group_summary.update(reconstructed[index])
        group_summary['group_id'] = f'group_{index:02d}'
    return summaries


def summarize_scheduler_effective_lrs(optimizer, scheduler):
    effective_lrs = list(scheduler.get_lr())
    rows = []
    for index, group in enumerate(optimizer.param_groups):
        rows.append(
            {
                'group_id': f'group_{index:02d}',
                'index': index,
                'name': str(group.get('name', f'group_{index}')),
                'scheduler_effective_lr': float(effective_lrs[index]) if index < len(effective_lrs) else 0.0,
                'optimizer_initial_lr': float(group.get('initial_lr', group.get('lr', 0.0))),
                'optimizer_current_lr': float(group.get('lr', 0.0)),
            }
        )
    return rows


def summarize_config_declared_optimizer_settings(args):
    lr_fields = (
        'lr',
        'lr_prototype_bank',
        'lr_projectors',
        'lr_prototype_routing',
        'lr_prototype_pooling',
        'lr_prototype_contextualization',
        'lr_host_projectors',
        'lr_class_proxies',
        'lr_image_backbone',
        'lr_text_backbone',
    )
    wd_fields = (
        'weight_decay',
        'weight_decay_projectors',
        'weight_decay_prototype_bank',
        'weight_decay_prototype_routing',
        'weight_decay_prototype_pooling',
        'weight_decay_prototype_contextualization',
        'weight_decay_host_projectors',
        'weight_decay_class_proxies',
        'weight_decay_image_backbone',
        'weight_decay_text_backbone',
        'weight_decay_bias',
    )
    declared_lrs = {name: getattr(args, name, None) for name in lr_fields}
    declared_wds = {name: getattr(args, name, None) for name in wd_fields}
    freeze_schedule = getattr(args, 'freeze_schedule', None)
    schedule_lr_overrides = []
    if isinstance(freeze_schedule, list):
        for phase in freeze_schedule:
            if not isinstance(phase, dict):
                continue
            phase_name = str(phase.get('name', '<unnamed_phase>'))
            lr_overrides = phase.get('lr_overrides')
            if isinstance(lr_overrides, dict) and lr_overrides:
                schedule_lr_overrides.append({'phase': phase_name, 'lr_overrides': dict(lr_overrides)})
    return {
        'optimizer_type': getattr(args, 'optimizer', None),
        'declared_lrs': declared_lrs,
        'declared_weight_decays': declared_wds,
        'lr_factor': getattr(args, 'lr_factor', None),
        'bias_lr_factor': getattr(args, 'bias_lr_factor', None),
        'optimizer_eps': getattr(args, 'optimizer_eps', None),
        'freeze_schedule_lr_overrides': schedule_lr_overrides,
    }


def _build_original_itself_stage0_optimizer(args, model):
    optimizer = build_original_itself_optimizer(args, model)
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
        if _should_apply_itself_legacy_paramwise_lr_policy(args, group_name):
            param_groups.extend(
                _build_itself_legacy_paramwise_groups(
                    args=args,
                    group_name=group_name,
                    named_params=named_params,
                )
            )
            continue

        parameters = [parameter for _, parameter in named_params]
        param_groups.append(
            {
                'params': parameters,
                'lr': _group_lr(args, group_name),
                'weight_decay': _group_weight_decay(args, group_name),
                'name': group_name,
            }
        )

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
    if should_use_original_itself_runtime(args):
        return build_original_itself_lr_scheduler(args, optimizer)
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        lr_decay_epochs=getattr(args, 'lr_decay_epochs', args.num_epoch),
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )

