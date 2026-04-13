from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch


LOGICAL_MODULE_GROUP_PREFIXES: Dict[str, Tuple[str, ...]] = {
    'host_backbone': (
        'base_model.visual',
        'base_model.transformer',
        'base_model.token_embedding',
        'base_model.positional_embedding',
        'base_model.ln_final',
        'base_model.text_projection',
    ),
    'host_retrieval': (
        'host_head',
    ),
    'prototype_bank': (
        'prototype_head.prototype_bank',
    ),
    'prototype_projector': (
        'prototype_head.image_projector',
        'prototype_head.text_projector',
        'prototype_head.proto_query_proj',
        'prototype_head.image_adapter',
        'prototype_head.text_adapter',
        'prototype_head.losses.class_proxies',
    ),
    'routing': (
        'prototype_head.router',
        'prototype_head.local_routing_adapter',
        'prototype_head.contextualizer',
    ),
    'fusion': (
        'prototype_head.text_pool_query',
        'prototype_head.token_pooler',
        'prototype_head.token_scorer',
        'prototype_head.token_mask_builder',
        'prototype_head.aggregator',
        'fusion_module',
    ),
}

LOGICAL_MODULE_GROUPS: Tuple[str, ...] = tuple(LOGICAL_MODULE_GROUP_PREFIXES.keys())

CHECKPOINT_GROUP_MEMBERS: Dict[str, Tuple[str, ...]] = {
    'host': ('host_backbone', 'host_retrieval'),
    'prototype_bank': ('prototype_bank',),
    'prototype_projector': ('prototype_projector',),
    'fusion': ('fusion',),
}

CHECKPOINT_GROUPS: Tuple[str, ...] = tuple(CHECKPOINT_GROUP_MEMBERS.keys())


def _normalize_checkpoint_key(key: str) -> str:
    normalized = str(key)
    if normalized.startswith('module.'):
        normalized = normalized[len('module.'):]
    if normalized.startswith('model.'):
        normalized = normalized[len('model.'):]
    return normalized


def _belongs_to_prefixes(parameter_name: str, prefixes: Iterable[str]) -> bool:
    name = str(parameter_name)
    return any(name.startswith(prefix) for prefix in prefixes)


def get_prefixes_for_logical_group(group_name: str) -> Tuple[str, ...]:
    if group_name not in LOGICAL_MODULE_GROUP_PREFIXES:
        raise ValueError(f'Unsupported logical module group `{group_name}`.')
    return LOGICAL_MODULE_GROUP_PREFIXES[group_name]


def get_prefixes_for_checkpoint_group(group_name: str) -> Tuple[str, ...]:
    if group_name not in CHECKPOINT_GROUP_MEMBERS:
        raise ValueError(
            f'Unsupported checkpoint group `{group_name}`. '
            f'Allowed: {list(CHECKPOINT_GROUP_MEMBERS.keys())}'
        )
    prefixes: List[str] = []
    for logical_group in CHECKPOINT_GROUP_MEMBERS[group_name]:
        prefixes.extend(get_prefixes_for_logical_group(logical_group))
    return tuple(prefixes)


def list_group_keys_from_model_state(model_state_dict: Dict[str, torch.Tensor], group_name: str) -> List[str]:
    prefixes = get_prefixes_for_checkpoint_group(group_name)
    return [key for key in model_state_dict.keys() if _belongs_to_prefixes(key, prefixes)]


def get_group_state_dict(model, group_name: str) -> Dict[str, torch.Tensor]:
    model_state_dict = model.state_dict()
    group_keys = list_group_keys_from_model_state(model_state_dict, group_name)
    return {key: model_state_dict[key].detach().cpu() for key in group_keys}


def load_group_state_dict(model, group_name: str, state_dict: Dict[str, torch.Tensor], strict: bool = True):
    model_state_dict = model.state_dict()
    target_group_keys = set(list_group_keys_from_model_state(model_state_dict, group_name))

    loaded_group_state = {}
    shape_mismatches = []
    candidate_group_keys = set()
    for raw_key, value in state_dict.items():
        normalized_key = _normalize_checkpoint_key(raw_key)
        if normalized_key in target_group_keys:
            candidate_group_keys.add(normalized_key)
            model_tensor = model_state_dict[normalized_key]
            if tuple(model_tensor.shape) != tuple(value.shape):
                shape_mismatches.append(
                    f'{normalized_key}: ckpt{tuple(value.shape)} != model{tuple(model_tensor.shape)}'
                )
                continue
            loaded_group_state[normalized_key] = value.detach().clone()
            continue

        # Track keys that look like they belong to this checkpoint group but do not map to this model.
        if _belongs_to_prefixes(normalized_key, get_prefixes_for_checkpoint_group(group_name)):
            candidate_group_keys.add(normalized_key)

    missing_keys = sorted(target_group_keys - set(loaded_group_state.keys()))
    unexpected_keys = sorted(candidate_group_keys - target_group_keys)

    if strict and (missing_keys or unexpected_keys or shape_mismatches):
        raise RuntimeError(
            'Strict checkpoint load failed for group `{}`: missing_keys={} unexpected_keys={} shape_mismatches={}'.format(
                group_name,
                len(missing_keys),
                len(unexpected_keys),
                len(shape_mismatches),
            )
        )

    updated_state_dict = dict(model_state_dict)
    updated_state_dict.update(loaded_group_state)
    model.load_state_dict(updated_state_dict, strict=False)

    return {
        'group_name': group_name,
        'loaded_keys': sorted(loaded_group_state.keys()),
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
        'shape_mismatches': shape_mismatches,
    }
