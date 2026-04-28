from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.module_group_registry import (
    LOGICAL_MODULE_GROUPS as REGISTRY_LOGICAL_MODULE_GROUPS,
    LOGICAL_MODULE_GROUP_PREFIXES,
)


LOGICAL_MODULE_GROUPS: Tuple[str, ...] = tuple(REGISTRY_LOGICAL_MODULE_GROUPS)

LOSS_WEIGHT_KEYS: Tuple[str, ...] = (
    'lambda_host',
    'lambda_bal',
    'lambda_div',
)

# Logical training groups -> parameter-name prefixes in PAS runtime models.
MODULE_GROUP_PREFIXES: Dict[str, Tuple[str, ...]] = dict(LOGICAL_MODULE_GROUP_PREFIXES)

# Logical schedule groups -> optimizer group names used by solver/build.py.
LOGICAL_TO_OPTIMIZER_GROUPS: Dict[str, Tuple[str, ...]] = {
    'host_backbone': ('image_backbone', 'text_backbone'),
    'host_retrieval': ('host_projectors',),
    'prototype_bank': ('prototype_bank',),
    'prototype_projector': ('prototype_projectors', 'class_proxies'),
    'routing': ('prototype_routing', 'prototype_contextualization'),
    'fusion': ('prototype_pooling',),
}

_PHASE_ALLOWED_KEYS = {
    'name',
    'epoch_start',
    'epoch_end',
    'trainable_groups',
    'frozen_groups',
    'lr_overrides',
    'loss_weights',
}


@dataclass(frozen=True)
class FreezePhase:
    name: str
    epoch_start: int
    epoch_end: int
    trainable_groups: Tuple[str, ...]
    frozen_groups: Tuple[str, ...]
    lr_overrides: Dict[str, float]
    loss_weights: Dict[str, float]

    def includes(self, epoch: int) -> bool:
        return self.epoch_start <= int(epoch) <= self.epoch_end


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{field_name} must be an integer, got boolean {value!r}.')
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field_name} must be an integer, got {value!r}.') from exc


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field_name} must be numeric, got {value!r}.') from exc


def _normalize_group_list(value: Any, field_name: str) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, list):
        raise ValueError(f'{field_name} must be a list of group names.')
    normalized: List[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f'{field_name}[{index}] must be a string group name.')
        group_name = item.strip()
        if group_name not in LOGICAL_MODULE_GROUPS:
            raise ValueError(
                f'Unsupported module group {group_name!r} in {field_name}. '
                f'Supported groups: {list(LOGICAL_MODULE_GROUPS)}'
            )
        if group_name in normalized:
            continue
        normalized.append(group_name)
    return tuple(normalized)


def _normalize_lr_overrides(value: Any, field_name: str) -> Dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f'{field_name} must be a mapping of module-group -> lr.')
    normalized: Dict[str, float] = {}
    for key, raw_lr in value.items():
        if not isinstance(key, str):
            raise ValueError(f'{field_name} keys must be string module-group names.')
        group_name = key.strip()
        if group_name not in LOGICAL_MODULE_GROUPS:
            raise ValueError(
                f'Unsupported lr_overrides group {group_name!r}. '
                f'Supported groups: {list(LOGICAL_MODULE_GROUPS)}'
            )
        lr_value = _coerce_float(raw_lr, f'{field_name}.{group_name}')
        if lr_value < 0:
            raise ValueError(f'{field_name}.{group_name} must be non-negative, got {lr_value}.')
        normalized[group_name] = lr_value
    return normalized


def _normalize_loss_weights(value: Any, field_name: str) -> Dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f'{field_name} must be a mapping of loss-weight overrides.')
    normalized: Dict[str, float] = {}
    for key, raw_weight in value.items():
        if not isinstance(key, str):
            raise ValueError(f'{field_name} keys must be strings.')
        weight_name = key.strip()
        if weight_name not in LOSS_WEIGHT_KEYS:
            raise ValueError(
                f'Unsupported loss weight {weight_name!r} in {field_name}. '
                f'Supported keys: {list(LOSS_WEIGHT_KEYS)}'
            )
        normalized[weight_name] = _coerce_float(raw_weight, f'{field_name}.{weight_name}')
    return normalized


def parse_freeze_schedule_config(raw_schedule: Any, num_epoch: Optional[int] = None) -> List[FreezePhase]:
    if raw_schedule in (None, []):
        return []
    if not isinstance(raw_schedule, list):
        raise ValueError('training.freeze_schedule must be a list of phase mappings.')

    parsed: List[FreezePhase] = []
    for phase_index, phase_data in enumerate(raw_schedule):
        if not isinstance(phase_data, dict):
            raise ValueError(f'training.freeze_schedule[{phase_index}] must be a mapping.')

        unknown_keys = sorted(set(phase_data.keys()) - _PHASE_ALLOWED_KEYS)
        if unknown_keys:
            raise ValueError(
                f'training.freeze_schedule[{phase_index}] has unknown keys {unknown_keys}. '
                f'Allowed keys: {sorted(_PHASE_ALLOWED_KEYS)}'
            )

        name = str(phase_data.get('name') or f'phase_{phase_index + 1}').strip()
        if not name:
            raise ValueError(f'training.freeze_schedule[{phase_index}].name cannot be empty.')

        epoch_start = _coerce_int(phase_data.get('epoch_start'), f'training.freeze_schedule[{phase_index}].epoch_start')
        epoch_end = _coerce_int(phase_data.get('epoch_end'), f'training.freeze_schedule[{phase_index}].epoch_end')
        if epoch_end < epoch_start:
            raise ValueError(
                f'training.freeze_schedule[{phase_index}] has epoch_end < epoch_start: '
                f'{epoch_end} < {epoch_start}.'
            )

        trainable_groups = _normalize_group_list(
            phase_data.get('trainable_groups', []),
            f'training.freeze_schedule[{phase_index}].trainable_groups',
        )
        frozen_groups = _normalize_group_list(
            phase_data.get('frozen_groups', []),
            f'training.freeze_schedule[{phase_index}].frozen_groups',
        )
        overlap = sorted(set(trainable_groups).intersection(frozen_groups))
        if overlap:
            raise ValueError(
                f'training.freeze_schedule[{phase_index}] cannot list the same group as trainable and frozen: {overlap}.'
            )

        lr_overrides = _normalize_lr_overrides(
            phase_data.get('lr_overrides', {}),
            f'training.freeze_schedule[{phase_index}].lr_overrides',
        )
        loss_weights = _normalize_loss_weights(
            phase_data.get('loss_weights', {}),
            f'training.freeze_schedule[{phase_index}].loss_weights',
        )
        if not trainable_groups and not frozen_groups and not lr_overrides and not loss_weights:
            raise ValueError(
                f'training.freeze_schedule[{phase_index}] is empty. '
                'Specify at least one of trainable_groups, frozen_groups, lr_overrides, or loss_weights.'
            )

        parsed.append(
            FreezePhase(
                name=name,
                epoch_start=epoch_start,
                epoch_end=epoch_end,
                trainable_groups=trainable_groups,
                frozen_groups=frozen_groups,
                lr_overrides=lr_overrides,
                loss_weights=loss_weights,
            )
        )

    zero_based = any(phase.epoch_start == 0 for phase in parsed)
    normalized: List[FreezePhase] = []
    for phase in parsed:
        start = phase.epoch_start + 1 if zero_based else phase.epoch_start
        end = phase.epoch_end + 1 if zero_based else phase.epoch_end
        if start < 1:
            raise ValueError(
                'training.freeze_schedule epochs must be >= 1 (or zero-based starting at 0 for every phase).'
            )
        if num_epoch is not None and end > int(num_epoch):
            raise ValueError(
                f'training.freeze_schedule phase `{phase.name}` ends at epoch {end}, '
                f'but training.epochs={int(num_epoch)}.'
            )
        normalized.append(
            FreezePhase(
                name=phase.name,
                epoch_start=start,
                epoch_end=end,
                trainable_groups=phase.trainable_groups,
                frozen_groups=phase.frozen_groups,
                lr_overrides=copy.deepcopy(phase.lr_overrides),
                loss_weights=copy.deepcopy(phase.loss_weights),
            )
        )

    normalized.sort(key=lambda phase: (phase.epoch_start, phase.epoch_end, phase.name))
    previous: Optional[FreezePhase] = None
    for phase in normalized:
        if previous is not None and phase.epoch_start <= previous.epoch_end:
            raise ValueError(
                'training.freeze_schedule contains overlapping phases: '
                f'`{previous.name}` ({previous.epoch_start}-{previous.epoch_end}) overlaps with '
                f'`{phase.name}` ({phase.epoch_start}-{phase.epoch_end}).'
            )
        previous = phase
    return normalized


def get_active_phase(phases: Sequence[FreezePhase], epoch: int) -> Optional[FreezePhase]:
    for phase in phases:
        if phase.includes(epoch):
            return phase
    return None


def _parameter_matches_group(parameter_name: str, group_name: str) -> bool:
    prefixes = MODULE_GROUP_PREFIXES[group_name]
    return any(parameter_name.startswith(prefix) for prefix in prefixes)


def get_group_trainability_snapshot(model, groups: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, int]]:
    selected_groups = tuple(groups) if groups is not None else LOGICAL_MODULE_GROUPS
    snapshot: Dict[str, Dict[str, int]] = {}
    for group_name in selected_groups:
        total_tensors = 0
        trainable_tensors = 0
        total_params = 0
        trainable_params = 0
        for parameter_name, parameter in model.named_parameters():
            if not _parameter_matches_group(parameter_name, group_name):
                continue
            total_tensors += 1
            tensor_params = int(parameter.numel())
            total_params += tensor_params
            if parameter.requires_grad:
                trainable_tensors += 1
                trainable_params += tensor_params
        snapshot[group_name] = {
            'trainable_tensors': trainable_tensors,
            'total_tensors': total_tensors,
            'trainable_params': trainable_params,
            'total_params': total_params,
        }
    return snapshot


def set_group_requires_grad(model, group_name: str, requires_grad: bool) -> int:
    touched = 0
    for parameter_name, parameter in model.named_parameters():
        if _parameter_matches_group(parameter_name, group_name):
            parameter.requires_grad = bool(requires_grad)
            touched += 1
    return touched


def apply_phase_trainability(model, phase: FreezePhase) -> Dict[str, Dict[str, int]]:
    summary = {'trainable': {}, 'frozen': {}}
    for group_name in phase.trainable_groups:
        summary['trainable'][group_name] = set_group_requires_grad(model, group_name, True)
    for group_name in phase.frozen_groups:
        summary['frozen'][group_name] = set_group_requires_grad(model, group_name, False)
    return summary


def apply_loss_weight_overrides(model, args, loss_weights: Dict[str, float]) -> Dict[str, float]:
    applied: Dict[str, float] = {}
    if not loss_weights:
        return applied

    prototype_losses = getattr(getattr(model, 'prototype_head', None), 'losses', None)
    for weight_name, value in loss_weights.items():
        scalar = float(value)
        setattr(args, weight_name, scalar)
        if weight_name == 'lambda_host' and hasattr(model, 'lambda_host'):
            setattr(model, 'lambda_host', scalar)
            host_loss_enabled = bool(scalar > 0.0)
            setattr(args, 'use_host_loss', host_loss_enabled)
            if hasattr(model, 'use_host_loss'):
                setattr(model, 'use_host_loss', host_loss_enabled)
            host_head = getattr(model, 'host_head', None)
            if host_head is not None and hasattr(host_head, 'use_host_loss'):
                setattr(host_head, 'use_host_loss', host_loss_enabled)
            # PAS CLIP host adapter path: host_head.core.losses.use_loss_ret
            host_head_core = getattr(host_head, 'core', None) if host_head is not None else None
            host_head_core_losses = getattr(host_head_core, 'losses', None) if host_head_core is not None else None
            if host_head_core_losses is not None and hasattr(host_head_core_losses, 'use_loss_ret'):
                host_head_core_losses.use_loss_ret = host_loss_enabled
            # Host-only CLIP model path: model.losses.use_loss_ret
            model_losses = getattr(model, 'losses', None)
            if model_losses is not None and hasattr(model_losses, 'use_loss_ret'):
                model_losses.use_loss_ret = host_loss_enabled
            applied[weight_name] = scalar
            continue
        if prototype_losses is None:
            continue
        target_attr = {
            'lambda_bal': 'lambda_bal',
            'lambda_div': 'lambda_div',
        }.get(weight_name)
        if target_attr is None or not hasattr(prototype_losses, target_attr):
            continue
        setattr(prototype_losses, target_attr, scalar)
        if weight_name == 'lambda_bal' and hasattr(prototype_losses, 'use_balance_loss'):
            prototype_losses.use_balance_loss = bool(scalar > 0.0)
            setattr(args, 'use_balancing_loss', prototype_losses.use_balance_loss)
        elif weight_name == 'lambda_div' and hasattr(prototype_losses, 'use_diversity_loss'):
            prototype_losses.use_diversity_loss = bool(scalar > 0.0)
            setattr(args, 'use_diversity_loss', prototype_losses.use_diversity_loss)
        applied[weight_name] = scalar
    return applied


def apply_optimizer_lr_overrides(optimizer, lr_overrides: Dict[str, float]) -> Dict[str, int]:
    applied_counts: Dict[str, int] = {}
    for group_name, lr_value in lr_overrides.items():
        optimizer_group_names = LOGICAL_TO_OPTIMIZER_GROUPS[group_name]
        touched = 0
        for param_group in optimizer.param_groups:
            current_group_name = str(param_group.get('name', ''))
            if current_group_name in optimizer_group_names:
                param_group['lr'] = float(lr_value)
                param_group['initial_lr'] = float(lr_value)
                touched += 1
        applied_counts[group_name] = touched
    return applied_counts
