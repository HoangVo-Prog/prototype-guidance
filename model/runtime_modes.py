from __future__ import annotations

from typing import Optional


RUNTIME_MODE_AUTO = 'auto'
RUNTIME_MODE_HOST_ONLY = 'host_only'
RUNTIME_MODE_PROTOTYPE_ONLY = 'prototype_only'
RUNTIME_MODE_FUSED_EXTERNAL = 'fused_external'
RUNTIME_MODE_JOINT_TRAINING = 'joint_training'
RUNTIME_MODE_CALIBRATION_ONLY = 'calibration_only'

RUNTIME_MODES = (
    RUNTIME_MODE_AUTO,
    RUNTIME_MODE_HOST_ONLY,
    RUNTIME_MODE_PROTOTYPE_ONLY,
    RUNTIME_MODE_FUSED_EXTERNAL,
    RUNTIME_MODE_JOINT_TRAINING,
    RUNTIME_MODE_CALIBRATION_ONLY,
)


def normalize_runtime_mode(value: Optional[str]) -> str:
    mode = str(value or RUNTIME_MODE_AUTO).strip().lower()
    if mode not in RUNTIME_MODES:
        raise ValueError(
            f'Unsupported runtime_mode={value!r}. Allowed values: {list(RUNTIME_MODES)}'
        )
    return mode


def resolve_runtime_mode_from_args(args, *, for_training: bool) -> str:
    explicit = normalize_runtime_mode(getattr(args, 'runtime_mode', RUNTIME_MODE_AUTO))
    if explicit != RUNTIME_MODE_AUTO:
        return explicit

    use_prototype_branch = bool(getattr(args, 'use_prototype_branch', False))
    if not use_prototype_branch:
        return RUNTIME_MODE_HOST_ONLY
    if for_training:
        return RUNTIME_MODE_JOINT_TRAINING
    return RUNTIME_MODE_FUSED_EXTERNAL


def runtime_mode_uses_prototype(mode: str) -> bool:
    normalized = normalize_runtime_mode(mode)
    return normalized in {
        RUNTIME_MODE_PROTOTYPE_ONLY,
        RUNTIME_MODE_FUSED_EXTERNAL,
        RUNTIME_MODE_JOINT_TRAINING,
        RUNTIME_MODE_CALIBRATION_ONLY,
    }
