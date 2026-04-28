"""Model builder routing for host-only and prototype-enabled runtime paths."""

from __future__ import annotations

import torch.nn as nn

from .hosts import build_clip_host, build_itself_host
from . import pas_model as _pas_model
from .plug_and_play import build_structural_split_model
from .runtime_modes import (
    RUNTIME_MODE_AUTO,
    RUNTIME_MODE_HOST_ONLY,
    normalize_runtime_mode,
    resolve_runtime_mode_from_args,
)

build_CLIP_from_openai_pretrained = _pas_model.build_CLIP_from_openai_pretrained
convert_weights = _pas_model.convert_weights


def _should_use_pas_model(args) -> bool:
    explicit = getattr(args, 'use_prototype_branch', None)
    if explicit is not None:
        return bool(explicit)
    return bool(
        getattr(args, 'use_prototype_bank', False)
        or getattr(args, 'use_image_conditioned_pooling', False)
    )


def _build_host_only_model(args, num_classes, **kwargs):
    host_type = str(getattr(args, 'host_type', 'clip')).lower()
    if host_type == 'itself':
        return build_itself_host(args=args, num_classes=num_classes, **kwargs)
    if host_type == 'clip':
        return build_clip_host(args=args, num_classes=num_classes, **kwargs)
    raise ValueError(f'Unsupported host.type={host_type!r}. Expected one of: clip, itself.')


def _build_model_impl(args, num_classes, **kwargs):
    runtime_mode = normalize_runtime_mode(getattr(args, 'runtime_mode', RUNTIME_MODE_AUTO))
    if runtime_mode == RUNTIME_MODE_HOST_ONLY and not _should_use_pas_model(args):
        return _build_host_only_model(args=args, num_classes=num_classes, **kwargs)

    if _should_use_pas_model(args):
        resolved_runtime_mode = resolve_runtime_mode_from_args(args, for_training=bool(getattr(args, 'training', True)))
        if resolved_runtime_mode != RUNTIME_MODE_AUTO:
            return build_structural_split_model(args=args, num_classes=num_classes, **kwargs)
        _pas_model.build_CLIP_from_openai_pretrained = build_CLIP_from_openai_pretrained
        _pas_model.convert_weights = convert_weights
        return _pas_model.build_model(args=args, num_classes=num_classes, **kwargs)
    return _build_host_only_model(args=args, num_classes=num_classes, **kwargs)


class PASModel(nn.Module):
    """Compatibility wrapper exposing a uniform model class for callers/tests."""

    def __init__(self, args, num_classes, **kwargs):
        super().__init__()
        self.impl = _build_model_impl(args=args, num_classes=num_classes, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            impl = self.__dict__.get('_modules', {}).get('impl')
            if impl is not None and hasattr(impl, name):
                return getattr(impl, name)
            raise exc

    def forward(self, *args, **kwargs):
        return self.impl(*args, **kwargs)


def build_model(args, num_classes, **kwargs):
    """Build the concrete model implementation for the current runtime config."""

    return _build_model_impl(args=args, num_classes=num_classes, **kwargs)


__all__ = [
    'PASModel',
    'build_model',
    'build_CLIP_from_openai_pretrained',
    'convert_weights',
]
