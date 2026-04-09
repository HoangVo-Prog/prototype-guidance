"""Model builder routing for host-only and prototype-enabled runtime paths."""

from __future__ import annotations

import torch.nn as nn

from .hosts import build_clip_host, build_itself_host
from . import pas_model as _pas_model

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
    if _should_use_pas_model(args):
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
