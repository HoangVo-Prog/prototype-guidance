"""Host router for model construction."""

from __future__ import annotations

import torch.nn as nn

from .hosts import build_clip_host, build_itself_host


class PASModel(nn.Module):
    """Compatibility wrapper that delegates to the host-specific implementation."""

    def __init__(self, args, num_classes, **kwargs):
        super().__init__()
        self.impl = _build_host_model(args, num_classes, **kwargs)

    def forward(self, *args, **kwargs):
        return self.impl(*args, **kwargs)


def _build_host_model(args, num_classes, **kwargs):
    host_type = str(getattr(args, 'host_type', 'clip')).lower()
    if host_type == 'itself':
        return build_itself_host(args=args, num_classes=num_classes, **kwargs)
    if host_type == 'clip':
        return build_clip_host(args=args, num_classes=num_classes, **kwargs)
    raise ValueError(f'Unsupported host.type={host_type!r}. Expected one of: clip, itself.')


def build_model(args, num_classes, **kwargs):
    """Construct model implementation from `host.type` only."""

    return _build_host_model(args=args, num_classes=num_classes, **kwargs)
