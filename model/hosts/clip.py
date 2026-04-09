"""CLIP host placeholder for the host router."""

from __future__ import annotations

import torch.nn as nn


class ClipHostPlaceholder(nn.Module):
    def __init__(self, args, num_classes, **kwargs):
        super().__init__()
        del kwargs
        self.args = args
        self.num_classes = num_classes

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            'host.type=clip is currently a placeholder in this phase. '
            'Use host.type=itself for ITSELF-only replication.'
        )


def build_clip_host(args, num_classes, **kwargs):
    return ClipHostPlaceholder(args=args, num_classes=num_classes, **kwargs)
