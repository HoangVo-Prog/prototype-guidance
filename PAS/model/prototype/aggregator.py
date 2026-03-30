from typing import Dict

import torch
import torch.nn as nn


class PrototypeAggregator(nn.Module):
    def forward(self, routing_weights: torch.Tensor, prototypes: torch.Tensor, return_debug: bool = False):
        if routing_weights.ndim != 2:
            raise ValueError('routing_weights must have shape [B, N].')
        if prototypes.ndim != 2:
            raise ValueError('prototypes must have shape [N, D].')
        if routing_weights.size(-1) != prototypes.size(0):
            raise ValueError('routing_weights prototype dimension must match prototypes.')
        summary = routing_weights @ prototypes
        if not return_debug:
            return summary
        return summary, {
            'prototype_summary': summary,
        }
