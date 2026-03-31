import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


CONTEXTUALIZATION_ALIASES = {
    'none': 'none',
    'off': 'none',
    'self_attention': 'self_attention',
    'similarity': 'self_attention',
    'similarity_residual': 'self_attention',
    'transformer': 'self_attention',
}


class PrototypeContextualizer(nn.Module):
    def __init__(
        self,
        enabled: bool = False,
        contextualization_type: str = 'none',
        residual: bool = True,
        normalize: bool = True,
        temperature: Optional[float] = None,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.contextualization_type = CONTEXTUALIZATION_ALIASES.get(str(contextualization_type).lower())
        if self.contextualization_type is None:
            raise ValueError(f'Unsupported contextualization type: {contextualization_type}')
        self.residual = bool(residual)
        self.normalize = bool(normalize)
        self.temperature = None if temperature is None else float(temperature)

    def _compute_similarity_logits(self, prototypes: torch.Tensor) -> torch.Tensor:
        features = F.normalize(prototypes, dim=-1) if self.normalize else prototypes
        logits = features @ features.t()
        logits = logits / math.sqrt(prototypes.size(-1))
        if self.temperature is not None and self.temperature > 0:
            logits = logits / self.temperature
        return logits

    def forward(self, prototypes: torch.Tensor, return_debug: bool = False):
        if prototypes.ndim != 2:
            raise ValueError('prototypes must have shape [N, D].')
        if not self.enabled or self.contextualization_type == 'none':
            if not return_debug:
                return prototypes
            identity = torch.eye(prototypes.size(0), device=prototypes.device, dtype=prototypes.dtype)
            return prototypes, {
                'contextualized_prototypes': prototypes,
                'prototype_similarity': identity,
                'contextualization_weights': identity,
                'contextualization_residual': int(self.residual),
                'contextualization_num_layers': 1,
            }

        logits = self._compute_similarity_logits(prototypes)
        weights = torch.softmax(logits, dim=-1)
        updated = weights @ prototypes
        contextualized = prototypes + updated if self.residual else updated

        if not return_debug:
            return contextualized
        return contextualized, {
            'contextualized_prototypes': contextualized,
            'prototype_similarity': logits,
            'contextualization_weights': weights,
            'contextualization_residual': int(self.residual),
            'contextualization_num_layers': 1,
            'prototype_contextualization_entropy': (-(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean()).detach(),
        }
