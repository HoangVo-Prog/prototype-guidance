from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


ROUTING_TYPE_ALIASES = {
    'cosine': 'cosine',
    'normalized_dot': 'cosine',
    'softmax': 'cosine',
    'dot': 'dot',
}


class Router(nn.Module):
    def __init__(
        self,
        routing_type: str = 'cosine',
        temperature: float = 0.07,
        normalize: bool = True,
    ):
        super().__init__()
        self.routing_type = ROUTING_TYPE_ALIASES.get(str(routing_type).lower())
        if self.routing_type is None:
            raise ValueError(f'Unsupported routing type: {routing_type}')
        if temperature <= 0:
            raise ValueError('temperature must be positive.')
        self.temperature = float(temperature)
        self.normalize = bool(normalize)

    def _compute_similarity(self, image_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        if self.routing_type == 'cosine' and self.normalize:
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            prototypes = F.normalize(prototypes, dim=-1)
        return image_embeddings @ prototypes.t()

    def forward(self, image_embeddings: torch.Tensor, prototypes: torch.Tensor, return_debug: bool = False):
        if image_embeddings.ndim != 2:
            raise ValueError('image_embeddings must have shape [B, D].')
        if prototypes.ndim != 2:
            raise ValueError('prototypes must have shape [N, D].')
        if image_embeddings.size(-1) != prototypes.size(-1):
            raise ValueError('image_embeddings and prototypes must share the same feature dimension.')

        similarity = self._compute_similarity(image_embeddings, prototypes)
        alpha_logits = similarity / self.temperature
        stable_logits = alpha_logits - alpha_logits.max(dim=-1, keepdim=True).values
        alpha = torch.softmax(stable_logits, dim=-1)
        if not torch.isfinite(alpha).all():
            raise FloatingPointError('Router produced non-finite routing weights.')

        if not return_debug:
            return alpha
        routing_entropy = -(alpha * alpha.clamp_min(1e-12).log()).sum(dim=-1)
        return alpha, {
            'routing_similarity': similarity,
            'alpha_logits': alpha_logits,
            'routing_logits': stable_logits,
            'routing_weights': alpha,
            'routing_max_prob': alpha.max(dim=-1).values.mean().detach(),
            'prototype_assignment_entropy': routing_entropy.mean().detach(),
            'routing_effective_support': routing_entropy.exp().mean().detach(),
        }

