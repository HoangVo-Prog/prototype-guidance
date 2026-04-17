from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


ROUTING_TYPE_ALIASES = {
    'cosine': 'cosine',
    'normalized_dot': 'cosine',
    'softmax': 'cosine',
    'dot': 'dot',
}

LOCAL_ROUTING_POOLING_ALIASES = {
    'logsumexp': 'logsumexp',
    'lse': 'logsumexp',
    'max': 'max',
    'mean': 'mean',
    'avg': 'mean',
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

    def _compute_local_similarity(
        self,
        local_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        normalize_inputs: bool = True,
    ) -> torch.Tensor:
        if self.routing_type == 'cosine' and normalize_inputs:
            local_embeddings = F.normalize(local_embeddings, dim=-1)
            prototypes = F.normalize(prototypes, dim=-1)
        return torch.einsum('bmd,nd->bmn', local_embeddings, prototypes)

    def route_from_local_evidence(
        self,
        local_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        *,
        pooling: str = 'logsumexp',
        temperature: Optional[float] = None,
        normalize_inputs: bool = True,
        return_debug: bool = False,
    ):
        if local_embeddings.ndim != 3:
            raise ValueError('local_embeddings must have shape [B, M, D].')
        if prototypes.ndim != 2:
            raise ValueError('prototypes must have shape [N, D].')
        if local_embeddings.size(-1) != prototypes.size(-1):
            raise ValueError('local_embeddings and prototypes must share the same feature dimension.')
        if local_embeddings.size(1) <= 0:
            raise ValueError('local_embeddings must contain at least one local token per image.')

        pooling_mode = LOCAL_ROUTING_POOLING_ALIASES.get(str(pooling).lower())
        if pooling_mode is None:
            raise ValueError(f'Unsupported local routing pooling: {pooling}')

        effective_temperature = self.temperature if temperature is None else float(temperature)
        if effective_temperature <= 0:
            raise ValueError('local routing temperature must be positive.')

        similarity = self._compute_local_similarity(
            local_embeddings=local_embeddings,
            prototypes=prototypes,
            normalize_inputs=bool(normalize_inputs),
        )
        token_logits = similarity / effective_temperature
        if pooling_mode == 'logsumexp':
            alpha_logits = torch.logsumexp(token_logits, dim=1)
        elif pooling_mode == 'max':
            alpha_logits = token_logits.max(dim=1).values
        else:
            alpha_logits = token_logits.mean(dim=1)

        stable_logits = alpha_logits - alpha_logits.max(dim=-1, keepdim=True).values
        alpha = torch.softmax(stable_logits, dim=-1)
        if not torch.isfinite(alpha).all():
            raise FloatingPointError('Local-evidence router produced non-finite routing weights.')

        if not return_debug:
            return alpha

        routing_entropy = -(alpha * alpha.clamp_min(1e-12).log()).sum(dim=-1)
        return alpha, {
            'routing_similarity': alpha_logits,
            'alpha_logits': alpha_logits,
            'routing_logits': stable_logits,
            'routing_weights': alpha,
            'routing_max_prob': alpha.max(dim=-1).values.mean().detach(),
            'prototype_assignment_entropy': routing_entropy.mean().detach(),
            'routing_effective_support': routing_entropy.exp().mean().detach(),
            'local_routing_entropy': routing_entropy.mean().detach(),
            'local_routing_max_mean': alpha.max(dim=-1).values.mean().detach(),
            'local_routing_effective_support': routing_entropy.exp().mean().detach(),
        }

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
