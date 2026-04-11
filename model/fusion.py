from typing import Optional, Tuple

import torch
import torch.nn as nn


class ResidualScoreFusion(nn.Module):
    def __init__(
        self,
        enabled: bool = True,
        lambda_host: float = 1.0,
        lambda_prototype: float = 0.0,
        coefficient_source: str = 'fixed',
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.lambda_host = float(lambda_host)
        self.lambda_prototype = float(lambda_prototype)
        self.coefficient_source = str(coefficient_source).lower()
        if self.coefficient_source not in {'fixed', 'validation_tuned'}:
            raise ValueError("coefficient_source must be one of ['fixed', 'validation_tuned'].")

    def resolve_weights(
        self,
        lambda_host: Optional[float] = None,
        lambda_prototype: Optional[float] = None,
    ) -> Tuple[float, float]:
        host_weight = self.lambda_host if lambda_host is None else float(lambda_host)
        prototype_weight = self.lambda_prototype if lambda_prototype is None else float(lambda_prototype)
        return host_weight, prototype_weight

    def forward(
        self,
        host_similarity: Optional[torch.Tensor],
        prototype_similarity: Optional[torch.Tensor] = None,
        lambda_host: Optional[float] = None,
        lambda_prototype: Optional[float] = None,
    ) -> torch.Tensor:
        if not isinstance(host_similarity, torch.Tensor):
            raise ValueError('host_similarity must be a tensor for score fusion.')
        host_weight, prototype_weight = self.resolve_weights(
            lambda_host=lambda_host,
            lambda_prototype=lambda_prototype,
        )

        if prototype_similarity is None or not self.enabled:
            if prototype_similarity is None and self.enabled and abs(prototype_weight) > 1e-12:
                raise RuntimeError(
                    'prototype_similarity is missing while lambda_prototype is non-zero. '
                    'Set fusion.lambda_prototype=0.0 for host-only evaluation or enable prototype scoring.'
                )
            return host_weight * host_similarity

        if host_similarity.shape != prototype_similarity.shape:
            raise ValueError('Host and prototype similarities must have the same shape for score fusion.')
        return (host_weight * host_similarity) + (prototype_weight * prototype_similarity)
