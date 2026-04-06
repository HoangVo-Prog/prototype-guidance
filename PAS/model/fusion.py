from typing import Optional

import torch
import torch.nn as nn


class ResidualScoreFusion(nn.Module):
    def __init__(
        self,
        enabled: bool = True,
        coefficient: float = 1.0,
        coefficient_source: str = 'fixed',
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.coefficient = float(coefficient)
        self.coefficient_source = str(coefficient_source).lower()
        if self.coefficient_source not in {'fixed', 'validation_tuned'}:
            raise ValueError("coefficient_source must be one of ['fixed', 'validation_tuned'].")

    def forward(
        self,
        host_similarity: torch.Tensor,
        prototype_similarity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prototype_similarity is None or not self.enabled:
            return host_similarity
        if host_similarity.shape != prototype_similarity.shape:
            raise ValueError('Host and prototype similarities must have the same shape for score fusion.')
        return host_similarity + (self.coefficient * prototype_similarity)
