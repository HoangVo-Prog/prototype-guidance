"""Optional contextualization for prototype bank."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PrototypeContextualizerConfig:
    """Contextualizer config.

    Attributes:
        enabled: If False, contextualization is identity.
        temperature: Attention temperature for prototype-prototype affinity.
    """

    enabled: bool = False
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")


class PrototypeContextualizer(nn.Module):
    """Contextualize prototypes with self-attention semantics from spec."""

    def __init__(self, config: PrototypeContextualizerConfig | None = None) -> None:
        super().__init__()
        self.config = config or PrototypeContextualizerConfig()

    def contextualize(self, prototypes: torch.Tensor) -> torch.Tensor:
        """Return contextualized prototypes with shape `[N, D]`."""
        if prototypes.ndim != 2:
            raise ValueError(
                f"prototypes must be rank-2 [N, D], got {tuple(prototypes.shape)}"
            )
        if not self.config.enabled:
            return prototypes

        dim = prototypes.shape[-1]
        affinity = (prototypes @ prototypes.t()) / (
            math.sqrt(dim) * self.config.temperature
        )
        attn = torch.softmax(affinity, dim=-1)
        contextualized = attn @ prototypes
        return contextualized

    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        """Alias for contextualization."""
        return self.contextualize(prototypes)
