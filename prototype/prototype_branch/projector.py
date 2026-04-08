"""Prototype-side text projector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PrototypeProjectorConfig:
    """Projector config.

    Attributes:
        input_dim: Input feature dim `D`.
        output_dim: Output feature dim `D_r`.
        bias: Use bias term in projection.
        use_layer_norm: Apply layer norm on projected outputs.
        normalize_output: L2-normalize projected outputs.
    """

    input_dim: int
    output_dim: int
    bias: bool = True
    use_layer_norm: bool = False
    normalize_output: bool = True

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be > 0")


class PrototypeProjector(nn.Module):
    """Project surrogate text from host-space `D` to prototype-space `D_r`."""

    def __init__(self, config: PrototypeProjectorConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.input_dim, config.output_dim, bias=config.bias)
        self.layer_norm = (
            nn.LayerNorm(config.output_dim) if config.use_layer_norm else nn.Identity()
        )

    def project(self, surrogate_text: torch.Tensor) -> torch.Tensor:
        """Return projected surrogate embeddings with trailing dim `D_r`."""
        if surrogate_text.ndim < 2:
            raise ValueError(
                "surrogate_text must have at least 2 dims and trailing feature dim"
            )
        if surrogate_text.shape[-1] != self.config.input_dim:
            raise ValueError(
                "surrogate_text feature dim mismatch: expected "
                f"{self.config.input_dim}, got {surrogate_text.shape[-1]}"
            )
        projected = self.proj(surrogate_text)
        projected = self.layer_norm(projected)
        if self.config.normalize_output:
            projected = F.normalize(projected, p=2, dim=-1)
        return projected

    def forward(self, surrogate_text: torch.Tensor) -> torch.Tensor:
        """Alias for projection."""
        return self.project(surrogate_text)
