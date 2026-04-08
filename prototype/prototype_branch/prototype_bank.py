"""Learnable prototype bank for the additive prototype branch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

InitMode = Literal["normal", "xavier_uniform", "uniform_unit_norm"]


@dataclass(frozen=True)
class PrototypeBankConfig:
    """Prototype bank config.

    Attributes:
        num_prototypes: Number of prototypes `N`.
        feature_dim: Prototype feature dimension `D`.
        init_mode: Initialization strategy for the learnable bank.
        init_std: Standard deviation for `normal` initialization.
    """

    num_prototypes: int
    feature_dim: int
    init_mode: InitMode = "normal"
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.num_prototypes <= 0:
            raise ValueError("num_prototypes must be > 0")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")
        if self.init_std <= 0:
            raise ValueError("init_std must be > 0")


@dataclass(frozen=True)
class PrototypeBankOutput:
    """Prototype bank output.

    Shape expectations:
        - prototypes: [N, D]
        - contextualized_prototypes: [N, D]
    """

    prototypes: torch.Tensor
    contextualized_prototypes: torch.Tensor

    def __post_init__(self) -> None:
        if self.prototypes.ndim != 2:
            raise ValueError(
                f"prototypes must be [N, D], got {tuple(self.prototypes.shape)}"
            )
        if self.contextualized_prototypes.ndim != 2:
            raise ValueError(
                "contextualized_prototypes must be [N, D], got "
                f"{tuple(self.contextualized_prototypes.shape)}"
            )
        if self.prototypes.shape != self.contextualized_prototypes.shape:
            raise ValueError(
                "prototypes and contextualized_prototypes must have identical shape"
            )


class PrototypeBankStore(nn.Module):
    """Owns learnable prototype bank `P` with shape `[N, D]`."""

    def __init__(self, config: PrototypeBankConfig) -> None:
        super().__init__()
        self.config = config
        self.prototypes = nn.Parameter(
            torch.empty(config.num_prototypes, config.feature_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the prototype bank using config-defined strategy."""
        if self.config.init_mode == "normal":
            nn.init.normal_(self.prototypes, mean=0.0, std=self.config.init_std)
            return
        if self.config.init_mode == "xavier_uniform":
            nn.init.xavier_uniform_(self.prototypes)
            return
        if self.config.init_mode == "uniform_unit_norm":
            bound = 1.0 / math.sqrt(self.config.feature_dim)
            nn.init.uniform_(self.prototypes, a=-bound, b=bound)
            with torch.no_grad():
                self.prototypes.copy_(
                    torch.nn.functional.normalize(self.prototypes, p=2, dim=-1)
                )
            return
        raise ValueError(f"Unsupported init_mode={self.config.init_mode!r}")

    def forward(self) -> torch.Tensor:
        """Return raw learnable prototype bank `[N, D]`."""
        return self.prototypes

    def get_bank(self, contextualized_prototypes: torch.Tensor | None = None) -> PrototypeBankOutput:
        """Return prototype bank output for downstream modules."""
        raw = self.forward()
        ctx = raw if contextualized_prototypes is None else contextualized_prototypes
        return PrototypeBankOutput(prototypes=raw, contextualized_prototypes=ctx)
