"""Prototype-conditioned basis builder from token-level text states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PrototypeBasisBuilderConfig:
    """Basis builder config."""

    temperature: float = 0.07

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")


@dataclass(frozen=True)
class BasisOutput:
    """Prototype-conditioned basis output.

    Shapes:
        - basis_bank: [B, N, D]
        - token_weights: [B, N, L]
        - token_logits: [B, N, L]
    """

    basis_bank: torch.Tensor
    token_weights: torch.Tensor
    token_logits: torch.Tensor

    def __post_init__(self) -> None:
        if self.basis_bank.ndim != 3:
            raise ValueError(
                f"basis_bank must be [B, N, D], got {tuple(self.basis_bank.shape)}"
            )
        if self.token_weights.ndim != 3:
            raise ValueError(
                "token_weights must be [B, N, L], got "
                f"{tuple(self.token_weights.shape)}"
            )
        if self.token_logits.ndim != 3:
            raise ValueError(
                "token_logits must be [B, N, L], got "
                f"{tuple(self.token_logits.shape)}"
            )


class PrototypeBasisBuilder(nn.Module):
    """Build caption basis bank from token-level host text states.

    Inputs:
        - h_j_tokens: [B, L, D]
        - contextualized_prototypes: [N, D]
        - token_mask (optional): [B, L], True for valid token
    """

    def __init__(self, config: PrototypeBasisBuilderConfig | None = None) -> None:
        super().__init__()
        self.config = config or PrototypeBasisBuilderConfig()

    def build_basis(
        self,
        h_j_tokens: torch.Tensor,
        contextualized_prototypes: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> BasisOutput:
        """Return basis bank `B_j` with shape `[B, N, D]`.

        Pooled text features `[B, D]` are forbidden by contract.
        """
        if h_j_tokens.ndim != 3:
            raise ValueError(
                "h_j_tokens must be token-level [B, L, D]; pooled text [B, D] is forbidden"
            )
        if contextualized_prototypes.ndim != 2:
            raise ValueError(
                "contextualized_prototypes must be [N, D], got "
                f"{tuple(contextualized_prototypes.shape)}"
            )
        if h_j_tokens.shape[-1] != contextualized_prototypes.shape[-1]:
            raise ValueError(
                "Feature dim mismatch between h_j_tokens and contextualized_prototypes: "
                f"{h_j_tokens.shape[-1]} vs {contextualized_prototypes.shape[-1]}"
            )

        token_logits = (
            torch.einsum("nd,bld->bnl", contextualized_prototypes, h_j_tokens)
            / self.config.temperature
        )

        if token_mask is not None:
            if token_mask.ndim != 2:
                raise ValueError(
                    f"token_mask must be [B, L], got {tuple(token_mask.shape)}"
                )
            if token_mask.shape != h_j_tokens.shape[:2]:
                raise ValueError(
                    "token_mask shape must match [B, L] from h_j_tokens, got "
                    f"{tuple(token_mask.shape)} vs {tuple(h_j_tokens.shape[:2])}"
                )
            valid = token_mask.to(dtype=torch.bool).unsqueeze(1)  # [B, 1, L]
            if not torch.all(valid.any(dim=-1)):
                raise ValueError("Each sample must have at least one valid token")
            token_logits = token_logits.masked_fill(~valid, float("-inf"))

        token_weights = torch.softmax(token_logits, dim=-1)
        basis_bank = torch.einsum("bnl,bld->bnd", token_weights, h_j_tokens)
        return BasisOutput(
            basis_bank=basis_bank, token_weights=token_weights, token_logits=token_logits
        )

    def forward(
        self,
        h_j_tokens: torch.Tensor,
        contextualized_prototypes: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> BasisOutput:
        """Alias for basis building."""
        return self.build_basis(h_j_tokens, contextualized_prototypes, token_mask)
