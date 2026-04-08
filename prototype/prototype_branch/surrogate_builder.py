"""Surrogate text builders with explicit row-wise semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrototypeSurrogateBuilderConfig:
    """Surrogate builder config."""

    teacher_temperature: float = 0.07

    def __post_init__(self) -> None:
        if self.teacher_temperature <= 0:
            raise ValueError("teacher_temperature must be > 0")


@dataclass(frozen=True)
class PairwiseSurrogateOutput:
    """Pairwise surrogate output.

    Shape:
        - pairwise_surrogate: [B_img, B_txt, D]
    """

    pairwise_surrogate: torch.Tensor

    def __post_init__(self) -> None:
        if self.pairwise_surrogate.ndim != 3:
            raise ValueError(
                "pairwise_surrogate must be [B_img, B_txt, D], got "
                f"{tuple(self.pairwise_surrogate.shape)}"
            )


@dataclass(frozen=True)
class DiagonalSurrogateOutput:
    """Diagonal surrogate output for matched `(i,i)` path.

    Shape:
        - diagonal_surrogate: [B, D]
    """

    diagonal_surrogate: torch.Tensor

    def __post_init__(self) -> None:
        if self.diagonal_surrogate.ndim != 2:
            raise ValueError(
                "diagonal_surrogate must be [B, D], got "
                f"{tuple(self.diagonal_surrogate.shape)}"
            )


@dataclass(frozen=True)
class ExactDiagonalTeacherOutput:
    """Exact diagonal helper output for future `L_diag` support.

    Shapes:
        - exact_text: [B, D]
        - token_weights: [B, L]
        - token_logits: [B, L]
    """

    exact_text: torch.Tensor
    token_weights: torch.Tensor
    token_logits: torch.Tensor

    def __post_init__(self) -> None:
        if self.exact_text.ndim != 2:
            raise ValueError(f"exact_text must be [B, D], got {tuple(self.exact_text.shape)}")
        if self.token_weights.ndim != 2:
            raise ValueError(
                f"token_weights must be [B, L], got {tuple(self.token_weights.shape)}"
            )
        if self.token_logits.ndim != 2:
            raise ValueError(
                f"token_logits must be [B, L], got {tuple(self.token_logits.shape)}"
            )


class PrototypeSurrogateBuilder:
    """Build surrogate text objects with row-wise `i -> j` semantics."""

    def __init__(self, config: PrototypeSurrogateBuilderConfig | None = None) -> None:
        self.config = config or PrototypeSurrogateBuilderConfig()

    def build_pairwise(
        self, alpha: torch.Tensor, basis_bank: torch.Tensor, chunk_size: int | None = None
    ) -> PairwiseSurrogateOutput:
        """Build pairwise surrogate text tensor `[B_img, B_txt, D]`."""
        if alpha.ndim != 2:
            raise ValueError(f"alpha must be [B_img, N], got {tuple(alpha.shape)}")
        if basis_bank.ndim != 3:
            raise ValueError(f"basis_bank must be [B_txt, N, D], got {tuple(basis_bank.shape)}")
        if alpha.shape[1] != basis_bank.shape[1]:
            raise ValueError(
                f"Prototype count mismatch between alpha and basis_bank: {alpha.shape[1]} vs {basis_bank.shape[1]}"
            )

        if chunk_size is None:
            pairwise = torch.einsum("in,jnd->ijd", alpha, basis_bank)
            return PairwiseSurrogateOutput(pairwise_surrogate=pairwise)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0 when provided")
        parts = []
        for start in range(0, basis_bank.shape[0], chunk_size):
            end = min(start + chunk_size, basis_bank.shape[0])
            chunk = basis_bank[start:end]  # [Jc, N, D]
            parts.append(torch.einsum("in,jnd->ijd", alpha, chunk))
        pairwise = torch.cat(parts, dim=1)
        return PairwiseSurrogateOutput(pairwise_surrogate=pairwise)

    def build_diagonal(self, alpha: torch.Tensor, basis_bank: torch.Tensor) -> DiagonalSurrogateOutput:
        """Build diagonal surrogate `[B, D]` from matched `(i,i)` pairs."""
        if alpha.ndim != 2:
            raise ValueError(f"alpha must be [B, N], got {tuple(alpha.shape)}")
        if basis_bank.ndim != 3:
            raise ValueError(f"basis_bank must be [B, N, D], got {tuple(basis_bank.shape)}")
        if alpha.shape[0] != basis_bank.shape[0]:
            raise ValueError("Diagonal surrogate requires matching batch dimension in alpha and basis_bank")
        if alpha.shape[1] != basis_bank.shape[1]:
            raise ValueError("Prototype count mismatch between alpha and basis_bank")
        diagonal = torch.einsum("bn,bnd->bd", alpha, basis_bank)
        return DiagonalSurrogateOutput(diagonal_surrogate=diagonal)

    def build_exact_diagonal_teacher(
        self,
        q_summary: torch.Tensor,
        h_j_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> ExactDiagonalTeacherOutput:
        """Build exact diagonal helper text from matched `(i,i)` only.

        This helper is diagonal-only and does not access off-diagonal `(i,j)` pairs.
        """
        if q_summary.ndim != 2:
            raise ValueError(f"q_summary must be [B, D], got {tuple(q_summary.shape)}")
        if h_j_tokens.ndim != 3:
            raise ValueError(f"h_j_tokens must be [B, L, D], got {tuple(h_j_tokens.shape)}")
        if q_summary.shape[0] != h_j_tokens.shape[0]:
            raise ValueError("q_summary and h_j_tokens batch mismatch")
        if q_summary.shape[1] != h_j_tokens.shape[2]:
            raise ValueError("Feature dim mismatch between q_summary and h_j_tokens")

        token_logits = (
            torch.einsum("bd,bld->bl", q_summary, h_j_tokens)
            / self.config.teacher_temperature
        )
        if token_mask is not None:
            if token_mask.shape != h_j_tokens.shape[:2]:
                raise ValueError("token_mask must match [B, L] from h_j_tokens")
            valid = token_mask.to(dtype=torch.bool)
            if not torch.all(valid.any(dim=-1)):
                raise ValueError("Each sample must have at least one valid token for exact diagonal helper")
            token_logits = token_logits.masked_fill(~valid, float("-inf"))

        token_weights = torch.softmax(token_logits, dim=-1)
        exact_text = torch.einsum("bl,bld->bd", token_weights, h_j_tokens)
        return ExactDiagonalTeacherOutput(
            exact_text=exact_text, token_weights=token_weights, token_logits=token_logits
        )

    def build_surrogate(
        self, alpha: torch.Tensor, basis_bank: torch.Tensor, chunk_size: int | None = None
    ) -> PairwiseSurrogateOutput:
        """Compatibility alias returning pairwise surrogate output."""
        return self.build_pairwise(alpha, basis_bank, chunk_size=chunk_size)
