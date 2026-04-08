"""Prototype score computation with explicit row-wise orientation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PrototypeScorerConfig:
    """Scorer config."""

    temperature: float = 0.07
    normalize_inputs: bool = True

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")


class PrototypeScorer:
    """Compute prototype score matrix `[B_img, B_txt]`."""

    def __init__(self, config: PrototypeScorerConfig | None = None) -> None:
        self.config = config or PrototypeScorerConfig()

    def score_pairwise(
        self, z_i_retrieval: torch.Tensor, z_hat_pairwise_text: torch.Tensor
    ) -> torch.Tensor:
        """Score image features against pairwise surrogate text embeddings.

        Inputs:
            - z_i_retrieval: [B_img, D_r]
            - z_hat_pairwise_text: [B_img, B_txt, D_r]
        Output:
            - scores: [B_img, B_txt]
        """
        if z_i_retrieval.ndim != 2:
            raise ValueError(
                f"z_i_retrieval must be [B_img, D_r], got {tuple(z_i_retrieval.shape)}"
            )
        if z_hat_pairwise_text.ndim != 3:
            raise ValueError(
                "z_hat_pairwise_text must be [B_img, B_txt, D_r], got "
                f"{tuple(z_hat_pairwise_text.shape)}"
            )
        if z_i_retrieval.shape[0] != z_hat_pairwise_text.shape[0]:
            raise ValueError("Batch mismatch: image rows must match pairwise row axis")
        if z_i_retrieval.shape[1] != z_hat_pairwise_text.shape[2]:
            raise ValueError("Feature dim mismatch between image and surrogate text")

        images = z_i_retrieval
        pairwise_text = z_hat_pairwise_text
        if self.config.normalize_inputs:
            images = F.normalize(images, p=2, dim=-1)
            pairwise_text = F.normalize(pairwise_text, p=2, dim=-1)

        scores = torch.einsum("id,ijd->ij", images, pairwise_text) / self.config.temperature
        return scores

    def score(self, z_i_retrieval: torch.Tensor, z_hat_text: torch.Tensor) -> torch.Tensor:
        """Return pairwise prototype score matrix `[B_img, B_txt]`.

        Accepts either:
            - pairwise surrogate embeddings `[B_img, B_txt, D_r]`, or
            - text embeddings `[B_txt, D_r]` for non-pairwise fallback.
        """
        if z_hat_text.ndim == 3:
            return self.score_pairwise(z_i_retrieval, z_hat_text)
        if z_hat_text.ndim != 2:
            raise ValueError(
                "z_hat_text must be [B_txt, D_r] or [B_img, B_txt, D_r], got "
                f"{tuple(z_hat_text.shape)}"
            )
        if z_i_retrieval.ndim != 2:
            raise ValueError(
                f"z_i_retrieval must be [B_img, D_r], got {tuple(z_i_retrieval.shape)}"
            )
        if z_i_retrieval.shape[1] != z_hat_text.shape[1]:
            raise ValueError("Feature dim mismatch between image and text embeddings")

        images = z_i_retrieval
        texts = z_hat_text
        if self.config.normalize_inputs:
            images = F.normalize(images, p=2, dim=-1)
            texts = F.normalize(texts, p=2, dim=-1)
        return (images @ texts.t()) / self.config.temperature

    def score_diagonal(
        self, z_i_retrieval: torch.Tensor, z_hat_diag_text: torch.Tensor
    ) -> torch.Tensor:
        """Return diagonal scores for matched `(i,i)` path.

        Inputs:
            - z_i_retrieval: [B, D_r]
            - z_hat_diag_text: [B, D_r]
        Output:
            - diagonal scores: [B]
        """
        if z_i_retrieval.ndim != 2 or z_hat_diag_text.ndim != 2:
            raise ValueError("Diagonal score inputs must both be rank-2 [B, D_r]")
        if z_i_retrieval.shape != z_hat_diag_text.shape:
            raise ValueError("Diagonal score inputs must have identical shape")

        images = z_i_retrieval
        texts = z_hat_diag_text
        if self.config.normalize_inputs:
            images = F.normalize(images, p=2, dim=-1)
            texts = F.normalize(texts, p=2, dim=-1)
        return torch.sum(images * texts, dim=-1) / self.config.temperature
