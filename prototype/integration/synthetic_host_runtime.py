"""Project-side synthetic host runtime for canonical launcher smoke execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from prototype.config.schema import TrainMode
from prototype.integration.feature_surface import (
    CLIPHostFeatureSurface,
    CLIPHostScoreSurface,
    HostFeatureSurface,
    HostScoreSurface,
    ITSELFHostFeatureSurface,
    ITSELFHostScoreSurface,
)


def _score_matrix(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs_norm = F.normalize(lhs, p=2, dim=1)
    rhs_norm = F.normalize(rhs, p=2, dim=1)
    return lhs_norm @ rhs_norm.t()


@dataclass(frozen=True)
class SyntheticHostRuntimeConfig:
    """Synthetic host runtime config for train launcher."""

    train_mode: TrainMode
    lambda_s: float | None
    feature_dim: int = 8

    def __post_init__(self) -> None:
        if self.train_mode == "itself" and self.lambda_s is None:
            raise ValueError("Synthetic ITSELF runtime requires lambda_s")
        if self.train_mode == "clip" and self.lambda_s is not None:
            raise ValueError("Synthetic CLIP runtime must use lambda_s=None")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")


class SyntheticHostRuntime:
    """Deterministic mode-aware host runtime surface for launcher recipes."""

    def __init__(self, config: SyntheticHostRuntimeConfig) -> None:
        self.config = config
        self.module = nn.Linear(config.feature_dim, config.feature_dim, bias=False)
        nn.init.eye_(self.module.weight)

    def extract_feature_surface(self, batch: Mapping[str, Any]) -> HostFeatureSurface:
        if "images" not in batch or "caption_ids" not in batch:
            raise KeyError("Batch must contain 'images' and 'caption_ids'")
        images = batch["images"]
        caption_ids = batch["caption_ids"]
        if not isinstance(images, torch.Tensor) or not isinstance(caption_ids, torch.Tensor):
            raise TypeError("Batch values must be torch.Tensor")
        bsz = int(images.shape[0])
        seq_len = int(caption_ids.shape[1])
        dim = self.config.feature_dim

        v_i_global = torch.linspace(
            0.1, 0.9, steps=bsz * dim, dtype=torch.float32
        ).reshape(bsz, dim)
        h_j_tokens = torch.linspace(
            0.2, 1.0, steps=bsz * seq_len * dim, dtype=torch.float32
        ).reshape(bsz, seq_len, dim)
        eot = caption_ids.argmax(dim=-1)
        t_j_global = h_j_tokens[torch.arange(bsz), eot]
        z_i_retrieval = v_i_global

        if self.config.train_mode == "itself":
            v_i_local = v_i_global + 0.3
            t_j_local = t_j_global + 0.4
            return ITSELFHostFeatureSurface(
                v_i_global=v_i_global,
                h_j_tokens=h_j_tokens,
                t_j_global=t_j_global,
                z_i_retrieval=z_i_retrieval,
                v_i_local=v_i_local,
                t_j_local=t_j_local,
                v_i_global_source="synthetic.itself.global",
                h_j_tokens_source="synthetic.itself.tokens",
                t_j_global_source="synthetic.itself.eot",
                z_i_retrieval_source="synthetic.itself.retrieval",
                v_i_local_source="synthetic.itself.local_image",
                t_j_local_source="synthetic.itself.local_text",
            )

        return CLIPHostFeatureSurface(
            v_i_global=v_i_global,
            h_j_tokens=h_j_tokens,
            t_j_global=t_j_global,
            z_i_retrieval=z_i_retrieval,
            v_i_global_source="synthetic.clip.global",
            h_j_tokens_source="synthetic.clip.tokens",
            t_j_global_source="synthetic.clip.eot",
            z_i_retrieval_source="synthetic.clip.retrieval",
        )

    def compute_host_score_surface(self, features: HostFeatureSurface) -> HostScoreSurface:
        if self.config.train_mode == "itself":
            if not isinstance(features, ITSELFHostFeatureSurface):
                raise TypeError("Expected ITSELFHostFeatureSurface for synthetic ITSELF mode")
            s_global = _score_matrix(features.v_i_global, features.t_j_global)
            s_local = _score_matrix(features.v_i_local, features.t_j_local)
            s_host = float(self.config.lambda_s) * s_global + (1.0 - float(self.config.lambda_s)) * s_local
            return ITSELFHostScoreSurface(
                s_host_itself=s_host,
                s_global_itself=s_global,
                s_local_itself=s_local,
                lambda_s=float(self.config.lambda_s),
            )

        if not isinstance(features, CLIPHostFeatureSurface):
            raise TypeError("Expected CLIPHostFeatureSurface for synthetic CLIP mode")
        return CLIPHostScoreSurface(
            s_host_clip=_score_matrix(features.v_i_global, features.t_j_global)
        )

    def compute_host_loss_itself(
        self,
        batch: Mapping[str, Any],
        features: HostFeatureSurface,
        host_scores: HostScoreSurface,
    ) -> torch.Tensor:
        if self.config.train_mode != "itself":
            raise RuntimeError("compute_host_loss_itself called in non-itself mode")
        return (self.module.weight ** 2).mean()

    def compute_host_loss_clip(
        self,
        batch: Mapping[str, Any],
        features: HostFeatureSurface,
        host_scores: HostScoreSurface,
    ) -> torch.Tensor:
        if self.config.train_mode != "clip":
            raise RuntimeError("compute_host_loss_clip called in non-clip mode")
        return ((self.module.weight - 0.25) ** 2).mean()


def build_synthetic_batch(
    batch_size: int,
    seq_len: int,
    image_shape: tuple[int, int, int] = (3, 32, 16),
) -> dict[str, torch.Tensor]:
    """Create deterministic batch payload for launcher smoke execution."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    images = torch.randn(batch_size, *image_shape)
    caption_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for idx in range(batch_size):
        caption_ids[idx, idx % seq_len] = 1
    token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return {"images": images, "caption_ids": caption_ids, "token_mask": token_mask}
