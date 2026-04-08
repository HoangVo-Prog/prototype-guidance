"""Shared dummy runtime utilities for integration verification tests."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from prototype.config.defaults import DEFAULT_CONFIG
from prototype.config.schema import IntegrationConfig, TrainMode, load_integration_config
from prototype.integration.feature_surface import (
    CLIPHostFeatureSurface,
    CLIPHostScoreSurface,
    HostFeatureSurface,
    HostScoreSurface,
    ITSELFHostFeatureSurface,
    ITSELFHostScoreSurface,
)


def score_matrix(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs_norm = F.normalize(lhs, p=2, dim=1)
    rhs_norm = F.normalize(rhs, p=2, dim=1)
    return lhs_norm @ rhs_norm.t()


class DummyHostRuntime:
    """Deterministic host runtime stub for integration/runtime verification."""

    def __init__(self, train_mode: TrainMode, lambda_s: float | None) -> None:
        self.train_mode = train_mode
        self.lambda_s = lambda_s
        self.module = nn.Linear(8, 8, bias=False)
        nn.init.eye_(self.module.weight)
        self.calls_host_loss_itself = 0
        self.calls_host_loss_clip = 0

    def extract_feature_surface(self, batch: Mapping[str, Any]) -> HostFeatureSurface:
        bsz = int(batch["images"].shape[0])
        seq_len = int(batch["caption_ids"].shape[1])
        dim = 8

        v_i_global = torch.linspace(
            0.1, 0.9, steps=bsz * dim, dtype=torch.float32
        ).reshape(bsz, dim)
        h_j_tokens = torch.linspace(
            0.2, 1.0, steps=bsz * seq_len * dim, dtype=torch.float32
        ).reshape(bsz, seq_len, dim)
        eot = batch["caption_ids"].argmax(dim=-1)
        t_j_global = h_j_tokens[torch.arange(bsz), eot]
        z_i_retrieval = v_i_global

        if self.train_mode == "itself":
            v_i_local = v_i_global + 0.3
            t_j_local = t_j_global + 0.4
            return ITSELFHostFeatureSurface(
                v_i_global=v_i_global,
                h_j_tokens=h_j_tokens,
                t_j_global=t_j_global,
                z_i_retrieval=z_i_retrieval,
                v_i_local=v_i_local,
                t_j_local=t_j_local,
                v_i_global_source="dummy.itself.global",
                h_j_tokens_source="dummy.itself.tokens",
                t_j_global_source="dummy.itself.eot",
                z_i_retrieval_source="dummy.itself.retrieval",
                v_i_local_source="dummy.itself.local_image",
                t_j_local_source="dummy.itself.local_text",
            )

        return CLIPHostFeatureSurface(
            v_i_global=v_i_global,
            h_j_tokens=h_j_tokens,
            t_j_global=t_j_global,
            z_i_retrieval=z_i_retrieval,
            v_i_global_source="dummy.clip.global",
            h_j_tokens_source="dummy.clip.tokens",
            t_j_global_source="dummy.clip.eot",
            z_i_retrieval_source="dummy.clip.retrieval",
        )

    def compute_host_score_surface(self, features: HostFeatureSurface) -> HostScoreSurface:
        if self.train_mode == "itself":
            assert isinstance(features, ITSELFHostFeatureSurface)
            s_global = score_matrix(features.v_i_global, features.t_j_global)
            s_local = score_matrix(features.v_i_local, features.t_j_local)
            s_host = float(self.lambda_s) * s_global + (1.0 - float(self.lambda_s)) * s_local
            return ITSELFHostScoreSurface(
                s_host_itself=s_host,
                s_global_itself=s_global,
                s_local_itself=s_local,
                lambda_s=float(self.lambda_s),
            )
        assert isinstance(features, CLIPHostFeatureSurface)
        return CLIPHostScoreSurface(
            s_host_clip=score_matrix(features.v_i_global, features.t_j_global)
        )

    def compute_host_loss_itself(
        self,
        batch: Mapping[str, Any],
        features: HostFeatureSurface,
        host_scores: HostScoreSurface,
    ) -> torch.Tensor:
        assert self.train_mode == "itself"
        self.calls_host_loss_itself += 1
        return (self.module.weight ** 2).mean()

    def compute_host_loss_clip(
        self,
        batch: Mapping[str, Any],
        features: HostFeatureSurface,
        host_scores: HostScoreSurface,
    ) -> torch.Tensor:
        assert self.train_mode == "clip"
        self.calls_host_loss_clip += 1
        return ((self.module.weight - 0.25) ** 2).mean()


def build_batch(batch_size: int = 4, seq_len: int = 6) -> dict[str, torch.Tensor]:
    images = torch.randn(batch_size, 3, 32, 16)
    caption_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for idx in range(batch_size):
        caption_ids[idx, idx % seq_len] = 1
    token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return {"images": images, "caption_ids": caption_ids, "token_mask": token_mask}


def build_config(train_mode: TrainMode, stage: str) -> IntegrationConfig:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["train_mode"] = train_mode
    cfg["host"]["lambda_s"] = 0.5 if train_mode == "itself" else None
    cfg["prototype"]["dim"] = 8
    cfg["prototype"]["num_prototypes"] = 5
    cfg["prototype"]["regularization"]["diversity"]["weight"] = 1.0
    cfg["prototype"]["regularization"]["balance"]["weight"] = 1.0

    cfg["training"]["stage"] = stage
    if stage == "stage0":
        cfg["prototype"]["enabled"] = False
        cfg["training"]["calibration_only"] = False
        cfg["training"]["freeze"]["host"] = False
        cfg["training"]["freeze"]["prototype"] = True
        cfg["training"]["freeze"]["prototype_policy"] = "freeze_all"
        cfg["loss"]["host_enabled"] = True
        cfg["loss"]["prototype_ret_enabled"] = False
        cfg["loss"]["prototype_diag_enabled"] = False
        cfg["loss"]["prototype_div_enabled"] = False
        cfg["loss"]["prototype_bal_enabled"] = False
    elif stage == "stage1":
        cfg["prototype"]["enabled"] = True
        cfg["training"]["calibration_only"] = False
        cfg["training"]["freeze"]["host"] = True
        cfg["training"]["freeze"]["prototype"] = False
        cfg["training"]["freeze"]["prototype_policy"] = "train_all"
        cfg["loss"]["host_enabled"] = False
        cfg["loss"]["prototype_ret_enabled"] = True
        cfg["loss"]["prototype_diag_enabled"] = True
        cfg["loss"]["prototype_div_enabled"] = True
        cfg["loss"]["prototype_bal_enabled"] = False
    elif stage == "stage2":
        cfg["prototype"]["enabled"] = True
        cfg["training"]["calibration_only"] = False
        cfg["training"]["freeze"]["host"] = False
        cfg["training"]["freeze"]["prototype"] = False
        cfg["training"]["freeze"]["prototype_policy"] = "train_all"
        cfg["training"]["initialization"]["clip_backbone_source"] = "openai:ViT-B/16"
        cfg["training"]["initialization"]["stage1_prototype_checkpoint"] = None
        cfg["training"]["initialization"]["host_checkpoint"] = None
        cfg["training"]["initialization"]["host_checkpoint_compatible"] = False
        cfg["loss"]["host_enabled"] = True
        cfg["loss"]["prototype_ret_enabled"] = True
        cfg["loss"]["prototype_diag_enabled"] = True
        cfg["loss"]["prototype_div_enabled"] = True
        cfg["loss"]["prototype_bal_enabled"] = True
    elif stage == "stage3":
        cfg["prototype"]["enabled"] = True
        cfg["training"]["calibration_only"] = True
        cfg["training"]["freeze"]["host"] = True
        cfg["training"]["freeze"]["prototype"] = True
        cfg["training"]["freeze"]["prototype_policy"] = "freeze_all"
        cfg["loss"]["host_enabled"] = False
        cfg["loss"]["prototype_ret_enabled"] = False
        cfg["loss"]["prototype_diag_enabled"] = False
        cfg["loss"]["prototype_div_enabled"] = False
        cfg["loss"]["prototype_bal_enabled"] = False
    else:
        raise ValueError(f"Unsupported stage {stage!r}")

    return load_integration_config(cfg)
