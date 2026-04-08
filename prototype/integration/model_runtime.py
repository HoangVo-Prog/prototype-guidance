"""Integrated score runtime for Phase D score-level fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from prototype.fusion import ResidualFusionConfig, ResidualFusionOutput, ResidualScoreFusion
from prototype.prototype_branch.scorer import PrototypeScoreSurface

from .feature_surface import CLIPHostScoreSurface, HostScoreSurface, ITSELFHostScoreSurface, TrainMode

Tensor = torch.Tensor


def _assert_score_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank-2 score tensor [B, B], got {tuple(value.shape)}")
    if value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be square [B, B], got {tuple(value.shape)}")


@dataclass(frozen=True)
class IntegratedRuntimeConfig:
    """Config for mode-bound integrated score fusion runtime."""

    train_mode: TrainMode
    lambda_f: float


@dataclass(frozen=True)
class IntegratedITSELFScoreOutput:
    """Integrated score output for `train_mode=itself`."""

    train_mode: Literal["itself"]
    s_host_itself: Tensor
    s_proto: Tensor
    s_total: Tensor
    lambda_f: float

    def __post_init__(self) -> None:
        _assert_score_tensor("s_host_itself", self.s_host_itself)
        _assert_score_tensor("s_proto", self.s_proto)
        _assert_score_tensor("s_total", self.s_total)
        if self.s_host_itself.shape != self.s_proto.shape or self.s_host_itself.shape != self.s_total.shape:
            raise ValueError("ITSELF score tensors must share identical [B, B] shape")


@dataclass(frozen=True)
class IntegratedCLIPScoreOutput:
    """Integrated score output for `train_mode=clip`."""

    train_mode: Literal["clip"]
    s_host_clip: Tensor
    s_proto: Tensor
    s_total: Tensor
    lambda_f: float

    def __post_init__(self) -> None:
        _assert_score_tensor("s_host_clip", self.s_host_clip)
        _assert_score_tensor("s_proto", self.s_proto)
        _assert_score_tensor("s_total", self.s_total)
        if self.s_host_clip.shape != self.s_proto.shape or self.s_host_clip.shape != self.s_total.shape:
            raise ValueError("CLIP score tensors must share identical [B, B] shape")


IntegratedScoreOutput = IntegratedITSELFScoreOutput | IntegratedCLIPScoreOutput


class IntegratedScoringRuntime:
    """Thin score-level integration runtime.

    This runtime consumes mode-bound host score surfaces and prototype score surfaces
    and applies residual score fusion only at `[B, B]` score tensor level.
    """

    def __init__(self, config: IntegratedRuntimeConfig) -> None:
        self.config = config
        self._fusion = ResidualScoreFusion(ResidualFusionConfig(lambda_f=config.lambda_f))

    @staticmethod
    def _to_s_proto_tensor(prototype_score: PrototypeScoreSurface | Tensor) -> Tensor:
        if isinstance(prototype_score, PrototypeScoreSurface):
            return prototype_score.s_proto
        return prototype_score

    def fuse_itself_mode(
        self, host_scores: ITSELFHostScoreSurface, prototype_score: PrototypeScoreSurface | Tensor
    ) -> IntegratedITSELFScoreOutput:
        """Fuse ITSELF host score surface with prototype score tensor."""
        s_proto = self._to_s_proto_tensor(prototype_score)
        fused: ResidualFusionOutput = self._fusion.fuse_itself_scores(
            s_host_itself=host_scores.s_host_itself, s_proto=s_proto
        )
        return IntegratedITSELFScoreOutput(
            train_mode="itself",
            s_host_itself=host_scores.s_host_itself,
            s_proto=fused.s_proto,
            s_total=fused.s_total,
            lambda_f=fused.lambda_f,
        )

    def fuse_clip_mode(
        self, host_scores: CLIPHostScoreSurface, prototype_score: PrototypeScoreSurface | Tensor
    ) -> IntegratedCLIPScoreOutput:
        """Fuse CLIP host score surface with prototype score tensor."""
        s_proto = self._to_s_proto_tensor(prototype_score)
        fused: ResidualFusionOutput = self._fusion.fuse_clip_scores(
            s_host_clip=host_scores.s_host_clip, s_proto=s_proto
        )
        return IntegratedCLIPScoreOutput(
            train_mode="clip",
            s_host_clip=host_scores.s_host_clip,
            s_proto=fused.s_proto,
            s_total=fused.s_total,
            lambda_f=fused.lambda_f,
        )

    def fuse_active_mode(
        self, host_scores: HostScoreSurface, prototype_score: PrototypeScoreSurface | Tensor
    ) -> IntegratedScoreOutput:
        """Fuse score surfaces for the configured active mode with explicit mode checks."""
        if self.config.train_mode == "itself":
            if not isinstance(host_scores, ITSELFHostScoreSurface):
                raise TypeError(
                    "Configured train_mode='itself' requires ITSELFHostScoreSurface input"
                )
            return self.fuse_itself_mode(host_scores=host_scores, prototype_score=prototype_score)

        if not isinstance(host_scores, CLIPHostScoreSurface):
            raise TypeError(
                "Configured train_mode='clip' requires CLIPHostScoreSurface input"
            )
        return self.fuse_clip_mode(host_scores=host_scores, prototype_score=prototype_score)
