"""Residual score-level fusion with strict score-tensor guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Tensor = torch.Tensor
TrainMode = Literal["itself", "clip"]


def _assert_score_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    if value.ndim != 2:
        raise ValueError(
            f"{name} must be rank-2 score tensor [B, B], got {tuple(value.shape)}"
        )
    if value.shape[0] != value.shape[1]:
        raise ValueError(
            f"{name} must be square score tensor [B, B], got {tuple(value.shape)}"
        )


@dataclass(frozen=True)
class ResidualFusionConfig:
    """Residual fusion config."""

    lambda_f: float


@dataclass(frozen=True)
class ResidualFusionOutput:
    """Residual fusion output."""

    train_mode: TrainMode
    s_total: Tensor
    s_proto: Tensor
    lambda_f: float
    s_host_itself: Tensor | None = None
    s_host_clip: Tensor | None = None

    def __post_init__(self) -> None:
        _assert_score_tensor("s_total", self.s_total)
        _assert_score_tensor("s_proto", self.s_proto)
        if self.s_total.shape != self.s_proto.shape:
            raise ValueError("s_total and s_proto must have identical [B, B] shape")
        if self.train_mode == "itself":
            if self.s_host_itself is None:
                raise ValueError("s_host_itself is required for train_mode='itself'")
            _assert_score_tensor("s_host_itself", self.s_host_itself)
            if self.s_host_clip is not None:
                raise ValueError("s_host_clip must be None for train_mode='itself'")
        else:
            if self.s_host_clip is None:
                raise ValueError("s_host_clip is required for train_mode='clip'")
            _assert_score_tensor("s_host_clip", self.s_host_clip)
            if self.s_host_itself is not None:
                raise ValueError("s_host_itself must be None for train_mode='clip'")


class ResidualScoreFusion:
    """Score-level fusion boundary.

    This module MUST NOT recompute host scores and MUST NOT fuse embeddings.
    """

    def __init__(self, config: ResidualFusionConfig) -> None:
        self.config = config

    def _fuse_scores(self, host_mode: TrainMode, host_score: Tensor, s_proto: Tensor) -> ResidualFusionOutput:
        _assert_score_tensor("host_score", host_score)
        _assert_score_tensor("s_proto", s_proto)
        if host_score.shape != s_proto.shape:
            raise ValueError(
                "host score and prototype score shapes must match exactly; "
                f"got {tuple(host_score.shape)} vs {tuple(s_proto.shape)}"
            )

        # Exact parity path.
        if self.config.lambda_f == 0.0:
            s_total = host_score
        else:
            s_total = host_score + (self.config.lambda_f * s_proto)

        if host_mode == "itself":
            return ResidualFusionOutput(
                train_mode="itself",
                s_host_itself=host_score,
                s_host_clip=None,
                s_proto=s_proto,
                s_total=s_total,
                lambda_f=self.config.lambda_f,
            )

        return ResidualFusionOutput(
            train_mode="clip",
            s_host_itself=None,
            s_host_clip=host_score,
            s_proto=s_proto,
            s_total=s_total,
            lambda_f=self.config.lambda_f,
        )

    def fuse_itself_scores(self, s_host_itself: Tensor, s_proto: Tensor) -> ResidualFusionOutput:
        """Fuse ITSELF host score and prototype score."""
        return self._fuse_scores(host_mode="itself", host_score=s_host_itself, s_proto=s_proto)

    def fuse_clip_scores(self, s_host_clip: Tensor, s_proto: Tensor) -> ResidualFusionOutput:
        """Fuse CLIP host score and prototype score."""
        return self._fuse_scores(host_mode="clip", host_score=s_host_clip, s_proto=s_proto)
