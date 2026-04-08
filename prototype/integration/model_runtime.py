"""Integrated score-runtime contracts.

This runtime fuses host and prototype scores at score level only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .feature_surface import HostFeatureSurface, HostScoreSurface, TrainMode


@dataclass(frozen=True)
class IntegratedRuntimeConfig:
    """Runtime config contract for integrated score computation."""

    train_mode: TrainMode
    lambda_f: float


class IntegratedScoringRuntime:
    """Interface-only integrated runtime.

    Future implementation MUST preserve:
        s_total = s_host + lambda_f * s_proto
    """

    def __init__(self, config: IntegratedRuntimeConfig) -> None:
        self.config = config

    def extract_host_features(self, batch: Mapping[str, Any]) -> HostFeatureSurface:
        """Return active-host feature surface for prototype branch."""
        raise NotImplementedError

    def compute_host_scores(self, features: HostFeatureSurface) -> HostScoreSurface:
        """Return active-host score surface bound to train_mode."""
        raise NotImplementedError

    def compute_prototype_scores(self, features: HostFeatureSurface) -> Any:
        """Return prototype score matrix with shape [B, B]."""
        raise NotImplementedError

    def compute_total_scores(self, score_surface: HostScoreSurface, s_proto: Any) -> Any:
        """Return fused score matrix with exact residual score-level fusion."""
        raise NotImplementedError
