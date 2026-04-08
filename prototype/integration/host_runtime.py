"""Host runtime contracts for `train_mode=itself` and `train_mode=clip`.

This module defines stable interfaces only. No host behavior is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .feature_surface import HostFeatureSurface, HostScoreSurface


@dataclass(frozen=True)
class HostRuntimeConfig:
    """Config contract for host runtime selection.

    Attributes:
        train_mode: Exact mode selector. Allowed values are `itself` or `clip`.
        lambda_s: Host global/local mix weight for `train_mode=itself`.
    """

    train_mode: str
    lambda_s: float | None = None


class HostRuntime(Protocol):
    """Read-only host runtime contract used by the prototype integration layer."""

    def extract_feature_surface(self, batch: Mapping[str, Any]) -> HostFeatureSurface:
        """Return active-host features required by the prototype branch.

        Required tensors by contract:
            - v_i_global: shape [B, D]
            - h_j_tokens: shape [B, L, D]
            - t_j_global: shape [B, D]
        """
        raise NotImplementedError

    def compute_host_score_surface(self, features: HostFeatureSurface) -> HostScoreSurface:
        """Return canonical host score components for the active mode.

        Required score semantics:
            - itself: s_host = lambda_s * s_global + (1 - lambda_s) * s_local
            - clip: s_host = s_host_clip
        """
        raise NotImplementedError
