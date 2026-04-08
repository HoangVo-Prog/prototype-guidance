"""Residual score-level fusion interface.

Fusion contract (exact):
    s_total = s_host + lambda_f * s_proto
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResidualFusionConfig:
    """Config contract for residual score-level fusion."""

    lambda_f: float


class ResidualScoreFusion:
    """Score-level fusion boundary.

    Embedding-level fusion is forbidden by contract.
    """

    def __init__(self, config: ResidualFusionConfig) -> None:
        self.config = config

    def fuse(self, s_host: Any, s_proto: Any) -> Any:
        """Fuse host and prototype scores at score tensor level only."""
        raise NotImplementedError
