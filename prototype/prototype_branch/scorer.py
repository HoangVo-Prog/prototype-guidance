"""Prototype score interface."""

from __future__ import annotations

from typing import Any


class PrototypeScorer:
    """Compute prototype score matrix from host image and surrogate text features."""

    def score(self, z_i_retrieval: Any, z_hat_text: Any) -> Any:
        """Return pairwise prototype score matrix with shape [B, B]."""
        raise NotImplementedError
