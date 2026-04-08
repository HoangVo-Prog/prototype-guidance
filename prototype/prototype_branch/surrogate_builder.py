"""Surrogate text construction interface."""

from __future__ import annotations

from typing import Any


class PrototypeSurrogateBuilder:
    """Build image-conditioned surrogate text objects.

    Row-wise semantics MUST be preserved:
        - row i corresponds to image i
        - columns correspond to captions j
    """

    def build_surrogate(self, alpha: Any, basis_bank: Any) -> Any:
        """Return surrogate text tensor for pairwise scoring.

        Expected logical output shape is [B, B, D] (chunked computation allowed).
        """
        raise NotImplementedError
