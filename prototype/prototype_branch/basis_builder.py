"""Prototype-conditioned text basis builder interface."""

from __future__ import annotations

from typing import Any


class PrototypeBasisBuilder:
    """Build caption basis bank from token-level host text states.

    Input MUST be token states `H_j` with shape [B, L, D].
    Pooled text features are forbidden for this interface.
    """

    def build_basis(self, h_j_tokens: Any, contextualized_prototypes: Any) -> Any:
        """Return caption basis bank `B_j` with shape [B, N, D]."""
        raise NotImplementedError
