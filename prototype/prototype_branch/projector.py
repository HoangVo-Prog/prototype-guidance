"""Prototype text projection interface."""

from __future__ import annotations

from typing import Any


class PrototypeProjector:
    """Project surrogate text objects into prototype retrieval space."""

    def project(self, surrogate_text: Any) -> Any:
        """Return projected surrogate embeddings with last dim D_r."""
        raise NotImplementedError
