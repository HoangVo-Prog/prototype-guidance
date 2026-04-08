"""Prototype contextualization interface."""

from __future__ import annotations

from typing import Any


class PrototypeContextualizer:
    """Contextualize prototypes without changing host semantics."""

    def contextualize(self, prototypes: Any) -> Any:
        """Return contextualized prototypes with shape [N, D]."""
        raise NotImplementedError
