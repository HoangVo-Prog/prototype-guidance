"""Prototype bank interfaces.

This module defines storage and contextualization boundaries for learnable prototypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrototypeBankConfig:
    """Prototype bank config contract."""

    num_prototypes: int
    feature_dim: int
    contextualization_enabled: bool = False


@dataclass(frozen=True)
class PrototypeBankOutput:
    """Prototype bank output contract.

    Shape expectations:
        - prototypes: [N, D]
        - contextualized_prototypes: [N, D]
    """

    prototypes: Any
    contextualized_prototypes: Any


class PrototypeBankStore:
    """Interface for prototype bank lifecycle."""

    def __init__(self, config: PrototypeBankConfig) -> None:
        self.config = config

    def get_bank(self) -> PrototypeBankOutput:
        """Return raw and contextualized prototype tensors."""
        raise NotImplementedError
