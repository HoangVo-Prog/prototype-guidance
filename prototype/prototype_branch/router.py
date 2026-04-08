"""Image-conditioned router interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoutingOutput:
    """Routing result contract.

    Shape expectations:
        - alpha: [B, N], row-wise softmax probabilities
        - q_summary: [B, D], alpha @ prototypes
    """

    alpha: Any
    q_summary: Any


class PrototypeRouter:
    """Route host global image features over prototype bank."""

    def route(self, v_i_global: Any, contextualized_prototypes: Any) -> RoutingOutput:
        """Return routing weights and image-conditioned prototype summary."""
        raise NotImplementedError
