"""Image-conditioned routing over prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PrototypeRouterConfig:
    """Router config.

    Attributes:
        temperature: Routing temperature `tau_p`.
        normalize_inputs: Use cosine-style normalized similarity when True.
    """

    temperature: float = 0.07
    normalize_inputs: bool = True

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")


@dataclass(frozen=True)
class RoutingOutput:
    """Routing result.

    Shapes:
        - alpha: [B, N]
        - q_summary: [B, D]
        - routing_logits: [B, N]
    """

    alpha: torch.Tensor
    q_summary: torch.Tensor
    routing_logits: torch.Tensor

    def __post_init__(self) -> None:
        if self.alpha.ndim != 2:
            raise ValueError(f"alpha must be [B, N], got {tuple(self.alpha.shape)}")
        if self.q_summary.ndim != 2:
            raise ValueError(
                f"q_summary must be [B, D], got {tuple(self.q_summary.shape)}"
            )
        if self.routing_logits.ndim != 2:
            raise ValueError(
                "routing_logits must be [B, N], got "
                f"{tuple(self.routing_logits.shape)}"
            )
        if self.alpha.shape != self.routing_logits.shape:
            raise ValueError("alpha and routing_logits must have identical shape")
        if self.alpha.shape[0] != self.q_summary.shape[0]:
            raise ValueError("alpha and q_summary batch size mismatch")


class PrototypeRouter(nn.Module):
    """Route host global image feature `[B, D]` over prototypes `[N, D]`."""

    def __init__(self, config: PrototypeRouterConfig | None = None) -> None:
        super().__init__()
        self.config = config or PrototypeRouterConfig()

    def route(
        self, v_i_global: torch.Tensor, contextualized_prototypes: torch.Tensor
    ) -> RoutingOutput:
        """Return routing weights and image-conditioned prototype summary."""
        if v_i_global.ndim != 2:
            raise ValueError(
                f"v_i_global must be rank-2 [B, D], got {tuple(v_i_global.shape)}"
            )
        if contextualized_prototypes.ndim != 2:
            raise ValueError(
                "contextualized_prototypes must be rank-2 [N, D], got "
                f"{tuple(contextualized_prototypes.shape)}"
            )
        if v_i_global.shape[-1] != contextualized_prototypes.shape[-1]:
            raise ValueError(
                "Feature dim mismatch between v_i_global and contextualized_prototypes: "
                f"{v_i_global.shape[-1]} vs {contextualized_prototypes.shape[-1]}"
            )

        if self.config.normalize_inputs:
            images = F.normalize(v_i_global, p=2, dim=-1)
            prototypes = F.normalize(contextualized_prototypes, p=2, dim=-1)
        else:
            images = v_i_global
            prototypes = contextualized_prototypes

        routing_logits = (images @ prototypes.t()) / self.config.temperature
        alpha = torch.softmax(routing_logits, dim=-1)
        q_summary = alpha @ contextualized_prototypes

        return RoutingOutput(
            alpha=alpha, q_summary=q_summary, routing_logits=routing_logits
        )

    def forward(
        self, v_i_global: torch.Tensor, contextualized_prototypes: torch.Tensor
    ) -> RoutingOutput:
        """Alias for routing call."""
        return self.route(v_i_global, contextualized_prototypes)
