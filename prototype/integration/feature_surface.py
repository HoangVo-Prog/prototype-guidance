"""Data containers for host feature and score surfaces.

These containers define tensor provenance and expected shapes for Phase A contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


TrainMode = Literal["itself", "clip"]


@dataclass(frozen=True)
class HostFeatureSurface:
    """Canonical host features consumed by the prototype branch.

    Tensor shape expectations:
        - v_i_global: [B, D] from active host global image retrieval feature
        - h_j_tokens: [B, L, D] token-level text states (never pooled text)
        - t_j_global: [B, D] pooled global text feature from active host
        - z_i_retrieval: [B, D_r] image retrieval feature used for prototype scoring
    """

    train_mode: TrainMode
    v_i_global: Any
    h_j_tokens: Any
    t_j_global: Any
    z_i_retrieval: Any
    v_i_local: Any | None = None
    t_j_local: Any | None = None


@dataclass(frozen=True)
class HostScoreSurface:
    """Canonical host score surface bound to the active train mode.

    Tensor shape expectation:
        - s_host: [B, B]
    """

    train_mode: TrainMode
    s_host: Any
    s_global: Any | None = None
    s_local: Any | None = None
