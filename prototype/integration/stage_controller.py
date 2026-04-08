"""Stage policy contracts for Stage 0/1/2/3 execution.

This file declares configuration and policy interfaces only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .feature_surface import TrainMode


StageName = Literal["stage0", "stage1", "stage2", "stage3"]


@dataclass(frozen=True)
class StageConfig:
    """Minimal stage configuration contract."""

    train_mode: TrainMode
    stage: StageName
    prototype_enabled: bool


@dataclass(frozen=True)
class StagePolicy:
    """Resolved stage policy contract.

    This policy defines what is trainable and which losses are active.
    """

    train_mode: TrainMode
    stage: StageName
    freeze_host: bool
    freeze_prototype: bool
    host_loss_enabled: bool
    prototype_ret_enabled: bool
    prototype_diag_enabled: bool
    prototype_div_enabled: bool
    prototype_bal_enabled: bool


class StageController:
    """Stage policy resolver interface."""

    def resolve(self, config: StageConfig) -> StagePolicy:
        """Resolve stage policy from config.

        Constraints are defined by the integration contract and implemented later.
        """
        raise NotImplementedError
