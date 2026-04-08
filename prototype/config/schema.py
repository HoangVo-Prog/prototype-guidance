"""Config schema contracts for integration runtime.

Phase A provides dataclass schema definitions and validation interface only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TrainMode = Literal["itself", "clip"]
TrainingStage = Literal["stage0", "stage1", "stage2", "stage3"]


@dataclass(frozen=True)
class PrototypeConfig:
    """Prototype-branch configuration contract."""

    enabled: bool
    num_prototypes: int
    dim: int
    contextualization_enabled: bool
    routing_temperature: float
    basis_temperature: float
    teacher_temperature: float
    retrieval_temperature: float


@dataclass(frozen=True)
class FusionConfig:
    """Fusion configuration contract."""

    enabled: bool
    lambda_f: float


@dataclass(frozen=True)
class HostConfig:
    """Host configuration contract."""

    lambda_s: float | None


@dataclass(frozen=True)
class TrainingConfig:
    """Training-stage control contract."""

    stage: TrainingStage
    freeze_host: bool
    freeze_prototype: bool


@dataclass(frozen=True)
class LossConfig:
    """Loss toggle contract."""

    host_enabled: bool
    prototype_ret_enabled: bool
    prototype_diag_enabled: bool
    prototype_div_enabled: bool
    prototype_bal_enabled: bool


@dataclass(frozen=True)
class IntegrationConfig:
    """Top-level integration config contract."""

    train_mode: TrainMode
    prototype: PrototypeConfig
    fusion: FusionConfig
    host: HostConfig
    training: TrainingConfig
    loss: LossConfig


def validate_integration_config(config: IntegrationConfig) -> None:
    """Validate config contract and stage/mode guards."""
    raise NotImplementedError
