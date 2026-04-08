"""Integration-layer interfaces for host runtime and stage control."""

from .feature_surface import (
    CLIPHostFeatureSurface,
    CLIPHostScoreSurface,
    HostFeatureSurface,
    HostScoreSurface,
    ITSELFHostFeatureSurface,
    ITSELFHostScoreSurface,
)
from .host_runtime import HostRuntime, HostRuntimeConfig
from .model_runtime import IntegratedRuntimeConfig, IntegratedScoringRuntime
from .stage_controller import StageConfig, StageController, StagePolicy
from .training_runtime import (
    IntegratedTrainingRuntime,
    PrototypeBranchRuntime,
    TrainingRuntimeHooks,
)

__all__ = [
    "CLIPHostFeatureSurface",
    "CLIPHostScoreSurface",
    "HostFeatureSurface",
    "HostRuntime",
    "HostRuntimeConfig",
    "HostScoreSurface",
    "IntegratedRuntimeConfig",
    "IntegratedScoringRuntime",
    "ITSELFHostFeatureSurface",
    "ITSELFHostScoreSurface",
    "IntegratedTrainingRuntime",
    "PrototypeBranchRuntime",
    "StageConfig",
    "StageController",
    "StagePolicy",
    "TrainingRuntimeHooks",
]
