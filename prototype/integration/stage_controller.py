"""Stage-control resolution from validated integration config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from prototype.config.schema import (
    IntegrationConfig,
    TrainMode,
    TrainingStage,
    validate_integration_config,
)


StageName = Literal["stage0", "stage1", "stage2", "stage3"]


@dataclass(frozen=True)
class StageConfig:
    """Stage-control input wrapper."""

    integration: IntegrationConfig


@dataclass(frozen=True)
class StagePolicy:
    """Resolved stage policy intent.

    This is declarative control intent only; it does not execute optimizer or loss routing.
    """

    train_mode: TrainMode
    stage: TrainingStage
    prototype_enabled: bool
    freeze_host: bool
    freeze_prototype: bool
    host_loss_enabled: bool
    prototype_ret_enabled: bool
    prototype_diag_enabled: bool
    prototype_div_enabled: bool
    prototype_bal_enabled: bool
    calibration_only: bool
    requires_clip_init: bool
    clip_backbone_source: str | None
    host_lambda_s_required: bool
    host_lambda_s_applies_to_s_host: bool


class StageController:
    """Resolve stage/mode policy from validated config."""

    def resolve(self, config: StageConfig) -> StagePolicy:
        """Resolve stage policy.

        Validation is enforced before policy resolution.
        """
        integration = config.integration
        validate_integration_config(integration)

        stage = integration.training.stage
        train_mode = integration.train_mode

        if stage == "stage0":
            return StagePolicy(
                train_mode=train_mode,
                stage=stage,
                prototype_enabled=False,
                freeze_host=False,
                freeze_prototype=True,
                host_loss_enabled=True,
                prototype_ret_enabled=False,
                prototype_diag_enabled=False,
                prototype_div_enabled=False,
                prototype_bal_enabled=False,
                calibration_only=False,
                requires_clip_init=False,
                clip_backbone_source=None,
                host_lambda_s_required=(train_mode == "itself"),
                host_lambda_s_applies_to_s_host=(train_mode == "itself"),
            )

        if stage == "stage1":
            return StagePolicy(
                train_mode=train_mode,
                stage=stage,
                prototype_enabled=True,
                freeze_host=True,
                freeze_prototype=False,
                host_loss_enabled=False,
                prototype_ret_enabled=True,
                prototype_diag_enabled=True,
                prototype_div_enabled=True,
                prototype_bal_enabled=integration.loss.prototype_bal_enabled,
                calibration_only=False,
                requires_clip_init=False,
                clip_backbone_source=None,
                host_lambda_s_required=(train_mode == "itself"),
                host_lambda_s_applies_to_s_host=(train_mode == "itself"),
            )

        if stage == "stage2":
            return StagePolicy(
                train_mode=train_mode,
                stage=stage,
                prototype_enabled=True,
                freeze_host=False,
                freeze_prototype=integration.training.freeze.prototype,
                host_loss_enabled=True,
                prototype_ret_enabled=integration.loss.prototype_ret_enabled,
                prototype_diag_enabled=integration.loss.prototype_diag_enabled,
                prototype_div_enabled=integration.loss.prototype_div_enabled,
                prototype_bal_enabled=integration.loss.prototype_bal_enabled,
                calibration_only=False,
                requires_clip_init=True,
                clip_backbone_source=integration.training.initialization.clip_backbone_source,
                host_lambda_s_required=(train_mode == "itself"),
                host_lambda_s_applies_to_s_host=(train_mode == "itself"),
            )

        return StagePolicy(
            train_mode=train_mode,
            stage=stage,
            prototype_enabled=True,
            freeze_host=True,
            freeze_prototype=True,
            host_loss_enabled=False,
            prototype_ret_enabled=False,
            prototype_diag_enabled=False,
            prototype_div_enabled=False,
            prototype_bal_enabled=False,
            calibration_only=True,
            requires_clip_init=False,
            clip_backbone_source=None,
            host_lambda_s_required=(train_mode == "itself"),
            host_lambda_s_applies_to_s_host=(train_mode == "itself"),
        )
