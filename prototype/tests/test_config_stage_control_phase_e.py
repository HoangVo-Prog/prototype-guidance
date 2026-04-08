"""Phase E tests for config schema and stage-control validation."""

from __future__ import annotations

import copy

import pytest

from prototype.config.defaults import DEFAULT_CONFIG
from prototype.config.schema import (
    SUPPORTED_STAGE_MODE_MATRIX,
    load_integration_config,
    validate_integration_config,
)
from prototype.integration.stage_controller import StageConfig, StageController


def _cfg() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def test_valid_train_mode_parsing() -> None:
    cfg = _cfg()
    cfg["train_mode"] = "itself"
    cfg["host"]["lambda_s"] = 0.5
    _ = load_integration_config(cfg)

    cfg = _cfg()
    cfg["train_mode"] = "clip"
    cfg["host"]["lambda_s"] = None
    _ = load_integration_config(cfg)


def test_invalid_train_mode_rejected() -> None:
    cfg = _cfg()
    cfg["train_mode"] = "bad_mode"
    with pytest.raises(ValueError):
        _ = load_integration_config(cfg)


def test_valid_stage_parsing() -> None:
    for stage in ("stage0", "stage1", "stage2", "stage3"):
        cfg = _cfg()
        cfg["training"]["stage"] = stage
        cfg["host"]["lambda_s"] = 0.5
        cfg["train_mode"] = "itself"

        if stage == "stage0":
            cfg["prototype"]["enabled"] = False
            cfg["training"]["freeze"]["host"] = False
            cfg["loss"]["host_enabled"] = True
        elif stage == "stage1":
            cfg["prototype"]["enabled"] = True
            cfg["training"]["freeze"]["host"] = True
            cfg["loss"]["host_enabled"] = False
            cfg["loss"]["prototype_ret_enabled"] = True
            cfg["loss"]["prototype_diag_enabled"] = True
            cfg["loss"]["prototype_div_enabled"] = True
        elif stage == "stage2":
            cfg["prototype"]["enabled"] = True
            cfg["training"]["freeze"]["host"] = False
            cfg["loss"]["host_enabled"] = True
            cfg["training"]["initialization"]["clip_backbone_source"] = "openai:ViT-B/16"
            cfg["loss"]["prototype_ret_enabled"] = True
            cfg["loss"]["prototype_diag_enabled"] = True
            cfg["loss"]["prototype_div_enabled"] = True
        elif stage == "stage3":
            cfg["prototype"]["enabled"] = True
            cfg["training"]["freeze"]["host"] = True
            cfg["training"]["freeze"]["prototype"] = True
            cfg["training"]["calibration_only"] = True
            cfg["loss"]["host_enabled"] = False
            cfg["loss"]["prototype_ret_enabled"] = False
            cfg["loss"]["prototype_diag_enabled"] = False
            cfg["loss"]["prototype_div_enabled"] = False
            cfg["loss"]["prototype_bal_enabled"] = False
        _ = load_integration_config(cfg)


def test_invalid_stage_rejected() -> None:
    cfg = _cfg()
    cfg["training"]["stage"] = "stage9"
    with pytest.raises(ValueError):
        _ = load_integration_config(cfg)


def test_itself_requires_host_lambda_s() -> None:
    cfg = _cfg()
    cfg["train_mode"] = "itself"
    cfg["host"]["lambda_s"] = None
    with pytest.raises(ValueError):
        _ = load_integration_config(cfg)


def test_clip_rejects_host_lambda_s_for_s_host_definition() -> None:
    cfg = _cfg()
    cfg["train_mode"] = "clip"
    cfg["host"]["lambda_s"] = 0.4
    with pytest.raises(ValueError):
        _ = load_integration_config(cfg)


def test_stage0_behavior_for_itself_and_clip() -> None:
    cfg_itself = _cfg()
    cfg_itself["train_mode"] = "itself"
    cfg_itself["host"]["lambda_s"] = 0.6
    parsed_itself = load_integration_config(cfg_itself)
    validate_integration_config(parsed_itself)

    cfg_clip = _cfg()
    cfg_clip["train_mode"] = "clip"
    cfg_clip["host"]["lambda_s"] = None
    parsed_clip = load_integration_config(cfg_clip)
    validate_integration_config(parsed_clip)


def test_stage1_requires_frozen_host_intent() -> None:
    cfg = _cfg()
    cfg["training"]["stage"] = "stage1"
    cfg["prototype"]["enabled"] = True
    cfg["training"]["freeze"]["host"] = False
    cfg["loss"]["host_enabled"] = False
    cfg["loss"]["prototype_ret_enabled"] = True
    cfg["loss"]["prototype_diag_enabled"] = True
    cfg["loss"]["prototype_div_enabled"] = True
    with pytest.raises(ValueError):
        _ = load_integration_config(cfg)


def test_stage2_requires_clip_init_compatible_path() -> None:
    cfg = _cfg()
    cfg["training"]["stage"] = "stage2"
    cfg["prototype"]["enabled"] = True
    cfg["training"]["freeze"]["host"] = False
    cfg["loss"]["host_enabled"] = True
    cfg["loss"]["prototype_ret_enabled"] = True
    cfg["loss"]["prototype_diag_enabled"] = True
    cfg["loss"]["prototype_div_enabled"] = True
    cfg["training"]["initialization"]["clip_backbone_source"] = None
    with pytest.raises(ValueError):
        _ = load_integration_config(cfg)


def test_stage3_requires_calibration_only_intent() -> None:
    cfg = _cfg()
    cfg["training"]["stage"] = "stage3"
    cfg["prototype"]["enabled"] = True
    cfg["training"]["freeze"]["host"] = True
    cfg["training"]["freeze"]["prototype"] = True
    cfg["training"]["calibration_only"] = False
    cfg["loss"]["host_enabled"] = False
    cfg["loss"]["prototype_ret_enabled"] = False
    cfg["loss"]["prototype_diag_enabled"] = False
    cfg["loss"]["prototype_div_enabled"] = False
    cfg["loss"]["prototype_bal_enabled"] = False
    with pytest.raises(ValueError):
        _ = load_integration_config(cfg)


def test_supported_stage_mode_matrix_exposed() -> None:
    assert ("itself", "stage0") in SUPPORTED_STAGE_MODE_MATRIX
    assert ("clip", "stage3") in SUPPORTED_STAGE_MODE_MATRIX


def test_stage_controller_resolves_stage2_policy() -> None:
    cfg = _cfg()
    cfg["training"]["stage"] = "stage2"
    cfg["prototype"]["enabled"] = True
    cfg["training"]["freeze"]["host"] = False
    cfg["training"]["freeze"]["prototype"] = False
    cfg["training"]["freeze"]["prototype_policy"] = "partial"
    cfg["training"]["initialization"]["clip_backbone_source"] = "openai:ViT-B/16"
    cfg["loss"]["host_enabled"] = True
    cfg["loss"]["prototype_ret_enabled"] = True
    cfg["loss"]["prototype_diag_enabled"] = True
    cfg["loss"]["prototype_div_enabled"] = True

    parsed = load_integration_config(cfg)
    policy = StageController().resolve(StageConfig(integration=parsed))
    assert policy.stage == "stage2"
    assert policy.requires_clip_init is True
    assert policy.freeze_host is False
