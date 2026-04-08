"""Phase F tests for executable stage/mode training runtime behavior."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from prototype.config.defaults import DEFAULT_CONFIG
from prototype.config.schema import IntegrationConfig, TrainMode, load_integration_config
from prototype.integration.feature_surface import (
    CLIPHostFeatureSurface,
    CLIPHostScoreSurface,
    HostFeatureSurface,
    HostScoreSurface,
    ITSELFHostFeatureSurface,
    ITSELFHostScoreSurface,
)
from prototype.integration.training_runtime import (
    IntegratedTrainingRuntime,
    TrainingRuntimeHooks,
)


def _score_matrix(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs_norm = F.normalize(lhs, p=2, dim=1)
    rhs_norm = F.normalize(rhs, p=2, dim=1)
    return lhs_norm @ rhs_norm.t()


class _DummyHostRuntime:
    def __init__(self, train_mode: TrainMode, lambda_s: float | None) -> None:
        self.train_mode = train_mode
        self.lambda_s = lambda_s
        self.module = nn.Linear(8, 8, bias=False)
        nn.init.eye_(self.module.weight)
        self.calls_host_loss_itself = 0
        self.calls_host_loss_clip = 0

    def extract_feature_surface(self, batch: Mapping[str, Any]) -> HostFeatureSurface:
        bsz = int(batch["images"].shape[0])
        seq_len = int(batch["caption_ids"].shape[1])
        dim = 8

        v_i_global = torch.linspace(0.1, 0.9, steps=bsz * dim, dtype=torch.float32).reshape(bsz, dim)
        h_j_tokens = torch.linspace(0.2, 1.0, steps=bsz * seq_len * dim, dtype=torch.float32).reshape(
            bsz, seq_len, dim
        )
        eot = batch["caption_ids"].argmax(dim=-1)
        t_j_global = h_j_tokens[torch.arange(bsz), eot]
        z_i_retrieval = v_i_global

        if self.train_mode == "itself":
            v_i_local = v_i_global + 0.3
            t_j_local = t_j_global + 0.4
            return ITSELFHostFeatureSurface(
                v_i_global=v_i_global,
                h_j_tokens=h_j_tokens,
                t_j_global=t_j_global,
                z_i_retrieval=z_i_retrieval,
                v_i_local=v_i_local,
                t_j_local=t_j_local,
                v_i_global_source="dummy.itself.global",
                h_j_tokens_source="dummy.itself.tokens",
                t_j_global_source="dummy.itself.eot",
                z_i_retrieval_source="dummy.itself.retrieval",
                v_i_local_source="dummy.itself.local_image",
                t_j_local_source="dummy.itself.local_text",
            )

        return CLIPHostFeatureSurface(
            v_i_global=v_i_global,
            h_j_tokens=h_j_tokens,
            t_j_global=t_j_global,
            z_i_retrieval=z_i_retrieval,
            v_i_global_source="dummy.clip.global",
            h_j_tokens_source="dummy.clip.tokens",
            t_j_global_source="dummy.clip.eot",
            z_i_retrieval_source="dummy.clip.retrieval",
        )

    def compute_host_score_surface(self, features: HostFeatureSurface) -> HostScoreSurface:
        if self.train_mode == "itself":
            assert isinstance(features, ITSELFHostFeatureSurface)
            s_global = _score_matrix(features.v_i_global, features.t_j_global)
            s_local = _score_matrix(features.v_i_local, features.t_j_local)
            s_host = float(self.lambda_s) * s_global + (1.0 - float(self.lambda_s)) * s_local
            return ITSELFHostScoreSurface(
                s_host_itself=s_host,
                s_global_itself=s_global,
                s_local_itself=s_local,
                lambda_s=float(self.lambda_s),
            )
        assert isinstance(features, CLIPHostFeatureSurface)
        return CLIPHostScoreSurface(s_host_clip=_score_matrix(features.v_i_global, features.t_j_global))

    def compute_host_loss_itself(
        self,
        batch: Mapping[str, Any],
        features: HostFeatureSurface,
        host_scores: HostScoreSurface,
    ) -> torch.Tensor:
        assert self.train_mode == "itself"
        self.calls_host_loss_itself += 1
        return (self.module.weight ** 2).mean()

    def compute_host_loss_clip(
        self,
        batch: Mapping[str, Any],
        features: HostFeatureSurface,
        host_scores: HostScoreSurface,
    ) -> torch.Tensor:
        assert self.train_mode == "clip"
        self.calls_host_loss_clip += 1
        return ((self.module.weight - 0.25) ** 2).mean()


def _batch(batch_size: int = 4, seq_len: int = 6) -> dict[str, torch.Tensor]:
    images = torch.randn(batch_size, 3, 32, 16)
    caption_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for idx in range(batch_size):
        caption_ids[idx, idx % seq_len] = 1
    token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return {"images": images, "caption_ids": caption_ids, "token_mask": token_mask}


def _build_config(train_mode: TrainMode, stage: str) -> IntegrationConfig:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["train_mode"] = train_mode
    cfg["host"]["lambda_s"] = 0.5 if train_mode == "itself" else None
    cfg["prototype"]["dim"] = 8
    cfg["prototype"]["num_prototypes"] = 5
    cfg["prototype"]["regularization"]["diversity"]["weight"] = 1.0
    cfg["prototype"]["regularization"]["balance"]["weight"] = 1.0

    cfg["training"]["stage"] = stage
    if stage == "stage0":
        cfg["prototype"]["enabled"] = False
        cfg["training"]["calibration_only"] = False
        cfg["training"]["freeze"]["host"] = False
        cfg["training"]["freeze"]["prototype"] = True
        cfg["training"]["freeze"]["prototype_policy"] = "freeze_all"
        cfg["loss"]["host_enabled"] = True
        cfg["loss"]["prototype_ret_enabled"] = False
        cfg["loss"]["prototype_diag_enabled"] = False
        cfg["loss"]["prototype_div_enabled"] = False
        cfg["loss"]["prototype_bal_enabled"] = False
    elif stage == "stage1":
        cfg["prototype"]["enabled"] = True
        cfg["training"]["calibration_only"] = False
        cfg["training"]["freeze"]["host"] = True
        cfg["training"]["freeze"]["prototype"] = False
        cfg["training"]["freeze"]["prototype_policy"] = "train_all"
        cfg["loss"]["host_enabled"] = False
        cfg["loss"]["prototype_ret_enabled"] = True
        cfg["loss"]["prototype_diag_enabled"] = True
        cfg["loss"]["prototype_div_enabled"] = True
        cfg["loss"]["prototype_bal_enabled"] = False
    elif stage == "stage2":
        cfg["prototype"]["enabled"] = True
        cfg["training"]["calibration_only"] = False
        cfg["training"]["freeze"]["host"] = False
        cfg["training"]["freeze"]["prototype"] = False
        cfg["training"]["freeze"]["prototype_policy"] = "train_all"
        cfg["training"]["initialization"]["clip_backbone_source"] = "openai:ViT-B/16"
        cfg["training"]["initialization"]["stage1_prototype_checkpoint"] = None
        cfg["training"]["initialization"]["host_checkpoint"] = None
        cfg["training"]["initialization"]["host_checkpoint_compatible"] = False
        cfg["loss"]["host_enabled"] = True
        cfg["loss"]["prototype_ret_enabled"] = True
        cfg["loss"]["prototype_diag_enabled"] = True
        cfg["loss"]["prototype_div_enabled"] = True
        cfg["loss"]["prototype_bal_enabled"] = True
    elif stage == "stage3":
        cfg["prototype"]["enabled"] = True
        cfg["training"]["calibration_only"] = True
        cfg["training"]["freeze"]["host"] = True
        cfg["training"]["freeze"]["prototype"] = True
        cfg["training"]["freeze"]["prototype_policy"] = "freeze_all"
        cfg["loss"]["host_enabled"] = False
        cfg["loss"]["prototype_ret_enabled"] = False
        cfg["loss"]["prototype_diag_enabled"] = False
        cfg["loss"]["prototype_div_enabled"] = False
        cfg["loss"]["prototype_bal_enabled"] = False
    else:
        raise ValueError(f"Unsupported stage {stage!r}")

    return load_integration_config(cfg)


def _build_optimizer(runtime: IntegratedTrainingRuntime) -> torch.optim.Optimizer | None:
    groups = runtime.build_optimizer_param_groups(host_lr=1e-2, prototype_lr=1e-2)
    if not groups:
        return None
    return torch.optim.SGD(groups, momentum=0.0)


def test_supported_stage_mode_matrix_exposed_for_phase_f_runtime() -> None:
    matrix = set(IntegratedTrainingRuntime.supported_stage_mode_matrix())
    assert ("itself", "stage0") in matrix
    assert ("itself", "stage3") in matrix
    assert ("clip", "stage0") in matrix
    assert ("clip", "stage3") in matrix
    assert len(matrix) == 8


def test_host_loss_routing_itself_mode_stage0() -> None:
    config = _build_config(train_mode="itself", stage="stage0")
    host_runtime = _DummyHostRuntime(train_mode="itself", lambda_s=0.5)
    runtime = IntegratedTrainingRuntime(config=config, host_runtime=host_runtime, host_module=host_runtime.module)
    optimizer = _build_optimizer(runtime)
    assert optimizer is not None

    output = runtime.training_step(_batch(), optimizer=optimizer)

    assert host_runtime.calls_host_loss_itself == 1
    assert host_runtime.calls_host_loss_clip == 0
    assert "host_loss_itself" in output.losses.active_loss_names
    assert output.s_proto is None
    assert torch.equal(output.s_total, output.s_host)
    assert tuple(group.group_name for group in output.optimizer_groups) == ("host",)


def test_host_loss_routing_clip_mode_stage0() -> None:
    config = _build_config(train_mode="clip", stage="stage0")
    host_runtime = _DummyHostRuntime(train_mode="clip", lambda_s=None)
    runtime = IntegratedTrainingRuntime(config=config, host_runtime=host_runtime, host_module=host_runtime.module)
    optimizer = _build_optimizer(runtime)
    assert optimizer is not None

    output = runtime.training_step(_batch(), optimizer=optimizer)

    assert host_runtime.calls_host_loss_itself == 0
    assert host_runtime.calls_host_loss_clip == 1
    assert "host_loss_clip" in output.losses.active_loss_names
    assert output.s_proto is None
    assert torch.equal(output.s_total, output.s_host)


def test_stage1_prototype_only_objective_and_frozen_host() -> None:
    config = _build_config(train_mode="itself", stage="stage1")
    host_runtime = _DummyHostRuntime(train_mode="itself", lambda_s=0.5)
    runtime = IntegratedTrainingRuntime(config=config, host_runtime=host_runtime, host_module=host_runtime.module)
    optimizer = _build_optimizer(runtime)
    assert optimizer is not None

    host_before = host_runtime.module.weight.detach().clone()
    output = runtime.training_step(_batch(), optimizer=optimizer)
    host_after = host_runtime.module.weight.detach().clone()

    assert host_runtime.calls_host_loss_itself == 0
    assert output.losses.host_loss_itself is None
    assert "prototype_loss_ret" in output.losses.active_loss_names
    assert "prototype_loss_diag" in output.losses.active_loss_names
    assert "prototype_loss_div" in output.losses.active_loss_names
    assert tuple(group.group_name for group in output.optimizer_groups) == ("prototype",)
    assert torch.allclose(host_before, host_after)


def test_stage2_initialization_precedence_and_joint_loss_routing() -> None:
    raw_cfg = copy.deepcopy(DEFAULT_CONFIG)
    raw_cfg["train_mode"] = "clip"
    raw_cfg["host"]["lambda_s"] = None
    raw_cfg["prototype"]["enabled"] = True
    raw_cfg["prototype"]["dim"] = 8
    raw_cfg["prototype"]["num_prototypes"] = 5
    raw_cfg["prototype"]["regularization"]["diversity"]["weight"] = 1.0
    raw_cfg["prototype"]["regularization"]["balance"]["weight"] = 1.0
    raw_cfg["training"]["stage"] = "stage2"
    raw_cfg["training"]["calibration_only"] = False
    raw_cfg["training"]["freeze"]["host"] = False
    raw_cfg["training"]["freeze"]["prototype"] = False
    raw_cfg["training"]["freeze"]["prototype_policy"] = "train_all"
    raw_cfg["training"]["initialization"]["clip_backbone_source"] = "openai:ViT-B/16"
    raw_cfg["training"]["initialization"]["stage1_prototype_checkpoint"] = "stage1_proto.pt"
    raw_cfg["training"]["initialization"]["host_checkpoint"] = "host_stage0.pt"
    raw_cfg["training"]["initialization"]["host_checkpoint_compatible"] = True
    raw_cfg["loss"]["host_enabled"] = True
    raw_cfg["loss"]["prototype_ret_enabled"] = True
    raw_cfg["loss"]["prototype_diag_enabled"] = True
    raw_cfg["loss"]["prototype_div_enabled"] = True
    raw_cfg["loss"]["prototype_bal_enabled"] = True
    config = load_integration_config(raw_cfg)

    host_runtime = _DummyHostRuntime(train_mode="clip", lambda_s=None)
    calls: list[str] = []
    hooks = TrainingRuntimeHooks(
        clip_backbone_loader=lambda _: calls.append("clip_backbone"),
        stage1_prototype_loader=lambda _: calls.append("stage1_prototype"),
        host_checkpoint_loader=lambda _: calls.append("host_checkpoint"),
    )
    runtime = IntegratedTrainingRuntime(
        config=config,
        host_runtime=host_runtime,
        host_module=host_runtime.module,
        hooks=hooks,
    )
    optimizer = _build_optimizer(runtime)
    assert optimizer is not None

    output = runtime.training_step(_batch(), optimizer=optimizer)

    assert runtime.policy.stage == "stage2"
    assert runtime.policy.host_loss_enabled is True
    assert runtime.policy.prototype_enabled is True
    assert runtime.policy.calibration_only is False
    assert calls == ["clip_backbone", "stage1_prototype", "host_checkpoint"]
    assert output.initialized_stage2 is True
    assert "host_loss_clip" in output.losses.active_loss_names
    assert "prototype_loss_ret" in output.losses.active_loss_names
    assert "prototype_loss_diag" in output.losses.active_loss_names
    assert "prototype_loss_div" in output.losses.active_loss_names
    assert "prototype_loss_bal" in output.losses.active_loss_names
    assert tuple(group.group_name for group in output.optimizer_groups) == ("host", "prototype")


def test_stage3_calibration_only_blocks_representation_updates() -> None:
    config = _build_config(train_mode="itself", stage="stage3")
    host_runtime = _DummyHostRuntime(train_mode="itself", lambda_s=0.5)
    runtime = IntegratedTrainingRuntime(config=config, host_runtime=host_runtime, host_module=host_runtime.module)
    groups = runtime.build_optimizer_param_groups()
    assert groups == []

    output = runtime.training_step(_batch(), optimizer=None)
    assert output.losses.active_loss_names == ()
    assert output.losses.total_objective is None

    dummy_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=1e-2)
    with pytest.raises(RuntimeError):
        _ = runtime.training_step(_batch(), optimizer=dummy_optimizer)


@pytest.mark.parametrize(
    ("train_mode", "stage"),
    [
        ("itself", "stage0"),
        ("itself", "stage1"),
        ("itself", "stage2"),
        ("itself", "stage3"),
        ("clip", "stage0"),
        ("clip", "stage1"),
        ("clip", "stage2"),
        ("clip", "stage3"),
    ],
)
def test_smoke_training_step_for_all_supported_stage_mode_pairs(
    train_mode: TrainMode, stage: str
) -> None:
    config = _build_config(train_mode=train_mode, stage=stage)
    host_runtime = _DummyHostRuntime(train_mode=train_mode, lambda_s=0.5 if train_mode == "itself" else None)
    hooks = TrainingRuntimeHooks(
        clip_backbone_loader=(lambda _: None) if stage == "stage2" else None
    )
    runtime = IntegratedTrainingRuntime(
        config=config,
        host_runtime=host_runtime,
        host_module=host_runtime.module,
        hooks=hooks,
    )

    optimizer = _build_optimizer(runtime)
    output = runtime.training_step(_batch(), optimizer=optimizer)

    assert output.train_mode == train_mode
    assert output.stage == stage
    assert output.s_total.shape == output.s_host.shape
    if stage == "stage0":
        assert output.s_proto is None
    else:
        assert output.s_proto is not None
        assert output.s_proto.shape == output.s_host.shape
