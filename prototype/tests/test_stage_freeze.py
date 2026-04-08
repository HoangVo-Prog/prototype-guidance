"""Phase G freeze/update behavior verification tests."""

from __future__ import annotations

import torch

from prototype.integration.training_runtime import IntegratedTrainingRuntime

from ._runtime_test_utils import DummyHostRuntime, build_batch, build_config


def _build_optimizer(
    runtime: IntegratedTrainingRuntime,
) -> torch.optim.Optimizer | None:
    groups = runtime.build_optimizer_param_groups(host_lr=1e-2, prototype_lr=1e-2)
    if not groups:
        return None
    return torch.optim.SGD(groups, momentum=0.0)


def test_stage1_host_parameters_unchanged_after_optimization_step() -> None:
    config = build_config(train_mode="itself", stage="stage1")
    host_runtime = DummyHostRuntime(train_mode="itself", lambda_s=0.5)
    runtime = IntegratedTrainingRuntime(
        config=config,
        host_runtime=host_runtime,
        host_module=host_runtime.module,
    )
    optimizer = _build_optimizer(runtime)
    assert optimizer is not None

    host_before = host_runtime.module.weight.detach().clone()
    _ = runtime.training_step(build_batch(), optimizer=optimizer)
    host_after = host_runtime.module.weight.detach().clone()

    assert torch.allclose(host_before, host_after), "Stage1 must keep host params frozen"
    assert tuple(g.group_name for g in runtime.last_optimizer_groups) == ("prototype",)


def test_stage3_disables_representation_learning_optimizer_groups() -> None:
    config = build_config(train_mode="clip", stage="stage3")
    host_runtime = DummyHostRuntime(train_mode="clip", lambda_s=None)
    runtime = IntegratedTrainingRuntime(
        config=config,
        host_runtime=host_runtime,
        host_module=host_runtime.module,
    )

    groups = runtime.build_optimizer_param_groups(host_lr=1e-2, prototype_lr=1e-2)
    assert groups == []
    assert runtime.last_optimizer_groups == ()

    out = runtime.training_step(build_batch(), optimizer=None)
    assert out.losses.total_objective is None
    assert out.losses.active_loss_names == ()
