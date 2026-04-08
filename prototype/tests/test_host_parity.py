"""Phase G host-parity verification for mode-bound host score surfaces."""

from __future__ import annotations

import torch

from prototype.integration.training_runtime import IntegratedTrainingRuntime

from ._runtime_test_utils import DummyHostRuntime, build_batch, build_config


def test_host_parity_itself_mode_with_prototype_disabled() -> None:
    config = build_config(train_mode="itself", stage="stage0")
    host_runtime = DummyHostRuntime(train_mode="itself", lambda_s=0.5)
    runtime = IntegratedTrainingRuntime(
        config=config,
        host_runtime=host_runtime,
        host_module=host_runtime.module,
    )

    out = runtime.training_step(build_batch(), optimizer=None)
    assert out.s_proto is None
    assert torch.equal(out.s_total, out.s_host)


def test_host_parity_clip_mode_with_prototype_disabled() -> None:
    config = build_config(train_mode="clip", stage="stage0")
    host_runtime = DummyHostRuntime(train_mode="clip", lambda_s=None)
    runtime = IntegratedTrainingRuntime(
        config=config,
        host_runtime=host_runtime,
        host_module=host_runtime.module,
    )

    out = runtime.training_step(build_batch(), optimizer=None)
    assert out.s_proto is None
    assert torch.equal(out.s_total, out.s_host)


def test_host_parity_itself_mode_with_lambda_f_zero() -> None:
    config = build_config(train_mode="itself", stage="stage1")
    host_runtime = DummyHostRuntime(train_mode="itself", lambda_s=0.5)
    runtime = IntegratedTrainingRuntime(
        config=config,
        host_runtime=host_runtime,
        host_module=host_runtime.module,
    )
    out = runtime.training_step(build_batch(), optimizer=None)
    assert out.s_proto is not None
    assert torch.equal(out.s_total, out.s_host)


def test_host_parity_clip_mode_with_lambda_f_zero_no_grab_assumption() -> None:
    config = build_config(train_mode="clip", stage="stage1")
    host_runtime = DummyHostRuntime(train_mode="clip", lambda_s=None)
    runtime = IntegratedTrainingRuntime(
        config=config,
        host_runtime=host_runtime,
        host_module=host_runtime.module,
    )
    features = host_runtime.extract_feature_surface(build_batch())
    assert not hasattr(features, "v_i_local")
    assert not hasattr(features, "t_j_local")

    out = runtime.training_step(build_batch(), optimizer=None)
    assert out.s_proto is not None
    assert torch.equal(out.s_total, out.s_host)
