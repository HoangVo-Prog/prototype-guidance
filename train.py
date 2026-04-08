"""Canonical user-facing launcher for project-side staged runtime execution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'pyyaml'. Install it and retry: pip install pyyaml"
    ) from exc

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'torch'. Install it and retry before running train.py."
    ) from exc

from prototype.config.schema import IntegrationConfig, load_integration_config
from prototype.integration import (
    IntegratedTrainingRuntime,
    SyntheticHostRuntime,
    SyntheticHostRuntimeConfig,
    TrainingRuntimeHooks,
    build_synthetic_batch,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype integration train launcher")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Config root must be a mapping, got {type(payload)!r}")
    return payload


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value)!r}")
    return value


def _float_value(root: Mapping[str, Any], key: str, default: float) -> float:
    value = root.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be float-compatible, got {type(value)!r}")
    return float(value)


def _int_value(root: Mapping[str, Any], key: str, default: int) -> int:
    value = root.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be int, got {type(value)!r}")
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def _build_hooks() -> TrainingRuntimeHooks:
    # Stage2 requires explicit initialization precedence callables.
    return TrainingRuntimeHooks(
        clip_backbone_loader=lambda _: None,
        stage1_prototype_loader=lambda _: None,
        host_checkpoint_loader=lambda _: None,
    )


def _build_runtime(
    config: IntegrationConfig,
    raw_cfg: Mapping[str, Any],
) -> tuple[IntegratedTrainingRuntime, Mapping[str, Any]]:
    runtime_cfg = _as_mapping(raw_cfg.get("runtime"), name="runtime")
    backend = str(runtime_cfg.get("host_runtime", "synthetic")).strip().lower()
    if backend != "synthetic":
        raise ValueError(
            "Only runtime.host_runtime='synthetic' is currently supported by canonical train.py"
        )

    synthetic_cfg = _as_mapping(runtime_cfg.get("synthetic"), name="runtime.synthetic")
    feature_dim = _int_value(
        synthetic_cfg,
        "feature_dim",
        default=int(config.prototype.dim),
    )
    host_runtime = SyntheticHostRuntime(
        SyntheticHostRuntimeConfig(
            train_mode=config.train_mode,
            lambda_s=config.host.lambda_s,
            feature_dim=feature_dim,
        )
    )
    runtime = IntegratedTrainingRuntime(
        config=config,
        host_runtime=host_runtime,
        host_module=host_runtime.module,
        hooks=_build_hooks(),
    )
    return runtime, runtime_cfg


def _optimizer_for_runtime(
    runtime: IntegratedTrainingRuntime,
    runtime_cfg: Mapping[str, Any],
) -> torch.optim.Optimizer | None:
    optim_cfg = _as_mapping(runtime_cfg.get("optimizer"), name="runtime.optimizer")
    host_lr = _float_value(optim_cfg, "host_lr", default=1e-4)
    proto_lr = _float_value(optim_cfg, "prototype_lr", default=1e-4)
    groups = runtime.build_optimizer_param_groups(host_lr=host_lr, prototype_lr=proto_lr)
    if not groups:
        return None
    return torch.optim.SGD(groups, momentum=0.0)


def _run_steps(
    runtime: IntegratedTrainingRuntime,
    runtime_cfg: Mapping[str, Any],
    *,
    stage_name: str,
) -> None:
    batch_cfg = _as_mapping(runtime_cfg.get("batch"), name="runtime.batch")
    batch_size = _int_value(batch_cfg, "batch_size", default=4)
    seq_len = _int_value(batch_cfg, "seq_len", default=6)
    steps = _int_value(runtime_cfg, "steps", default=1)

    optimizer = _optimizer_for_runtime(runtime, runtime_cfg)
    for step_idx in range(steps):
        batch = build_synthetic_batch(batch_size=batch_size, seq_len=seq_len)
        output = runtime.training_step(batch=batch, optimizer=optimizer)
        print(
            f"[{stage_name}] step={step_idx + 1}/{steps} "
            f"mode={output.train_mode} stage={output.stage} "
            f"s_total_shape={tuple(output.s_total.shape)} "
            f"active_losses={output.losses.active_loss_names}"
        )


def _run_stage(runtime: IntegratedTrainingRuntime, runtime_cfg: Mapping[str, Any]) -> None:
    stage = runtime.policy.stage
    if stage == "stage0":
        _run_steps(runtime, runtime_cfg, stage_name="stage0")
        return
    if stage == "stage1":
        _run_steps(runtime, runtime_cfg, stage_name="stage1")
        return
    if stage == "stage2":
        _run_steps(runtime, runtime_cfg, stage_name="stage2")
        return
    if stage == "stage3":
        _run_steps(runtime, runtime_cfg, stage_name="stage3")
        return
    raise RuntimeError(f"Unsupported stage from validated config: {stage!r}")


def main() -> int:
    args = _parse_args()
    raw_cfg = _load_yaml(args.config)
    config = load_integration_config(raw_cfg)
    runtime, runtime_cfg = _build_runtime(config=config, raw_cfg=raw_cfg)

    print(
        "Launching canonical train runtime with "
        f"train_mode={config.train_mode}, training.stage={config.training.stage}"
    )
    _run_stage(runtime=runtime, runtime_cfg=runtime_cfg)
    print("Run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
