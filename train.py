"""Canonical user-facing launcher with dataloader/epoch/checkpoint training loop."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from torch.utils.data import DataLoader, Dataset

from prototype.config.schema import IntegrationConfig, load_integration_config
from prototype.integration import (
    IntegratedTrainingRuntime,
    SyntheticHostRuntime,
    SyntheticHostRuntimeConfig,
    TrainingRuntimeHooks,
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


def _str_value(root: Mapping[str, Any], key: str, default: str) -> str:
    value = root.get(key, default)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be str, got {type(value)!r}")
    stripped = value.strip()
    if stripped == "":
        raise ValueError(f"{key} must be non-empty")
    return stripped


def _optional_str_value(root: Mapping[str, Any], key: str) -> str | None:
    value = root.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be str or null, got {type(value)!r}")
    stripped = value.strip()
    if stripped == "":
        raise ValueError(f"{key} must be non-empty when provided")
    return stripped


def _bool_value(root: Mapping[str, Any], key: str, default: bool) -> bool:
    value = root.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be bool, got {type(value)!r}")
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


def _non_negative_int_value(root: Mapping[str, Any], key: str, default: int) -> int:
    value = root.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be int, got {type(value)!r}")
    if value < 0:
        raise ValueError(f"{key} must be >= 0")
    return value


def _image_shape_value(
    root: Mapping[str, Any], key: str, default: tuple[int, int, int]
) -> tuple[int, int, int]:
    value = root.get(key, default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{key} must be sequence[int] of length 3")
    if len(value) != 3:
        raise ValueError(f"{key} must have length 3, got {len(value)}")
    parsed: list[int] = []
    for idx, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{key}[{idx}] must be int, got {type(item)!r}")
        if item <= 0:
            raise ValueError(f"{key}[{idx}] must be > 0")
        parsed.append(item)
    return (parsed[0], parsed[1], parsed[2])


@dataclass(frozen=True)
class RuntimeBatchConfig:
    batch_size: int
    seq_len: int
    image_shape: tuple[int, int, int]


@dataclass(frozen=True)
class RuntimeDataConfig:
    dataset_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool
    drop_last: bool


@dataclass(frozen=True)
class RuntimeOptimizerConfig:
    host_lr: float
    prototype_lr: float


@dataclass(frozen=True)
class RuntimeCheckpointConfig:
    checkpoint_dir: Path
    save_every_epochs: int
    resume_from: Path | None


@dataclass(frozen=True)
class RuntimeOutputConfig:
    log_dir: Path


@dataclass(frozen=True)
class RuntimeLoopConfig:
    host_runtime_backend: str
    seed: int
    epochs: int
    batch: RuntimeBatchConfig
    data: RuntimeDataConfig
    optimizer: RuntimeOptimizerConfig
    checkpoint: RuntimeCheckpointConfig
    output: RuntimeOutputConfig
    synthetic_feature_dim: int


class SyntheticRetrievalDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic synthetic retrieval dataset for launcher training loops."""

    def __init__(
        self,
        *,
        dataset_size: int,
        seq_len: int,
        image_shape: tuple[int, int, int],
        seed: int,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be > 0")
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        self.dataset_size = dataset_size
        self.seq_len = seq_len
        self.image_shape = image_shape
        self.seed = seed

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.dataset_size:
            raise IndexError(index)
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(index))
        image = torch.randn(*self.image_shape, generator=generator, dtype=torch.float32)
        caption_ids = torch.zeros(self.seq_len, dtype=torch.long)
        caption_ids[index % self.seq_len] = 1
        token_mask = torch.ones(self.seq_len, dtype=torch.bool)
        return {
            "images": image,
            "caption_ids": caption_ids,
            "token_mask": token_mask,
        }


def _collate_synthetic_batch(
    samples: Sequence[Mapping[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    if len(samples) == 0:
        raise ValueError("Cannot collate an empty batch")
    return {
        "images": torch.stack([sample["images"] for sample in samples], dim=0),
        "caption_ids": torch.stack([sample["caption_ids"] for sample in samples], dim=0),
        "token_mask": torch.stack([sample["token_mask"] for sample in samples], dim=0),
    }


class _RunLogger:
    """Console/file logger for launcher runtime loop."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "train.log"
        self._handle = self.log_path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message)
        self._handle.write(message + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def _build_hooks() -> TrainingRuntimeHooks:
    # Stage2 requires explicit initialization precedence callables.
    return TrainingRuntimeHooks(
        clip_backbone_loader=lambda _: None,
        stage1_prototype_loader=lambda _: None,
        host_checkpoint_loader=lambda _: None,
    )


def _parse_loop_config(
    config: IntegrationConfig,
    raw_cfg: Mapping[str, Any],
) -> RuntimeLoopConfig:
    runtime_cfg = _as_mapping(raw_cfg.get("runtime"), name="runtime")
    backend = _str_value(runtime_cfg, "runtime.host_runtime", "synthetic").lower()
    epochs_default = runtime_cfg.get("steps", 1)
    epochs = _int_value(runtime_cfg, "epochs", default=int(epochs_default))
    seed = _non_negative_int_value(runtime_cfg, "seed", default=1337)

    batch_cfg = _as_mapping(runtime_cfg.get("batch"), name="runtime.batch")
    batch = RuntimeBatchConfig(
        batch_size=_int_value(batch_cfg, "runtime.batch.batch_size", default=4),
        seq_len=_int_value(batch_cfg, "runtime.batch.seq_len", default=6),
        image_shape=_image_shape_value(
            batch_cfg, "runtime.batch.image_shape", default=(3, 32, 16)
        ),
    )

    data_cfg = _as_mapping(runtime_cfg.get("data"), name="runtime.data")
    data = RuntimeDataConfig(
        dataset_size=_int_value(
            data_cfg,
            "runtime.data.dataset_size",
            default=max(batch.batch_size * 4, batch.batch_size),
        ),
        shuffle=_bool_value(data_cfg, "runtime.data.shuffle", default=True),
        num_workers=_non_negative_int_value(data_cfg, "runtime.data.num_workers", default=0),
        pin_memory=_bool_value(data_cfg, "runtime.data.pin_memory", default=False),
        drop_last=_bool_value(data_cfg, "runtime.data.drop_last", default=False),
    )

    optimizer_cfg = _as_mapping(runtime_cfg.get("optimizer"), name="runtime.optimizer")
    optimizer = RuntimeOptimizerConfig(
        host_lr=_float_value(optimizer_cfg, "runtime.optimizer.host_lr", default=1e-4),
        prototype_lr=_float_value(
            optimizer_cfg, "runtime.optimizer.prototype_lr", default=1e-4
        ),
    )

    output_cfg = _as_mapping(runtime_cfg.get("output", {}), name="runtime.output")
    checkpoint_cfg = _as_mapping(
        runtime_cfg.get("checkpoint", {}), name="runtime.checkpoint"
    )
    checkpoint_dir_str = (
        checkpoint_cfg.get("checkpoint_dir")
        or output_cfg.get("checkpoint_dir")
        or "outputs/checkpoints"
    )
    if not isinstance(checkpoint_dir_str, str):
        raise TypeError("runtime.checkpoint.checkpoint_dir must be str")
    resume_from_value = _optional_str_value(
        checkpoint_cfg, "runtime.checkpoint.resume_from"
    )
    checkpoint = RuntimeCheckpointConfig(
        checkpoint_dir=Path(checkpoint_dir_str),
        save_every_epochs=_int_value(
            checkpoint_cfg, "runtime.checkpoint.save_every_epochs", default=1
        ),
        resume_from=Path(resume_from_value) if resume_from_value else None,
    )
    output = RuntimeOutputConfig(
        log_dir=Path(_str_value(output_cfg, "runtime.output.log_dir", "outputs/logs"))
    )

    synthetic_cfg = _as_mapping(runtime_cfg.get("synthetic"), name="runtime.synthetic")
    feature_dim = _int_value(
        synthetic_cfg,
        "runtime.synthetic.feature_dim",
        default=int(config.prototype.dim),
    )
    return RuntimeLoopConfig(
        host_runtime_backend=backend,
        seed=seed,
        epochs=epochs,
        batch=batch,
        data=data,
        optimizer=optimizer,
        checkpoint=checkpoint,
        output=output,
        synthetic_feature_dim=feature_dim,
    )


def _build_runtime(
    config: IntegrationConfig,
    loop_cfg: RuntimeLoopConfig,
) -> IntegratedTrainingRuntime:
    if loop_cfg.host_runtime_backend != "synthetic":
        raise ValueError(
            "Only runtime.host_runtime='synthetic' is currently supported by canonical train.py"
        )
    host_runtime = SyntheticHostRuntime(
        SyntheticHostRuntimeConfig(
            train_mode=config.train_mode,
            lambda_s=config.host.lambda_s,
            feature_dim=loop_cfg.synthetic_feature_dim,
        )
    )
    return IntegratedTrainingRuntime(
        config=config,
        host_runtime=host_runtime,
        host_module=host_runtime.module,
        hooks=_build_hooks(),
    )


def _optimizer_for_runtime(
    runtime: IntegratedTrainingRuntime,
    loop_cfg: RuntimeLoopConfig,
) -> torch.optim.Optimizer | None:
    groups = runtime.build_optimizer_param_groups(
        host_lr=loop_cfg.optimizer.host_lr,
        prototype_lr=loop_cfg.optimizer.prototype_lr,
    )
    if not groups:
        return None
    return torch.optim.SGD(groups, momentum=0.0)


def _build_dataloader(loop_cfg: RuntimeLoopConfig) -> DataLoader[dict[str, torch.Tensor]]:
    dataset = SyntheticRetrievalDataset(
        dataset_size=loop_cfg.data.dataset_size,
        seq_len=loop_cfg.batch.seq_len,
        image_shape=loop_cfg.batch.image_shape,
        seed=loop_cfg.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=loop_cfg.batch.batch_size,
        shuffle=loop_cfg.data.shuffle,
        num_workers=loop_cfg.data.num_workers,
        pin_memory=loop_cfg.data.pin_memory,
        drop_last=loop_cfg.data.drop_last,
        collate_fn=_collate_synthetic_batch,
    )
    if len(dataloader) == 0:
        raise ValueError(
            "DataLoader produced zero batches; adjust runtime.data.dataset_size, "
            "runtime.batch.batch_size, or runtime.data.drop_last."
        )
    return dataloader


def _save_checkpoint(
    *,
    runtime: IntegratedTrainingRuntime,
    optimizer: torch.optim.Optimizer | None,
    loop_cfg: RuntimeLoopConfig,
    raw_cfg: Mapping[str, Any],
    config_path: Path,
    epoch: int,
    global_step: int,
    logger: _RunLogger,
) -> None:
    loop_cfg.checkpoint.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "train_mode": runtime.config.train_mode,
        "training_stage": runtime.policy.stage,
        "host_state_dict": runtime.host_module.state_dict(),
        "prototype_state_dict": runtime.prototype_runtime.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "stage2_initialized": runtime.policy.stage == "stage2" and runtime._stage2_initialized,
        "config_path": str(config_path),
        "config_snapshot": dict(raw_cfg),
    }
    epoch_ckpt = loop_cfg.checkpoint.checkpoint_dir / f"epoch_{epoch:04d}.pt"
    latest_ckpt = loop_cfg.checkpoint.checkpoint_dir / "latest.pt"
    torch.save(payload, epoch_ckpt)
    torch.save(payload, latest_ckpt)
    logger.log(f"checkpoint saved: {epoch_ckpt}")


def _load_checkpoint_if_requested(
    *,
    runtime: IntegratedTrainingRuntime,
    optimizer: torch.optim.Optimizer | None,
    loop_cfg: RuntimeLoopConfig,
    logger: _RunLogger,
) -> tuple[int, int]:
    resume_path = loop_cfg.checkpoint.resume_from
    if resume_path is None:
        return 1, 0
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint resume file not found: {resume_path}")
    payload = torch.load(resume_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a mapping")
    resume_mode = payload.get("train_mode")
    resume_stage = payload.get("training_stage")
    if resume_mode != runtime.config.train_mode or resume_stage != runtime.policy.stage:
        raise ValueError(
            "Checkpoint mode/stage mismatch: "
            f"checkpoint=({resume_mode!r}, {resume_stage!r}) "
            f"runtime=({runtime.config.train_mode!r}, {runtime.policy.stage!r})"
        )

    host_state = payload.get("host_state_dict")
    prototype_state = payload.get("prototype_state_dict")
    if not isinstance(host_state, Mapping) or not isinstance(prototype_state, Mapping):
        raise TypeError("Checkpoint must contain host_state_dict and prototype_state_dict mappings")
    runtime.host_module.load_state_dict(host_state)
    runtime.prototype_runtime.load_state_dict(prototype_state)

    optim_state = payload.get("optimizer_state_dict")
    if optimizer is not None and optim_state is not None:
        if not isinstance(optim_state, Mapping):
            raise TypeError("optimizer_state_dict must be mapping when provided")
        optimizer.load_state_dict(optim_state)

    if runtime.policy.stage == "stage2" and bool(payload.get("stage2_initialized", False)):
        runtime._stage2_initialized = True

    start_epoch = int(payload.get("epoch", 0)) + 1
    global_step = int(payload.get("global_step", 0))
    logger.log(
        f"resumed from checkpoint={resume_path} "
        f"at epoch={start_epoch} global_step={global_step}"
    )
    return start_epoch, global_step


def _format_objective(value: torch.Tensor | None) -> str:
    if value is None:
        return "none"
    return f"{float(value.detach().cpu()):.6f}"


def _run_epoch_training_loop(
    *,
    runtime: IntegratedTrainingRuntime,
    loop_cfg: RuntimeLoopConfig,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    raw_cfg: Mapping[str, Any],
    config_path: Path,
) -> None:
    logger = _RunLogger(loop_cfg.output.log_dir)
    try:
        start_epoch, global_step = _load_checkpoint_if_requested(
            runtime=runtime,
            optimizer=optimizer,
            loop_cfg=loop_cfg,
            logger=logger,
        )
        logger.log(
            "launching training loop with "
            f"mode={runtime.config.train_mode} stage={runtime.policy.stage} "
            f"epochs={loop_cfg.epochs} batches_per_epoch={len(dataloader)} "
            f"batch_size={loop_cfg.batch.batch_size}"
        )
        logger.log(
            "runtime summary: "
            + json.dumps(
                {
                    "checkpoint_dir": str(loop_cfg.checkpoint.checkpoint_dir),
                    "log_dir": str(loop_cfg.output.log_dir),
                    "save_every_epochs": loop_cfg.checkpoint.save_every_epochs,
                    "seed": loop_cfg.seed,
                }
            )
        )

        if start_epoch > loop_cfg.epochs:
            logger.log(
                f"nothing to run: resumed start_epoch={start_epoch} exceeds configured epochs={loop_cfg.epochs}"
            )
            return

        for epoch in range(start_epoch, loop_cfg.epochs + 1):
            epoch_objectives: list[float] = []
            for batch_idx, batch in enumerate(dataloader, start=1):
                output = runtime.training_step(batch=batch, optimizer=optimizer)
                global_step += 1
                if output.losses.total_objective is not None:
                    epoch_objectives.append(float(output.losses.total_objective.detach().cpu()))
                logger.log(
                    f"[{output.stage}] epoch={epoch}/{loop_cfg.epochs} "
                    f"batch={batch_idx}/{len(dataloader)} global_step={global_step} "
                    f"mode={output.train_mode} s_total_shape={tuple(output.s_total.shape)} "
                    f"active_losses={output.losses.active_loss_names} "
                    f"objective={_format_objective(output.losses.total_objective)}"
                )

            mean_objective = (
                sum(epoch_objectives) / len(epoch_objectives) if epoch_objectives else float("nan")
            )
            logger.log(
                f"epoch={epoch} completed "
                f"mean_objective={mean_objective:.6f} "
                f"updates={len(epoch_objectives)}"
            )

            if epoch % loop_cfg.checkpoint.save_every_epochs == 0:
                _save_checkpoint(
                    runtime=runtime,
                    optimizer=optimizer,
                    loop_cfg=loop_cfg,
                    raw_cfg=raw_cfg,
                    config_path=config_path,
                    epoch=epoch,
                    global_step=global_step,
                    logger=logger,
                )
    finally:
        logger.close()


def main() -> int:
    args = _parse_args()
    raw_cfg = _load_yaml(args.config)
    config = load_integration_config(raw_cfg)
    loop_cfg = _parse_loop_config(config=config, raw_cfg=raw_cfg)
    torch.manual_seed(loop_cfg.seed)
    runtime = _build_runtime(config=config, loop_cfg=loop_cfg)
    dataloader = _build_dataloader(loop_cfg)
    optimizer = _optimizer_for_runtime(runtime=runtime, loop_cfg=loop_cfg)
    _run_epoch_training_loop(
        runtime=runtime,
        loop_cfg=loop_cfg,
        dataloader=dataloader,
        optimizer=optimizer,
        raw_cfg=raw_cfg,
        config_path=args.config,
    )
    print("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
