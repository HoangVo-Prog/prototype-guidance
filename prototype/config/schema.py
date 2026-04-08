"""Typed config schema and validation for stage/mode control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


TrainMode = Literal["itself", "clip"]
TrainingStage = Literal["stage0", "stage1", "stage2", "stage3"]
PrototypePolicy = Literal["train_all", "freeze_all", "partial"]
ContextualizationType = Literal["none", "self_attention"]

SUPPORTED_STAGE_MODE_MATRIX: frozenset[tuple[TrainMode, TrainingStage]] = frozenset(
    {
        ("itself", "stage0"),
        ("itself", "stage1"),
        ("itself", "stage2"),
        ("itself", "stage3"),
        ("clip", "stage0"),
        ("clip", "stage1"),
        ("clip", "stage2"),
        ("clip", "stage3"),
    }
)


def _ensure_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping/dict, got {type(value)!r}")
    return value


def _ensure_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool, got {type(value)!r}")
    return value


def _ensure_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value)!r}")
    return value


def _ensure_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be float-compatible, got {type(value)!r}")
    return float(value)


def _ensure_str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value)!r}")
    if value.strip() == "":
        raise ValueError(f"{name} must be non-empty")
    return value


def _optional_str(name: str, value: Any) -> str | None:
    if value is None:
        return None
    return _ensure_str(name, value)


def _ensure_str_list(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be list[str], got {type(value)!r}")
    items: list[str] = []
    for idx, item in enumerate(value):
        items.append(_ensure_str(f"{name}[{idx}]", item))
    return tuple(items)


@dataclass(frozen=True)
class PrototypeTemperatureConfig:
    """Prototype temperature config."""

    routing: float
    basis: float
    teacher: float
    retrieval: float

    def __post_init__(self) -> None:
        if self.routing <= 0:
            raise ValueError("prototype.temperatures.routing must be > 0")
        if self.basis <= 0:
            raise ValueError("prototype.temperatures.basis must be > 0")
        if self.teacher <= 0:
            raise ValueError("prototype.temperatures.teacher must be > 0")
        if self.retrieval <= 0:
            raise ValueError("prototype.temperatures.retrieval must be > 0")


@dataclass(frozen=True)
class RegularizationTermConfig:
    """Single regularization term config."""

    enabled: bool
    weight: float

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("regularization weight must be >= 0")


@dataclass(frozen=True)
class PrototypeRegularizationConfig:
    """Prototype regularization config."""

    diversity: RegularizationTermConfig
    balance: RegularizationTermConfig


@dataclass(frozen=True)
class PrototypeContextualizationConfig:
    """Prototype contextualization config."""

    enabled: bool
    type: ContextualizationType


@dataclass(frozen=True)
class PrototypeConfig:
    """Prototype-branch config."""

    enabled: bool
    num_prototypes: int
    dim: int
    contextualization: PrototypeContextualizationConfig
    temperatures: PrototypeTemperatureConfig
    regularization: PrototypeRegularizationConfig

    def __post_init__(self) -> None:
        if self.num_prototypes <= 0:
            raise ValueError("prototype.num_prototypes must be > 0")
        if self.dim <= 0:
            raise ValueError("prototype.dim must be > 0")


@dataclass(frozen=True)
class FusionConfig:
    """Fusion config."""

    enabled: bool
    lambda_f: float


@dataclass(frozen=True)
class HostConfig:
    """Host config."""

    lambda_s: float | None


@dataclass(frozen=True)
class FreezePolicyConfig:
    """Freeze-policy intent config."""

    host: bool
    prototype: bool
    host_allowlist: tuple[str, ...]
    prototype_policy: PrototypePolicy


@dataclass(frozen=True)
class InitializationConfig:
    """Initialization intent config."""

    clip_backbone_source: str | None
    stage1_prototype_checkpoint: str | None
    host_checkpoint: str | None
    host_checkpoint_compatible: bool


@dataclass(frozen=True)
class TrainingConfig:
    """Training-stage control config."""

    stage: TrainingStage
    calibration_only: bool
    freeze: FreezePolicyConfig
    initialization: InitializationConfig


@dataclass(frozen=True)
class LossIntentConfig:
    """Loss intent flags for stage-control layer."""

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
    loss: LossIntentConfig


def _parse_train_mode(value: Any) -> TrainMode:
    mode = _ensure_str("train_mode", value)
    if mode not in ("itself", "clip"):
        raise ValueError(f"train_mode must be one of ['itself', 'clip'], got {mode!r}")
    return mode


def _parse_stage(value: Any) -> TrainingStage:
    stage = _ensure_str("training.stage", value)
    if stage not in ("stage0", "stage1", "stage2", "stage3"):
        raise ValueError(
            "training.stage must be one of ['stage0', 'stage1', 'stage2', 'stage3'], "
            f"got {stage!r}"
        )
    return stage


def load_integration_config(data: Mapping[str, Any]) -> IntegrationConfig:
    """Parse raw dict config into typed config and validate semantic rules."""
    root = _ensure_mapping("config", data)
    train_mode = _parse_train_mode(root["train_mode"])

    prototype_raw = _ensure_mapping("prototype", root["prototype"])
    contextualization_raw = _ensure_mapping(
        "prototype.contextualization", prototype_raw["contextualization"]
    )
    contextualization_type = _ensure_str(
        "prototype.contextualization.type", contextualization_raw["type"]
    )
    if contextualization_type not in ("none", "self_attention"):
        raise ValueError(
            "prototype.contextualization.type must be one of ['none', 'self_attention']"
        )
    temperatures_raw = _ensure_mapping(
        "prototype.temperatures", prototype_raw["temperatures"]
    )
    regularization_raw = _ensure_mapping(
        "prototype.regularization", prototype_raw["regularization"]
    )
    diversity_raw = _ensure_mapping(
        "prototype.regularization.diversity", regularization_raw["diversity"]
    )
    balance_raw = _ensure_mapping(
        "prototype.regularization.balance", regularization_raw["balance"]
    )

    prototype = PrototypeConfig(
        enabled=_ensure_bool("prototype.enabled", prototype_raw["enabled"]),
        num_prototypes=_ensure_int(
            "prototype.num_prototypes", prototype_raw["num_prototypes"]
        ),
        dim=_ensure_int("prototype.dim", prototype_raw["dim"]),
        contextualization=PrototypeContextualizationConfig(
            enabled=_ensure_bool(
                "prototype.contextualization.enabled", contextualization_raw["enabled"]
            ),
            type=contextualization_type,  # type: ignore[arg-type]
        ),
        temperatures=PrototypeTemperatureConfig(
            routing=_ensure_float(
                "prototype.temperatures.routing", temperatures_raw["routing"]
            ),
            basis=_ensure_float(
                "prototype.temperatures.basis", temperatures_raw["basis"]
            ),
            teacher=_ensure_float(
                "prototype.temperatures.teacher", temperatures_raw["teacher"]
            ),
            retrieval=_ensure_float(
                "prototype.temperatures.retrieval", temperatures_raw["retrieval"]
            ),
        ),
        regularization=PrototypeRegularizationConfig(
            diversity=RegularizationTermConfig(
                enabled=_ensure_bool(
                    "prototype.regularization.diversity.enabled",
                    diversity_raw["enabled"],
                ),
                weight=_ensure_float(
                    "prototype.regularization.diversity.weight",
                    diversity_raw["weight"],
                ),
            ),
            balance=RegularizationTermConfig(
                enabled=_ensure_bool(
                    "prototype.regularization.balance.enabled", balance_raw["enabled"]
                ),
                weight=_ensure_float(
                    "prototype.regularization.balance.weight", balance_raw["weight"]
                ),
            ),
        ),
    )

    fusion_raw = _ensure_mapping("fusion", root["fusion"])
    fusion = FusionConfig(
        enabled=_ensure_bool("fusion.enabled", fusion_raw["enabled"]),
        lambda_f=_ensure_float("fusion.lambda_f", fusion_raw["lambda_f"]),
    )

    host_raw = _ensure_mapping("host", root["host"])
    host_lambda_s = host_raw.get("lambda_s")
    if host_lambda_s is None:
        lambda_s: float | None = None
    else:
        lambda_s = _ensure_float("host.lambda_s", host_lambda_s)
    host = HostConfig(lambda_s=lambda_s)

    training_raw = _ensure_mapping("training", root["training"])
    freeze_raw = _ensure_mapping("training.freeze", training_raw["freeze"])
    init_raw = _ensure_mapping("training.initialization", training_raw["initialization"])
    prototype_policy = _ensure_str(
        "training.freeze.prototype_policy", freeze_raw["prototype_policy"]
    )
    if prototype_policy not in ("train_all", "freeze_all", "partial"):
        raise ValueError(
            "training.freeze.prototype_policy must be one of "
            "['train_all', 'freeze_all', 'partial']"
        )
    training = TrainingConfig(
        stage=_parse_stage(training_raw["stage"]),
        calibration_only=_ensure_bool(
            "training.calibration_only", training_raw["calibration_only"]
        ),
        freeze=FreezePolicyConfig(
            host=_ensure_bool("training.freeze.host", freeze_raw["host"]),
            prototype=_ensure_bool(
                "training.freeze.prototype", freeze_raw["prototype"]
            ),
            host_allowlist=_ensure_str_list(
                "training.freeze.host_allowlist", freeze_raw["host_allowlist"]
            ),
            prototype_policy=prototype_policy,  # type: ignore[arg-type]
        ),
        initialization=InitializationConfig(
            clip_backbone_source=_optional_str(
                "training.initialization.clip_backbone_source",
                init_raw.get("clip_backbone_source"),
            ),
            stage1_prototype_checkpoint=_optional_str(
                "training.initialization.stage1_prototype_checkpoint",
                init_raw.get("stage1_prototype_checkpoint"),
            ),
            host_checkpoint=_optional_str(
                "training.initialization.host_checkpoint", init_raw.get("host_checkpoint")
            ),
            host_checkpoint_compatible=_ensure_bool(
                "training.initialization.host_checkpoint_compatible",
                init_raw["host_checkpoint_compatible"],
            ),
        ),
    )

    loss_raw = _ensure_mapping("loss", root["loss"])
    loss = LossIntentConfig(
        host_enabled=_ensure_bool("loss.host_enabled", loss_raw["host_enabled"]),
        prototype_ret_enabled=_ensure_bool(
            "loss.prototype_ret_enabled", loss_raw["prototype_ret_enabled"]
        ),
        prototype_diag_enabled=_ensure_bool(
            "loss.prototype_diag_enabled", loss_raw["prototype_diag_enabled"]
        ),
        prototype_div_enabled=_ensure_bool(
            "loss.prototype_div_enabled", loss_raw["prototype_div_enabled"]
        ),
        prototype_bal_enabled=_ensure_bool(
            "loss.prototype_bal_enabled", loss_raw["prototype_bal_enabled"]
        ),
    )

    cfg = IntegrationConfig(
        train_mode=train_mode,
        prototype=prototype,
        fusion=fusion,
        host=host,
        training=training,
        loss=loss,
    )
    validate_integration_config(cfg)
    return cfg


def validate_integration_config(config: IntegrationConfig) -> None:
    """Validate config contract and stage/mode guards."""
    stage_mode = (config.train_mode, config.training.stage)
    if stage_mode not in SUPPORTED_STAGE_MODE_MATRIX:
        raise ValueError(
            "Unsupported (train_mode, stage) pair: "
            f"{stage_mode!r}. Supported pairs: {sorted(SUPPORTED_STAGE_MODE_MATRIX)!r}"
        )

    if config.train_mode == "itself":
        if config.host.lambda_s is None:
            raise ValueError(
                "host.lambda_s is required for train_mode='itself' to define canonical s_host^itself"
            )
        if not 0.0 <= config.host.lambda_s <= 1.0:
            raise ValueError("host.lambda_s must be in [0, 1] for train_mode='itself'")
    else:
        if config.host.lambda_s is not None:
            raise ValueError(
                "host.lambda_s must be null in train_mode='clip'; clip host score must not be defined via lambda_s"
            )

    if config.prototype.enabled and not config.fusion.enabled:
        raise ValueError(
            "fusion.enabled must be true when prototype.enabled is true"
        )

    stage = config.training.stage
    if stage == "stage0":
        if config.prototype.enabled:
            raise ValueError("stage0 requires prototype.enabled=false (host-only baseline)")
        if config.loss.prototype_ret_enabled or config.loss.prototype_diag_enabled or config.loss.prototype_div_enabled or config.loss.prototype_bal_enabled:
            raise ValueError("stage0 must not enable prototype-side losses")
        if not config.loss.host_enabled:
            raise ValueError("stage0 requires host loss intent enabled")
        if config.training.freeze.host:
            raise ValueError("stage0 must not set training.freeze.host=true")
        if config.training.calibration_only:
            raise ValueError("stage0 must not set training.calibration_only=true")

    elif stage == "stage1":
        if not config.prototype.enabled:
            raise ValueError("stage1 requires prototype.enabled=true")
        if not config.training.freeze.host:
            raise ValueError("stage1 requires training.freeze.host=true")
        if config.loss.host_enabled:
            raise ValueError(
                "stage1 requires loss.host_enabled=false (host forward is feature-extraction-only)"
            )
        if not (config.loss.prototype_ret_enabled and config.loss.prototype_diag_enabled and config.loss.prototype_div_enabled):
            raise ValueError(
                "stage1 requires prototype loss intents enabled for ret/diag/div"
            )
        if config.training.calibration_only:
            raise ValueError("stage1 must not set training.calibration_only=true")

    elif stage == "stage2":
        if not config.prototype.enabled:
            raise ValueError("stage2 requires prototype.enabled=true")
        if config.training.freeze.host:
            raise ValueError("stage2 requires training.freeze.host=false (unfrozen host intent)")
        if not config.loss.host_enabled:
            raise ValueError("stage2 requires host loss intent enabled")
        if config.training.initialization.clip_backbone_source is None:
            raise ValueError(
                "stage2 requires training.initialization.clip_backbone_source for CLIP-init-compatible retraining path"
            )
        if config.training.calibration_only:
            raise ValueError("stage2 must not set training.calibration_only=true")

    elif stage == "stage3":
        if not config.prototype.enabled:
            raise ValueError("stage3 requires prototype.enabled=true for calibration on fused scores")
        if not config.fusion.enabled:
            raise ValueError("stage3 requires fusion.enabled=true")
        if not config.training.calibration_only:
            raise ValueError("stage3 requires training.calibration_only=true")
        if not config.training.freeze.host or not config.training.freeze.prototype:
            raise ValueError(
                "stage3 requires training.freeze.host=true and training.freeze.prototype=true"
            )
        if config.loss.host_enabled:
            raise ValueError("stage3 requires loss.host_enabled=false")
        if config.loss.prototype_ret_enabled or config.loss.prototype_diag_enabled or config.loss.prototype_div_enabled or config.loss.prototype_bal_enabled:
            raise ValueError("stage3 must disable prototype representation-learning loss intents")

    if (
        config.training.initialization.host_checkpoint is not None
        and not config.training.initialization.host_checkpoint_compatible
    ):
        raise ValueError(
            "training.initialization.host_checkpoint requires host_checkpoint_compatible=true"
        )
