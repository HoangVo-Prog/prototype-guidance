"""Phase F training runtime: executable stage/mode behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

import torch
import torch.nn.functional as F
from torch import nn

from prototype.config.schema import IntegrationConfig, SUPPORTED_STAGE_MODE_MATRIX, TrainMode
from prototype.integration.feature_surface import (
    CLIPHostFeatureSurface,
    CLIPHostScoreSurface,
    HostFeatureSurface,
    HostScoreSurface,
    ITSELFHostFeatureSurface,
    ITSELFHostScoreSurface,
)
from prototype.prototype_branch import (
    PrototypeBasisBuilder,
    PrototypeBasisBuilderConfig,
    PrototypeBankConfig,
    PrototypeBankStore,
    PrototypeContextualizer,
    PrototypeContextualizerConfig,
    PrototypeProjector,
    PrototypeProjectorConfig,
    PrototypeRouter,
    PrototypeRouterConfig,
    PrototypeScorer,
    PrototypeScorerConfig,
    PrototypeSurrogateBuilder,
    PrototypeSurrogateBuilderConfig,
)
from prototype.prototype_branch.scorer import PrototypeScoreSurface

from .model_runtime import IntegratedRuntimeConfig, IntegratedScoringRuntime
from .stage_controller import StageConfig, StageController, StagePolicy

Tensor = torch.Tensor


class HostRuntimeProtocol(Protocol):
    """Required host runtime surface for Phase F execution."""

    def extract_feature_surface(self, batch: Mapping[str, Any]) -> HostFeatureSurface:
        raise NotImplementedError

    def compute_host_score_surface(self, features: HostFeatureSurface) -> HostScoreSurface:
        raise NotImplementedError


HostLossFn = Callable[[Mapping[str, Any], HostFeatureSurface, HostScoreSurface], Tensor]
CheckpointLoader = Callable[[str], None]
CalibrationObjectiveFn = Callable[[Tensor, Mapping[str, Any]], Tensor]


@dataclass(frozen=True)
class TrainingRuntimeHooks:
    """Optional runtime hooks for canonical host-loss and checkpoint calls."""

    host_loss_itself_fn: HostLossFn | None = None
    host_loss_clip_fn: HostLossFn | None = None
    clip_backbone_loader: CheckpointLoader | None = None
    stage1_prototype_loader: CheckpointLoader | None = None
    host_checkpoint_loader: CheckpointLoader | None = None
    calibration_objective_fn: CalibrationObjectiveFn | None = None


@dataclass(frozen=True)
class PrototypeLossTerms:
    """Unweighted prototype loss terms."""

    prototype_loss_ret: Tensor
    prototype_loss_diag: Tensor
    prototype_loss_div: Tensor
    prototype_loss_bal: Tensor


@dataclass(frozen=True)
class PrototypeForwardArtifacts:
    """Prototype-branch outputs needed for loss routing."""

    s_proto: Tensor
    score_surface: PrototypeScoreSurface
    routing_alpha: Tensor
    q_summary: Tensor
    projected_diag_surrogate: Tensor
    projected_diag_exact: Tensor
    raw_prototypes: Tensor


@dataclass(frozen=True)
class TrainableGroupSummary:
    """Auditable trainable parameter group summary."""

    group_name: str
    param_names: tuple[str, ...]
    num_params: int


@dataclass(frozen=True)
class TrainingLossReport:
    """Loss report with mode-bound host loss names."""

    host_loss_itself: Tensor | None
    host_loss_clip: Tensor | None
    prototype_loss_ret: Tensor | None
    prototype_loss_diag: Tensor | None
    prototype_loss_div: Tensor | None
    prototype_loss_bal: Tensor | None
    total_objective: Tensor | None
    active_loss_names: tuple[str, ...]


@dataclass(frozen=True)
class TrainingStepOutput:
    """Single-step runtime output."""

    train_mode: TrainMode
    stage: str
    s_host: Tensor
    s_proto: Tensor | None
    s_total: Tensor
    losses: TrainingLossReport
    optimizer_groups: tuple[TrainableGroupSummary, ...]
    initialized_stage2: bool


def _resolve_host_module(host_runtime: HostRuntimeProtocol, explicit: nn.Module | None) -> nn.Module:
    if explicit is not None:
        return explicit
    for attr_name in ("host_model", "model", "module"):
        module = getattr(host_runtime, attr_name, None)
        if isinstance(module, nn.Module):
            return module
    impl = getattr(host_runtime, "_impl", None)
    impl_model = getattr(impl, "model", None)
    if isinstance(impl_model, nn.Module):
        return impl_model
    raise ValueError(
        "Could not resolve host nn.Module for freeze/optimizer enforcement. "
        "Pass host_module explicitly to IntegratedTrainingRuntime."
    )


def _ensure_scalar_loss(name: str, loss: Tensor) -> None:
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(loss)!r}")
    if loss.ndim != 0:
        raise ValueError(f"{name} must be scalar tensor, got shape {tuple(loss.shape)}")


def _host_bound_score_tensor(host_scores: HostScoreSurface) -> Tensor:
    if isinstance(host_scores, ITSELFHostScoreSurface):
        return host_scores.s_host_itself
    if isinstance(host_scores, CLIPHostScoreSurface):
        return host_scores.s_host_clip
    raise TypeError(f"Unsupported host score surface type: {type(host_scores)!r}")


def _mode_matches_features(train_mode: TrainMode, features: HostFeatureSurface) -> bool:
    if train_mode == "itself":
        return isinstance(features, ITSELFHostFeatureSurface)
    return isinstance(features, CLIPHostFeatureSurface)


def _mode_matches_scores(train_mode: TrainMode, scores: HostScoreSurface) -> bool:
    if train_mode == "itself":
        return isinstance(scores, ITSELFHostScoreSurface)
    return isinstance(scores, CLIPHostScoreSurface)


class PrototypeBranchRuntime(nn.Module):
    """Executable prototype-branch runtime using Phase C modules."""

    def __init__(self, config: IntegrationConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.prototype.dim
        self.bank = PrototypeBankStore(
            PrototypeBankConfig(
                num_prototypes=config.prototype.num_prototypes,
                feature_dim=dim,
            )
        )
        self.contextualizer = PrototypeContextualizer(
            PrototypeContextualizerConfig(
                enabled=config.prototype.contextualization.enabled,
                temperature=1.0,
            )
        )
        self.router = PrototypeRouter(
            PrototypeRouterConfig(temperature=config.prototype.temperatures.routing)
        )
        self.basis_builder = PrototypeBasisBuilder(
            PrototypeBasisBuilderConfig(temperature=config.prototype.temperatures.basis)
        )
        self.surrogate_builder = PrototypeSurrogateBuilder(
            PrototypeSurrogateBuilderConfig(
                teacher_temperature=config.prototype.temperatures.teacher
            )
        )
        self.projector = PrototypeProjector(
            PrototypeProjectorConfig(input_dim=dim, output_dim=dim)
        )
        self.scorer = PrototypeScorer(
            PrototypeScorerConfig(temperature=config.prototype.temperatures.retrieval)
        )

    def forward(
        self,
        features: HostFeatureSurface,
        token_mask: Tensor | None = None,
        chunk_size: int | None = None,
    ) -> PrototypeForwardArtifacts:
        if not hasattr(features, "v_i_global") or not hasattr(features, "h_j_tokens"):
            raise TypeError("Feature surface must expose v_i_global and h_j_tokens")
        if not hasattr(features, "z_i_retrieval"):
            raise TypeError("Feature surface must expose z_i_retrieval")

        raw_prototypes = self.bank()
        contextualized = self.contextualizer(raw_prototypes)
        routing = self.router(v_i_global=features.v_i_global, contextualized_prototypes=contextualized)
        basis = self.basis_builder(
            h_j_tokens=features.h_j_tokens,
            contextualized_prototypes=contextualized,
            token_mask=token_mask,
        )
        pairwise_surrogate = self.surrogate_builder.build_pairwise(
            alpha=routing.alpha,
            basis_bank=basis.basis_bank,
            chunk_size=chunk_size,
        )
        pairwise_projected = self.projector(pairwise_surrogate.pairwise_surrogate)
        score_surface = self.scorer.build_score_surface(
            z_i_retrieval=features.z_i_retrieval,
            z_hat_text=pairwise_projected,
        )

        diagonal_surrogate = self.surrogate_builder.build_diagonal(
            alpha=routing.alpha, basis_bank=basis.basis_bank
        )
        exact_teacher = self.surrogate_builder.build_exact_diagonal_teacher(
            q_summary=routing.q_summary,
            h_j_tokens=features.h_j_tokens,
            token_mask=token_mask,
        )
        projected_diag_surrogate = self.projector(diagonal_surrogate.diagonal_surrogate)
        projected_diag_exact = self.projector(exact_teacher.exact_text)

        return PrototypeForwardArtifacts(
            s_proto=score_surface.s_proto,
            score_surface=score_surface,
            routing_alpha=routing.alpha,
            q_summary=routing.q_summary,
            projected_diag_surrogate=projected_diag_surrogate,
            projected_diag_exact=projected_diag_exact,
            raw_prototypes=raw_prototypes,
        )

    def compute_losses(self, artifacts: PrototypeForwardArtifacts) -> PrototypeLossTerms:
        s_proto = artifacts.s_proto
        if s_proto.ndim != 2:
            raise ValueError(f"s_proto must be rank-2 [B, B], got {tuple(s_proto.shape)}")
        if s_proto.shape[0] != s_proto.shape[1]:
            raise ValueError(f"s_proto must be square [B, B], got {tuple(s_proto.shape)}")

        targets = torch.arange(s_proto.shape[0], device=s_proto.device)
        prototype_loss_ret = F.cross_entropy(s_proto, targets)

        diag_cos = F.cosine_similarity(
            artifacts.projected_diag_surrogate,
            artifacts.projected_diag_exact.detach(),
            dim=-1,
        )
        prototype_loss_diag = (1.0 - diag_cos).mean()

        normalized = F.normalize(artifacts.raw_prototypes, p=2, dim=-1)
        gram = normalized @ normalized.t()
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        prototype_loss_div = torch.mean((gram - identity) ** 2)

        alpha_mean = artifacts.routing_alpha.mean(dim=0)
        uniform = torch.full_like(alpha_mean, 1.0 / alpha_mean.numel())
        prototype_loss_bal = F.mse_loss(alpha_mean, uniform)

        return PrototypeLossTerms(
            prototype_loss_ret=prototype_loss_ret,
            prototype_loss_diag=prototype_loss_diag,
            prototype_loss_div=prototype_loss_div,
            prototype_loss_bal=prototype_loss_bal,
        )


class IntegratedTrainingRuntime:
    """Executable Phase F runtime for stage-aware training smoke steps."""

    def __init__(
        self,
        config: IntegrationConfig,
        host_runtime: HostRuntimeProtocol,
        prototype_runtime: PrototypeBranchRuntime | None = None,
        host_module: nn.Module | None = None,
        hooks: TrainingRuntimeHooks | None = None,
    ) -> None:
        self.config = config
        self.host_runtime = host_runtime
        self.host_module = _resolve_host_module(host_runtime=host_runtime, explicit=host_module)
        self.prototype_runtime = prototype_runtime or PrototypeBranchRuntime(config)
        self.hooks = hooks or TrainingRuntimeHooks()
        self.policy: StagePolicy = StageController().resolve(StageConfig(integration=config))
        self.integrated_scoring = IntegratedScoringRuntime(
            IntegratedRuntimeConfig(train_mode=config.train_mode, lambda_f=config.fusion.lambda_f)
        )
        self._stage2_initialized = False
        self._last_optimizer_groups: tuple[TrainableGroupSummary, ...] = ()
        self.apply_freeze_policy()

    @staticmethod
    def supported_stage_mode_matrix() -> tuple[tuple[TrainMode, str], ...]:
        """Return supported stage/mode matrix for executable smoke-step runtime."""
        return tuple(sorted(SUPPORTED_STAGE_MODE_MATRIX))

    @property
    def last_optimizer_groups(self) -> tuple[TrainableGroupSummary, ...]:
        return self._last_optimizer_groups

    def initialize_stage2(self) -> None:
        """Apply Stage 2 initialization precedence exactly once."""
        if self.policy.stage != "stage2":
            return
        if self._stage2_initialized:
            return

        source = self.config.training.initialization.clip_backbone_source
        if source is None:
            raise RuntimeError("Stage 2 requires clip_backbone_source before execution")
        if self.hooks.clip_backbone_loader is None:
            raise RuntimeError(
                "Stage 2 requires clip_backbone_loader hook for canonical CLIP initialization"
            )
        self.hooks.clip_backbone_loader(source)

        stage1_ckpt = self.config.training.initialization.stage1_prototype_checkpoint
        if stage1_ckpt is not None:
            if self.hooks.stage1_prototype_loader is None:
                raise RuntimeError(
                    "stage1_prototype_checkpoint is configured but stage1_prototype_loader hook is missing"
                )
            self.hooks.stage1_prototype_loader(stage1_ckpt)

        host_ckpt = self.config.training.initialization.host_checkpoint
        if host_ckpt is not None:
            if not self.config.training.initialization.host_checkpoint_compatible:
                raise RuntimeError(
                    "host_checkpoint configured without host_checkpoint_compatible=true"
                )
            if self.hooks.host_checkpoint_loader is None:
                raise RuntimeError(
                    "host_checkpoint is configured but host_checkpoint_loader hook is missing"
                )
            self.hooks.host_checkpoint_loader(host_ckpt)
        self._stage2_initialized = True

    def _set_requires_grad(
        self,
        module: nn.Module,
        allow_train: bool,
        allowlist: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        trainable: list[str] = []
        for name, param in module.named_parameters():
            enabled = allow_train
            if allow_train and allowlist:
                enabled = any(token in name for token in allowlist)
            param.requires_grad = enabled
            if enabled:
                trainable.append(name)
        return tuple(trainable)

    def apply_freeze_policy(self) -> None:
        """Enforce stage policy on parameter requires_grad flags."""
        host_allowlist = self.config.training.freeze.host_allowlist
        if self.policy.freeze_host:
            self._set_requires_grad(self.host_module, allow_train=False)
        elif self.policy.stage == "stage2" and host_allowlist:
            self._set_requires_grad(
                self.host_module,
                allow_train=True,
                allowlist=host_allowlist,
            )
        else:
            self._set_requires_grad(self.host_module, allow_train=True)

        prototype_policy = self.config.training.freeze.prototype_policy
        if not self.policy.prototype_enabled or self.policy.freeze_prototype:
            self._set_requires_grad(self.prototype_runtime, allow_train=False)
            return
        if prototype_policy == "freeze_all":
            self._set_requires_grad(self.prototype_runtime, allow_train=False)
            return
        if prototype_policy == "partial":
            self._set_requires_grad(
                self.prototype_runtime,
                allow_train=True,
                allowlist=("projector.", "scorer."),
            )
            return
        self._set_requires_grad(self.prototype_runtime, allow_train=True)

    def build_optimizer_param_groups(
        self, host_lr: float = 1e-4, prototype_lr: float = 1e-4
    ) -> list[dict[str, Any]]:
        """Build optimizer groups aligned with active freeze policy."""
        groups: list[dict[str, Any]] = []
        summaries: list[TrainableGroupSummary] = []

        host_named = [
            (n, p)
            for n, p in self.host_module.named_parameters()
            if p.requires_grad
        ]
        if host_named:
            groups.append({"name": "host", "params": [p for _, p in host_named], "lr": host_lr})
            summaries.append(
                TrainableGroupSummary(
                    group_name="host",
                    param_names=tuple(name for name, _ in host_named),
                    num_params=sum(p.numel() for _, p in host_named),
                )
            )

        prototype_named = [
            (n, p)
            for n, p in self.prototype_runtime.named_parameters()
            if p.requires_grad
        ]
        if prototype_named and self.policy.prototype_enabled:
            groups.append(
                {
                    "name": "prototype",
                    "params": [p for _, p in prototype_named],
                    "lr": prototype_lr,
                }
            )
            summaries.append(
                TrainableGroupSummary(
                    group_name="prototype",
                    param_names=tuple(name for name, _ in prototype_named),
                    num_params=sum(p.numel() for _, p in prototype_named),
                )
            )

        self._last_optimizer_groups = tuple(summaries)

        if self.policy.calibration_only:
            if groups:
                raise RuntimeError(
                    "Stage 3 calibration-only mode forbids representation-learning optimizer groups"
                )
            return []
        if self.policy.stage in ("stage0", "stage2") and not host_named:
            raise RuntimeError(f"{self.policy.stage} requires trainable host parameters")
        if self.policy.stage == "stage1":
            if host_named:
                raise RuntimeError("Stage 1 must not expose trainable host parameters")
            if not prototype_named:
                raise RuntimeError("Stage 1 requires trainable prototype parameters")
        return groups

    def _get_host_loss_callable(self) -> HostLossFn | None:
        if self.config.train_mode == "itself":
            if hasattr(self.host_runtime, "compute_host_loss_itself"):
                return getattr(self.host_runtime, "compute_host_loss_itself")
            return self.hooks.host_loss_itself_fn
        if hasattr(self.host_runtime, "compute_host_loss_clip"):
            return getattr(self.host_runtime, "compute_host_loss_clip")
        return self.hooks.host_loss_clip_fn

    def _compute_host_loss(
        self,
        batch: Mapping[str, Any],
        features: HostFeatureSurface,
        host_scores: HostScoreSurface,
    ) -> Tensor | None:
        if not self.policy.host_loss_enabled:
            return None
        loss_fn = self._get_host_loss_callable()
        if loss_fn is None:
            raise RuntimeError(
                f"{self.config.train_mode} mode host loss is enabled but no canonical host-loss callable is available"
            )
        host_loss = loss_fn(batch, features, host_scores)
        _ensure_scalar_loss(
            "host_loss_itself" if self.config.train_mode == "itself" else "host_loss_clip",
            host_loss,
        )
        return host_loss

    def _compose_loss(
        self,
        host_loss: Tensor | None,
        proto_losses: PrototypeLossTerms | None,
    ) -> TrainingLossReport:
        active: list[str] = []
        total: Tensor | None = None

        host_loss_itself: Tensor | None = None
        host_loss_clip: Tensor | None = None
        if host_loss is not None:
            if self.config.train_mode == "itself":
                host_loss_itself = host_loss
                active.append("host_loss_itself")
            else:
                host_loss_clip = host_loss
                active.append("host_loss_clip")
            total = host_loss if total is None else total + host_loss

        ret_loss: Tensor | None = None
        diag_loss: Tensor | None = None
        div_loss: Tensor | None = None
        bal_loss: Tensor | None = None
        if proto_losses is not None:
            if self.policy.prototype_ret_enabled:
                ret_loss = proto_losses.prototype_loss_ret
                total = ret_loss if total is None else total + ret_loss
                active.append("prototype_loss_ret")
            if self.policy.prototype_diag_enabled:
                diag_loss = proto_losses.prototype_loss_diag
                total = diag_loss if total is None else total + diag_loss
                active.append("prototype_loss_diag")
            if self.policy.prototype_div_enabled:
                weighted_div = (
                    proto_losses.prototype_loss_div
                    * self.config.prototype.regularization.diversity.weight
                )
                div_loss = weighted_div
                total = weighted_div if total is None else total + weighted_div
                active.append("prototype_loss_div")
            if self.policy.prototype_bal_enabled:
                weighted_bal = (
                    proto_losses.prototype_loss_bal
                    * self.config.prototype.regularization.balance.weight
                )
                bal_loss = weighted_bal
                total = weighted_bal if total is None else total + weighted_bal
                active.append("prototype_loss_bal")

        if self.policy.stage == "stage1" and host_loss is not None:
            raise RuntimeError("Stage 1 objective must not include host loss")
        if self.policy.calibration_only and active:
            raise RuntimeError("Stage 3 calibration-only mode must not include representation-learning losses")

        return TrainingLossReport(
            host_loss_itself=host_loss_itself,
            host_loss_clip=host_loss_clip,
            prototype_loss_ret=ret_loss,
            prototype_loss_diag=diag_loss,
            prototype_loss_div=div_loss,
            prototype_loss_bal=bal_loss,
            total_objective=total,
            active_loss_names=tuple(active),
        )

    def _assert_frozen_grads(self) -> None:
        for _, param in self.host_module.named_parameters():
            if not param.requires_grad and param.grad is not None:
                if torch.any(param.grad != 0):
                    raise RuntimeError("Frozen host parameter received non-zero gradient")
        for _, param in self.prototype_runtime.named_parameters():
            if not param.requires_grad and param.grad is not None:
                if torch.any(param.grad != 0):
                    raise RuntimeError("Frozen prototype parameter received non-zero gradient")

    def training_step(
        self,
        batch: Mapping[str, Any],
        optimizer: torch.optim.Optimizer | None = None,
    ) -> TrainingStepOutput:
        """Run one stage-aware training smoke step."""
        if (self.config.train_mode, self.policy.stage) not in SUPPORTED_STAGE_MODE_MATRIX:
            raise RuntimeError(
                f"Unsupported runtime pair {(self.config.train_mode, self.policy.stage)!r}"
            )
        if self.policy.stage == "stage2":
            self.initialize_stage2()

        features = self.host_runtime.extract_feature_surface(batch)
        if not _mode_matches_features(self.config.train_mode, features):
            raise TypeError(
                f"Feature surface type {type(features)!r} does not match train_mode={self.config.train_mode!r}"
            )
        host_scores = self.host_runtime.compute_host_score_surface(features)
        if not _mode_matches_scores(self.config.train_mode, host_scores):
            raise TypeError(
                f"Host score surface type {type(host_scores)!r} does not match train_mode={self.config.train_mode!r}"
            )
        s_host = _host_bound_score_tensor(host_scores)

        token_mask = batch.get("token_mask")
        proto_artifacts: PrototypeForwardArtifacts | None = None
        proto_losses: PrototypeLossTerms | None = None
        s_proto: Tensor | None = None
        s_total = s_host
        if self.policy.prototype_enabled:
            proto_artifacts = self.prototype_runtime(
                features=features,
                token_mask=token_mask,
            )
            s_proto = proto_artifacts.s_proto
            proto_losses = self.prototype_runtime.compute_losses(proto_artifacts)
            fused = self.integrated_scoring.fuse_active_mode(
                host_scores=host_scores,
                prototype_score=proto_artifacts.score_surface,
            )
            s_total = fused.s_total

        host_loss = self._compute_host_loss(batch, features, host_scores)
        loss_report = self._compose_loss(host_loss=host_loss, proto_losses=proto_losses)

        if self.policy.calibration_only and self.hooks.calibration_objective_fn is not None:
            calibration_loss = self.hooks.calibration_objective_fn(s_total, batch)
            _ensure_scalar_loss("calibration_objective", calibration_loss)
            loss_report = TrainingLossReport(
                host_loss_itself=loss_report.host_loss_itself,
                host_loss_clip=loss_report.host_loss_clip,
                prototype_loss_ret=loss_report.prototype_loss_ret,
                prototype_loss_diag=loss_report.prototype_loss_diag,
                prototype_loss_div=loss_report.prototype_loss_div,
                prototype_loss_bal=loss_report.prototype_loss_bal,
                total_objective=calibration_loss,
                active_loss_names=loss_report.active_loss_names,
            )

        if optimizer is not None:
            if self.policy.calibration_only:
                raise RuntimeError(
                    "Stage 3 calibration-only mode forbids representation-learning optimizer updates"
                )
            if loss_report.total_objective is None:
                raise RuntimeError("No objective available for optimizer step")
            optimizer.zero_grad(set_to_none=True)
            loss_report.total_objective.backward()
            self._assert_frozen_grads()
            optimizer.step()

        return TrainingStepOutput(
            train_mode=self.config.train_mode,
            stage=self.policy.stage,
            s_host=s_host,
            s_proto=s_proto,
            s_total=s_total,
            losses=loss_report,
            optimizer_groups=self._last_optimizer_groups,
            initialized_stage2=self._stage2_initialized,
        )
