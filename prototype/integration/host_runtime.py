"""Host runtime loading and feature/score surface extraction for Phase B."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
import torch.nn.functional as F

from .feature_surface import (
    CLIPHostFeatureSurface,
    CLIPHostScoreSurface,
    HostFeatureSurface,
    HostScoreSurface,
    ITSELFHostFeatureSurface,
    ITSELFHostScoreSurface,
    TrainMode,
)


class ITSELFModelProtocol(Protocol):
    """Required callable surface of ITSELF host model."""

    base_model: Any

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_image_grab(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_text_grab(self, text: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CLIPModelProtocol(Protocol):
    """Required callable surface of canonical CLIP host runtime."""

    def encode_image(self, image: torch.Tensor) -> Any:
        raise NotImplementedError

    def encode_text(self, text: torch.Tensor) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class HostRuntimeConfig:
    """Host runtime config for mode-aware loading and extraction."""

    train_mode: TrainMode
    lambda_s: float | None = None
    host_root: Path = Path("prototype/adapter/WACV2026-Oral-ITSELF")
    device: str | None = None
    itself_model: ITSELFModelProtocol | None = None
    clip_model: CLIPModelProtocol | None = None
    itself_build_args: Any | None = None
    itself_num_classes: int = 11003
    clip_pretrain_choice: str | None = None
    clip_image_size: tuple[int, int] | None = None
    clip_stride_size: int | None = None

    def __post_init__(self) -> None:
        if self.train_mode not in ("itself", "clip"):
            raise ValueError(f"Unsupported train_mode={self.train_mode!r}")
        if self.train_mode == "itself" and self.lambda_s is None:
            raise ValueError("lambda_s is required for train_mode='itself'")
        if self.train_mode == "clip" and self.lambda_s is not None:
            raise ValueError("lambda_s must be None for train_mode='clip'")


def _import_host_module(host_root: Path, module_name: str) -> Any:
    host_root_abs = str(host_root.resolve())
    if host_root_abs not in sys.path:
        sys.path.insert(0, host_root_abs)
    return importlib.import_module(module_name)


def _require_batch_fields(batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if "images" not in batch:
        raise KeyError("Batch must contain key 'images'")
    if "caption_ids" not in batch:
        raise KeyError("Batch must contain key 'caption_ids'")
    images = batch["images"]
    caption_ids = batch["caption_ids"]
    if not isinstance(images, torch.Tensor):
        raise TypeError("batch['images'] must be torch.Tensor")
    if not isinstance(caption_ids, torch.Tensor):
        raise TypeError("batch['caption_ids'] must be torch.Tensor")
    return images, caption_ids


def _eot_index(caption_ids: torch.Tensor) -> torch.Tensor:
    return caption_ids.argmax(dim=-1)


def _cosine_similarity_matrix(v_feats: torch.Tensor, t_feats: torch.Tensor) -> torch.Tensor:
    v_norm = F.normalize(v_feats, p=2, dim=1)
    t_norm = F.normalize(t_feats, p=2, dim=1)
    return v_norm @ t_norm.t()


class _ITSELFHostRuntime:
    """Read-only ITSELF runtime adapter."""

    def __init__(self, config: HostRuntimeConfig, model: ITSELFModelProtocol) -> None:
        self.config = config
        self.model = model

    def extract_feature_surface(self, batch: Mapping[str, Any]) -> ITSELFHostFeatureSurface:
        images, caption_ids = _require_batch_fields(batch)
        caption_ids_long = caption_ids.long()

        v_i_global = self.model.encode_image(images).float()

        # Host token states H_j from base text encoder at the same level before pooled EOT extraction.
        h_j_tokens, _ = self.model.base_model.encode_text(caption_ids_long)
        h_j_tokens = h_j_tokens.float()

        eot = _eot_index(caption_ids_long)
        t_j_global = h_j_tokens[torch.arange(h_j_tokens.shape[0], device=h_j_tokens.device), eot].float()
        z_i_retrieval = v_i_global

        v_i_local = self.model.encode_image_grab(images).float()
        t_j_local = self.model.encode_text_grab(caption_ids_long).float()

        return ITSELFHostFeatureSurface(
            v_i_global=v_i_global,
            h_j_tokens=h_j_tokens,
            t_j_global=t_j_global,
            z_i_retrieval=z_i_retrieval,
            v_i_local=v_i_local,
            t_j_local=t_j_local,
            v_i_global_source="itself.encode_image -> base_model.encode_image[:,0,:]",
            h_j_tokens_source="itself.base_model.encode_text tokens before EOT pooling",
            t_j_global_source="EOT pooling over itself.base_model.encode_text token states",
            z_i_retrieval_source="itself global retrieval feature (same semantic space as v_i_global)",
            v_i_local_source="itself.encode_image_grab",
            t_j_local_source="itself.encode_text_grab",
        )

    def compute_host_score_surface(
        self, features: ITSELFHostFeatureSurface
    ) -> ITSELFHostScoreSurface:
        lambda_s = float(self.config.lambda_s)
        s_global_itself = _cosine_similarity_matrix(features.v_i_global, features.t_j_global)
        s_local_itself = _cosine_similarity_matrix(features.v_i_local, features.t_j_local)
        s_host_itself = lambda_s * s_global_itself + (1.0 - lambda_s) * s_local_itself
        return ITSELFHostScoreSurface(
            s_host_itself=s_host_itself,
            s_global_itself=s_global_itself,
            s_local_itself=s_local_itself,
            lambda_s=lambda_s,
        )


class _CLIPHostRuntime:
    """Read-only CLIP runtime adapter."""

    def __init__(self, config: HostRuntimeConfig, model: CLIPModelProtocol) -> None:
        self.config = config
        self.model = model

    def extract_feature_surface(self, batch: Mapping[str, Any]) -> CLIPHostFeatureSurface:
        images, caption_ids = _require_batch_fields(batch)
        caption_ids_long = caption_ids.long()

        image_features, _ = self.model.encode_image(images)
        if not isinstance(image_features, torch.Tensor):
            raise TypeError("clip.encode_image must return Tensor as first output")
        if image_features.ndim == 3:
            v_i_global = image_features[:, 0, :].float()
        elif image_features.ndim == 2:
            v_i_global = image_features.float()
        else:
            raise ValueError(f"Unsupported CLIP image feature shape: {tuple(image_features.shape)}")

        h_j_tokens, _ = self.model.encode_text(caption_ids_long)
        if not isinstance(h_j_tokens, torch.Tensor):
            raise TypeError("clip.encode_text must return Tensor as first output")
        if h_j_tokens.ndim != 3:
            raise ValueError(
                "Canonical CLIP runtime for this integration requires token-level text states [B, L, D]"
            )
        h_j_tokens = h_j_tokens.float()

        eot = _eot_index(caption_ids_long)
        t_j_global = h_j_tokens[torch.arange(h_j_tokens.shape[0], device=h_j_tokens.device), eot].float()
        z_i_retrieval = v_i_global

        return CLIPHostFeatureSurface(
            v_i_global=v_i_global,
            h_j_tokens=h_j_tokens,
            t_j_global=t_j_global,
            z_i_retrieval=z_i_retrieval,
            v_i_global_source="clip.encode_image global retrieval feature",
            h_j_tokens_source="clip.encode_text token states before CLIP EOT pooling",
            t_j_global_source="EOT pooling over clip.encode_text token states",
            z_i_retrieval_source="clip global retrieval feature (same semantic space as v_i_global)",
        )

    def compute_host_score_surface(self, features: CLIPHostFeatureSurface) -> CLIPHostScoreSurface:
        s_host_clip = _cosine_similarity_matrix(features.v_i_global, features.t_j_global)
        return CLIPHostScoreSurface(s_host_clip=s_host_clip)


def _load_itself_model(config: HostRuntimeConfig) -> ITSELFModelProtocol:
    if config.itself_model is not None:
        return config.itself_model
    if config.itself_build_args is None:
        raise ValueError(
            "itself_build_args is required to build ITSELF host model when itself_model is not provided"
        )
    build_mod = _import_host_module(config.host_root, "model.build")
    build_model = getattr(build_mod, "build_model")
    model = build_model(config.itself_build_args, config.itself_num_classes)
    if config.device:
        model = model.to(config.device)
    return model


def _load_clip_model(config: HostRuntimeConfig) -> CLIPModelProtocol:
    if config.clip_model is not None:
        return config.clip_model
    if config.clip_pretrain_choice is None:
        raise ValueError(
            "clip_pretrain_choice is required to build CLIP host model when clip_model is not provided"
        )
    if config.clip_image_size is None or config.clip_stride_size is None:
        raise ValueError(
            "clip_image_size and clip_stride_size are required for canonical CLIP host runtime"
        )
    clip_mod = _import_host_module(config.host_root, "model.clip_model")
    build_clip = getattr(clip_mod, "build_CLIP_from_openai_pretrained")
    model, _ = build_clip(
        config.clip_pretrain_choice, config.clip_image_size, config.clip_stride_size
    )
    if config.device:
        model = model.to(config.device)
    return model


class HostRuntime:
    """Mode-aware host runtime facade for feature and score extraction."""

    def __init__(self, config: HostRuntimeConfig) -> None:
        self.config = config
        if config.train_mode == "itself":
            self._impl: _ITSELFHostRuntime | _CLIPHostRuntime = _ITSELFHostRuntime(
                config=config, model=_load_itself_model(config)
            )
        elif config.train_mode == "clip":
            self._impl = _CLIPHostRuntime(config=config, model=_load_clip_model(config))
        else:
            raise ValueError(f"Unsupported train_mode={config.train_mode!r}")

    def extract_feature_surface(self, batch: Mapping[str, Any]) -> HostFeatureSurface:
        """Extract mode-bound host feature surface with provenance and shape guards."""
        return self._impl.extract_feature_surface(batch)

    def compute_host_score_surface(self, features: HostFeatureSurface) -> HostScoreSurface:
        """Compute mode-bound canonical host score surface (`s_host^itself` or `s_host^clip`)."""
        if self.config.train_mode == "itself":
            if not isinstance(features, ITSELFHostFeatureSurface):
                raise TypeError("Expected ITSELFHostFeatureSurface for train_mode='itself'")
            return self._impl.compute_host_score_surface(features)
        if not isinstance(features, CLIPHostFeatureSurface):
            raise TypeError("Expected CLIPHostFeatureSurface for train_mode='clip'")
        return self._impl.compute_host_score_surface(features)
