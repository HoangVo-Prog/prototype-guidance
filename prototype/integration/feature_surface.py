"""Typed feature/score surfaces with provenance and shape guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import torch

TrainMode = Literal["itself", "clip"]
Tensor = torch.Tensor


def _ensure_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")


def _ensure_ndim(name: str, value: Tensor, expected_ndim: int) -> None:
    _ensure_tensor(name, value)
    if value.ndim != expected_ndim:
        raise ValueError(
            f"{name} must be rank-{expected_ndim}, got shape {tuple(value.shape)}"
        )


def _ensure_same_batch(name: str, lhs: Tensor, rhs: Tensor) -> None:
    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError(
            f"{name} batch mismatch: {lhs.shape[0]} vs {rhs.shape[0]}"
        )


def _ensure_same_embedding_dim(name: str, lhs: Tensor, rhs: Tensor) -> None:
    if lhs.shape[-1] != rhs.shape[-1]:
        raise ValueError(
            f"{name} embedding dim mismatch: {lhs.shape[-1]} vs {rhs.shape[-1]}"
        )


def _ensure_square_score(name: str, value: Tensor) -> None:
    _ensure_ndim(name, value, expected_ndim=2)
    if value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be square [B, B], got {tuple(value.shape)}")


@dataclass(frozen=True)
class ITSELFHostFeatureSurface:
    """Feature surface for `train_mode=itself`."""

    v_i_global: Tensor
    h_j_tokens: Tensor
    t_j_global: Tensor
    z_i_retrieval: Tensor
    v_i_local: Tensor
    t_j_local: Tensor
    v_i_global_source: str
    h_j_tokens_source: str
    t_j_global_source: str
    z_i_retrieval_source: str
    v_i_local_source: str
    t_j_local_source: str
    train_mode: Literal["itself"] = "itself"

    def __post_init__(self) -> None:
        _ensure_ndim("v_i_global", self.v_i_global, 2)
        _ensure_ndim("h_j_tokens", self.h_j_tokens, 3)
        _ensure_ndim("t_j_global", self.t_j_global, 2)
        _ensure_ndim("z_i_retrieval", self.z_i_retrieval, 2)
        _ensure_ndim("v_i_local", self.v_i_local, 2)
        _ensure_ndim("t_j_local", self.t_j_local, 2)
        _ensure_same_batch("v_i_global/t_j_global", self.v_i_global, self.t_j_global)
        _ensure_same_batch("h_j_tokens/t_j_global", self.h_j_tokens, self.t_j_global)
        _ensure_same_batch("z_i_retrieval/v_i_global", self.z_i_retrieval, self.v_i_global)
        _ensure_same_batch("v_i_local/t_j_local", self.v_i_local, self.t_j_local)
        _ensure_same_embedding_dim("v_i_global/t_j_global", self.v_i_global, self.t_j_global)
        _ensure_same_embedding_dim(
            "z_i_retrieval/v_i_global", self.z_i_retrieval, self.v_i_global
        )
        _ensure_same_embedding_dim(
            "h_j_tokens/t_j_global", self.h_j_tokens, self.t_j_global
        )
        if not self.v_i_global_source or not self.h_j_tokens_source or not self.t_j_global_source:
            raise ValueError("Provenance sources for core ITSELF tensors must be non-empty")


@dataclass(frozen=True)
class CLIPHostFeatureSurface:
    """Feature surface for `train_mode=clip` (no GRAB/local branch)."""

    v_i_global: Tensor
    h_j_tokens: Tensor
    t_j_global: Tensor
    z_i_retrieval: Tensor
    v_i_global_source: str
    h_j_tokens_source: str
    t_j_global_source: str
    z_i_retrieval_source: str
    train_mode: Literal["clip"] = "clip"

    def __post_init__(self) -> None:
        _ensure_ndim("v_i_global", self.v_i_global, 2)
        _ensure_ndim("h_j_tokens", self.h_j_tokens, 3)
        _ensure_ndim("t_j_global", self.t_j_global, 2)
        _ensure_ndim("z_i_retrieval", self.z_i_retrieval, 2)
        _ensure_same_batch("v_i_global/t_j_global", self.v_i_global, self.t_j_global)
        _ensure_same_batch("h_j_tokens/t_j_global", self.h_j_tokens, self.t_j_global)
        _ensure_same_batch("z_i_retrieval/v_i_global", self.z_i_retrieval, self.v_i_global)
        _ensure_same_embedding_dim("v_i_global/t_j_global", self.v_i_global, self.t_j_global)
        _ensure_same_embedding_dim(
            "z_i_retrieval/v_i_global", self.z_i_retrieval, self.v_i_global
        )
        _ensure_same_embedding_dim(
            "h_j_tokens/t_j_global", self.h_j_tokens, self.t_j_global
        )
        if not self.v_i_global_source or not self.h_j_tokens_source or not self.t_j_global_source:
            raise ValueError("Provenance sources for core CLIP tensors must be non-empty")


@dataclass(frozen=True)
class ITSELFHostScoreSurface:
    """Canonical score surface for `train_mode=itself`."""

    s_host_itself: Tensor
    s_global_itself: Tensor
    s_local_itself: Tensor
    lambda_s: float
    score_orientation: Literal["image_to_text"] = "image_to_text"
    train_mode: Literal["itself"] = "itself"

    def __post_init__(self) -> None:
        _ensure_square_score("s_host_itself", self.s_host_itself)
        _ensure_square_score("s_global_itself", self.s_global_itself)
        _ensure_square_score("s_local_itself", self.s_local_itself)
        if not 0.0 <= self.lambda_s <= 1.0:
            raise ValueError(f"lambda_s must be in [0, 1], got {self.lambda_s}")

    @property
    def bound_host_score(self) -> Tensor:
        """Return the mode-bound host score tensor."""
        return self.s_host_itself


@dataclass(frozen=True)
class CLIPHostScoreSurface:
    """Canonical score surface for `train_mode=clip`."""

    s_host_clip: Tensor
    score_orientation: Literal["image_to_text"] = "image_to_text"
    train_mode: Literal["clip"] = "clip"

    def __post_init__(self) -> None:
        _ensure_square_score("s_host_clip", self.s_host_clip)

    @property
    def bound_host_score(self) -> Tensor:
        """Return the mode-bound host score tensor."""
        return self.s_host_clip


HostFeatureSurface = Union[ITSELFHostFeatureSurface, CLIPHostFeatureSurface]
HostScoreSurface = Union[ITSELFHostScoreSurface, CLIPHostScoreSurface]
