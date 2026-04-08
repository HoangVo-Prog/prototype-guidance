"""Phase B tests for host feature provenance and shape validation."""

from __future__ import annotations

import torch

from prototype.integration.feature_surface import (
    CLIPHostFeatureSurface,
    CLIPHostScoreSurface,
    ITSELFHostFeatureSurface,
    ITSELFHostScoreSurface,
)
from prototype.integration.host_runtime import HostRuntime, HostRuntimeConfig
from prototype.prototype_branch import PrototypeBasisBuilder, PrototypeRouter


class _DummyBaseModel:
    def __init__(self, token_dim: int) -> None:
        self.token_dim = token_dim

    def encode_text(self, text: torch.Tensor) -> tuple[torch.Tensor, None]:
        batch_size, seq_len = text.shape
        tokens = torch.arange(
            batch_size * seq_len * self.token_dim, dtype=torch.float32
        ).reshape(batch_size, seq_len, self.token_dim)
        return tokens, None


class _DummyITSELFModel:
    def __init__(self, global_dim: int = 4, local_dim: int = 6) -> None:
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.base_model = _DummyBaseModel(token_dim=global_dim)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]
        return torch.ones(batch_size, self.global_dim, dtype=torch.float32)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.base_model.encode_text(text)
        eot = text.argmax(dim=-1)
        return tokens[torch.arange(tokens.shape[0]), eot]

    def encode_image_grab(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]
        return torch.full((batch_size, self.local_dim), 2.0, dtype=torch.float32)

    def encode_text_grab(self, text: torch.Tensor) -> torch.Tensor:
        batch_size = text.shape[0]
        return torch.full((batch_size, self.local_dim), 3.0, dtype=torch.float32)


class _DummyCLIPModel:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def encode_image(self, image: torch.Tensor) -> tuple[torch.Tensor, None]:
        batch_size = image.shape[0]
        tokens = torch.arange(
            batch_size * 3 * self.dim, dtype=torch.float32
        ).reshape(batch_size, 3, self.dim)
        return tokens, None

    def encode_text(self, text: torch.Tensor) -> tuple[torch.Tensor, None]:
        batch_size, seq_len = text.shape
        tokens = torch.arange(
            batch_size * seq_len * self.dim, dtype=torch.float32
        ).reshape(batch_size, seq_len, self.dim)
        return tokens, None


def _build_batch(batch_size: int = 2, seq_len: int = 5) -> dict[str, torch.Tensor]:
    images = torch.randn(batch_size, 3, 32, 16)
    caption_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    caption_ids[0, 1] = 1
    caption_ids[1, 3] = 1
    return {"images": images, "caption_ids": caption_ids}


def test_itself_feature_surface_provenance_and_shapes() -> None:
    runtime = HostRuntime(
        HostRuntimeConfig(
            train_mode="itself",
            lambda_s=0.3,
            itself_model=_DummyITSELFModel(),
        )
    )
    batch = _build_batch()
    features = runtime.extract_feature_surface(batch)
    assert isinstance(features, ITSELFHostFeatureSurface)
    assert features.v_i_global.shape == (2, 4)
    assert features.h_j_tokens.shape == (2, 5, 4)
    assert features.t_j_global.shape == (2, 4)
    assert features.v_i_local.shape == (2, 6)
    assert features.t_j_local.shape == (2, 6)
    assert "base_model.encode_text" in features.h_j_tokens_source
    assert "encode_image" in features.v_i_global_source
    assert "retrieval" in features.z_i_retrieval_source

    # Routing input provenance must come from active host global image feature.
    router = PrototypeRouter()
    routed = router.route(features.v_i_global, torch.randn(3, 4))
    assert routed.alpha.shape == (2, 3)

    score_surface = runtime.compute_host_score_surface(features)
    assert isinstance(score_surface, ITSELFHostScoreSurface)
    expected = (
        0.3 * score_surface.s_global_itself + 0.7 * score_surface.s_local_itself
    )
    assert torch.allclose(score_surface.s_host_itself, expected)


def test_clip_feature_surface_provenance_and_shapes() -> None:
    runtime = HostRuntime(
        HostRuntimeConfig(
            train_mode="clip",
            clip_model=_DummyCLIPModel(),
        )
    )
    batch = _build_batch()
    features = runtime.extract_feature_surface(batch)
    assert isinstance(features, CLIPHostFeatureSurface)
    assert features.v_i_global.shape == (2, 4)
    assert features.h_j_tokens.shape == (2, 5, 4)
    assert features.t_j_global.shape == (2, 4)
    assert "clip.encode_text" in features.h_j_tokens_source
    assert not hasattr(features, "v_i_local")
    assert not hasattr(features, "t_j_local")

    score_surface = runtime.compute_host_score_surface(features)
    assert isinstance(score_surface, CLIPHostScoreSurface)
    assert score_surface.s_host_clip.shape == (2, 2)


def test_mode_binding_guards_for_lambda_s() -> None:
    _ = HostRuntimeConfig(train_mode="itself", lambda_s=0.5)
    _ = HostRuntimeConfig(train_mode="clip", lambda_s=None)

    try:
        HostRuntimeConfig(train_mode="clip", lambda_s=0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("clip mode must reject lambda_s")


def test_basis_builder_rejects_pooled_text_for_token_basis_substrate() -> None:
    builder = PrototypeBasisBuilder()
    pooled_text = torch.randn(2, 4)
    prototypes = torch.randn(3, 4)
    try:
        _ = builder.build_basis(h_j_tokens=pooled_text, contextualized_prototypes=prototypes)
    except ValueError:
        pass
    else:
        raise AssertionError("Basis builder must reject pooled text `[B, D]` as token substrate")
