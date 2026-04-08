"""Phase D tests for score-level fusion and integrated scoring runtime."""

from __future__ import annotations

import pytest
import torch

from prototype.fusion import ResidualFusionConfig, ResidualScoreFusion
from prototype.integration.feature_surface import CLIPHostScoreSurface, ITSELFHostScoreSurface
from prototype.integration.model_runtime import IntegratedRuntimeConfig, IntegratedScoringRuntime
from prototype.prototype_branch.scorer import PrototypeScoreSurface


def _score_tensor(batch: int, fill: float) -> torch.Tensor:
    return torch.full((batch, batch), fill, dtype=torch.float32)


def test_residual_fusion_exact_arithmetic_itself_mode() -> None:
    fusion = ResidualScoreFusion(ResidualFusionConfig(lambda_f=0.3))
    s_host_itself = _score_tensor(batch=3, fill=2.0)
    s_proto = _score_tensor(batch=3, fill=4.0)

    out = fusion.fuse_itself_scores(s_host_itself=s_host_itself, s_proto=s_proto)
    expected = s_host_itself + 0.3 * s_proto
    assert torch.equal(out.s_total, expected)
    assert out.train_mode == "itself"


def test_residual_fusion_lambda_zero_exact_parity_itself_mode() -> None:
    fusion = ResidualScoreFusion(ResidualFusionConfig(lambda_f=0.0))
    s_host_itself = torch.randn(4, 4)
    s_proto = torch.randn(4, 4)
    out = fusion.fuse_itself_scores(s_host_itself=s_host_itself, s_proto=s_proto)
    assert torch.equal(out.s_total, s_host_itself)


def test_residual_fusion_lambda_zero_exact_parity_clip_mode() -> None:
    fusion = ResidualScoreFusion(ResidualFusionConfig(lambda_f=0.0))
    s_host_clip = torch.randn(4, 4)
    s_proto = torch.randn(4, 4)
    out = fusion.fuse_clip_scores(s_host_clip=s_host_clip, s_proto=s_proto)
    assert torch.equal(out.s_total, s_host_clip)


def test_residual_fusion_shape_mismatch_rejected() -> None:
    fusion = ResidualScoreFusion(ResidualFusionConfig(lambda_f=0.2))
    with pytest.raises(ValueError):
        _ = fusion.fuse_itself_scores(
            s_host_itself=torch.randn(3, 3), s_proto=torch.randn(3, 4)
        )


def test_residual_fusion_rejects_non_score_tensor_rank() -> None:
    fusion = ResidualScoreFusion(ResidualFusionConfig(lambda_f=0.2))
    with pytest.raises(ValueError):
        _ = fusion.fuse_clip_scores(
            s_host_clip=torch.randn(2, 2), s_proto=torch.randn(2, 2, 8)
        )


def test_integrated_runtime_itself_mode_fusion() -> None:
    runtime = IntegratedScoringRuntime(
        IntegratedRuntimeConfig(train_mode="itself", lambda_f=0.5)
    )
    host_scores = ITSELFHostScoreSurface(
        s_host_itself=_score_tensor(batch=2, fill=1.0),
        s_global_itself=_score_tensor(batch=2, fill=0.8),
        s_local_itself=_score_tensor(batch=2, fill=1.2),
        lambda_s=0.5,
    )
    proto_surface = PrototypeScoreSurface(s_proto=_score_tensor(batch=2, fill=2.0))
    out = runtime.fuse_active_mode(host_scores=host_scores, prototype_score=proto_surface)
    assert out.train_mode == "itself"
    assert torch.equal(out.s_total, host_scores.s_host_itself + 0.5 * proto_surface.s_proto)


def test_integrated_runtime_clip_mode_fusion() -> None:
    runtime = IntegratedScoringRuntime(
        IntegratedRuntimeConfig(train_mode="clip", lambda_f=0.25)
    )
    host_scores = CLIPHostScoreSurface(s_host_clip=_score_tensor(batch=3, fill=1.0))
    proto_surface = PrototypeScoreSurface(s_proto=_score_tensor(batch=3, fill=4.0))
    out = runtime.fuse_active_mode(host_scores=host_scores, prototype_score=proto_surface)
    assert out.train_mode == "clip"
    assert torch.equal(out.s_total, host_scores.s_host_clip + 0.25 * proto_surface.s_proto)


def test_integrated_runtime_mode_binding_guard() -> None:
    runtime = IntegratedScoringRuntime(
        IntegratedRuntimeConfig(train_mode="clip", lambda_f=0.1)
    )
    host_scores_itself = ITSELFHostScoreSurface(
        s_host_itself=_score_tensor(batch=2, fill=1.0),
        s_global_itself=_score_tensor(batch=2, fill=1.0),
        s_local_itself=_score_tensor(batch=2, fill=1.0),
        lambda_s=0.5,
    )
    with pytest.raises(TypeError):
        _ = runtime.fuse_active_mode(
            host_scores=host_scores_itself,
            prototype_score=PrototypeScoreSurface(s_proto=_score_tensor(batch=2, fill=1.0)),
        )
