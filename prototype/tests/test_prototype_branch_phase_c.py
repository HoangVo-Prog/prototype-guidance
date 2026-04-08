"""Phase C safety and shape checks for prototype branch core modules."""

from __future__ import annotations

import pytest
import torch

from prototype.prototype_branch import (
    PrototypeBankConfig,
    PrototypeBankStore,
    PrototypeBasisBuilder,
    PrototypeBasisBuilderConfig,
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


def test_router_row_sum_behavior() -> None:
    torch.manual_seed(0)
    router = PrototypeRouter(PrototypeRouterConfig(temperature=0.2, normalize_inputs=True))
    v_i_global = torch.randn(3, 8)
    prototypes = torch.randn(5, 8)

    out = router.route(v_i_global=v_i_global, contextualized_prototypes=prototypes)
    assert out.alpha.shape == (3, 5)
    assert out.q_summary.shape == (3, 8)
    row_sum = out.alpha.sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-6)


def test_basis_builder_shape_and_token_level_requirement() -> None:
    torch.manual_seed(0)
    builder = PrototypeBasisBuilder(PrototypeBasisBuilderConfig(temperature=0.5))
    h_j_tokens = torch.randn(2, 7, 6)
    prototypes = torch.randn(4, 6)
    token_mask = torch.tensor(
        [[True, True, True, False, False, True, True], [True, True, True, True, False, False, True]]
    )

    out = builder.build_basis(
        h_j_tokens=h_j_tokens,
        contextualized_prototypes=prototypes,
        token_mask=token_mask,
    )
    assert out.basis_bank.shape == (2, 4, 6)
    assert out.token_weights.shape == (2, 4, 7)

    pooled_text = torch.randn(2, 6)
    with pytest.raises(ValueError):
        _ = builder.build_basis(h_j_tokens=pooled_text, contextualized_prototypes=prototypes)


def test_surrogate_rowwise_orientation() -> None:
    builder = PrototypeSurrogateBuilder(PrototypeSurrogateBuilderConfig(teacher_temperature=0.7))
    alpha = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)  # [B_img=2, N=2]

    basis_bank = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],  # caption 0: prototype 0 / prototype 1
            [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],  # caption 1: prototype 0 / prototype 1
        ],
        dtype=torch.float32,
    )  # [B_txt=2, N=2, D=3]

    pairwise = builder.build_pairwise(alpha=alpha, basis_bank=basis_bank).pairwise_surrogate
    assert pairwise.shape == (2, 2, 3)
    # Row 0 uses alpha for image 0 -> chooses prototype 0 for every caption.
    assert torch.allclose(pairwise[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(pairwise[0, 1], torch.tensor([0.0, 1.0, 0.0]))
    # Row 1 uses alpha for image 1 -> chooses prototype 1 for every caption.
    assert torch.allclose(pairwise[1, 0], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(pairwise[1, 1], torch.tensor([0.0, 2.0, 0.0]))


def test_prototype_scorer_outputs_pairwise_shape() -> None:
    torch.manual_seed(0)
    scorer = PrototypeScorer(PrototypeScorerConfig(temperature=0.11, normalize_inputs=True))
    z_i_retrieval = torch.randn(4, 9)
    z_hat_pairwise = torch.randn(4, 4, 9)
    scores = scorer.score_pairwise(z_i_retrieval=z_i_retrieval, z_hat_pairwise_text=z_hat_pairwise)
    assert scores.shape == (4, 4)


def test_rowwise_perturbation_sanity_behavior() -> None:
    torch.manual_seed(0)
    bsz, num_proto, dim = 3, 5, 7
    router = PrototypeRouter(PrototypeRouterConfig(temperature=0.2, normalize_inputs=True))
    surrogate_builder = PrototypeSurrogateBuilder()
    scorer = PrototypeScorer(PrototypeScorerConfig(temperature=0.3, normalize_inputs=True))

    images = torch.randn(bsz, dim)
    images_perturbed = images.clone()
    images_perturbed[1] += 0.5  # perturb only row 1

    prototypes = torch.randn(num_proto, dim)
    basis_bank = torch.randn(bsz, num_proto, dim)

    alpha_a = router.route(images, prototypes).alpha
    alpha_b = router.route(images_perturbed, prototypes).alpha

    pairwise_a = surrogate_builder.build_pairwise(alpha_a, basis_bank).pairwise_surrogate
    pairwise_b = surrogate_builder.build_pairwise(alpha_b, basis_bank).pairwise_surrogate

    score_a = scorer.score_pairwise(images, pairwise_a)
    score_b = scorer.score_pairwise(images_perturbed, pairwise_b)

    assert torch.allclose(score_a[0], score_b[0], atol=1e-6)
    assert torch.allclose(score_a[2], score_b[2], atol=1e-6)
    assert not torch.allclose(score_a[1], score_b[1], atol=1e-6)


def test_exact_diagonal_helper_isolation() -> None:
    torch.manual_seed(0)
    builder = PrototypeSurrogateBuilder(PrototypeSurrogateBuilderConfig(teacher_temperature=0.5))
    q_summary = torch.randn(2, 6)
    h_j_tokens = torch.randn(2, 5, 6)

    out_a = builder.build_exact_diagonal_teacher(q_summary=q_summary, h_j_tokens=h_j_tokens)

    h_j_tokens_modified = h_j_tokens.clone()
    h_j_tokens_modified[1] += 100.0  # modify sample 1 only
    out_b = builder.build_exact_diagonal_teacher(
        q_summary=q_summary, h_j_tokens=h_j_tokens_modified
    )

    # Sample 0 is isolated from sample 1 modifications.
    assert torch.allclose(out_a.exact_text[0], out_b.exact_text[0], atol=1e-6)
    assert torch.allclose(out_a.token_weights[0], out_b.token_weights[0], atol=1e-6)
    # Sample 1 should typically change after the modification.
    assert not torch.allclose(out_a.exact_text[1], out_b.exact_text[1], atol=1e-6)


def test_end_to_end_dummy_batch_phase_c_pipeline() -> None:
    torch.manual_seed(0)
    bsz, seq_len, dim, dim_r, num_proto = 4, 6, 8, 5, 3

    bank = PrototypeBankStore(PrototypeBankConfig(num_prototypes=num_proto, feature_dim=dim))
    contextualizer = PrototypeContextualizer(
        PrototypeContextualizerConfig(enabled=True, temperature=1.0)
    )
    router = PrototypeRouter(PrototypeRouterConfig(temperature=0.2, normalize_inputs=True))
    basis_builder = PrototypeBasisBuilder(PrototypeBasisBuilderConfig(temperature=0.15))
    surrogate_builder = PrototypeSurrogateBuilder()
    projector = PrototypeProjector(
        PrototypeProjectorConfig(
            input_dim=dim, output_dim=dim_r, bias=True, use_layer_norm=True, normalize_output=True
        )
    )
    scorer = PrototypeScorer(PrototypeScorerConfig(temperature=0.3, normalize_inputs=True))

    v_i_global = torch.randn(bsz, dim)
    h_j_tokens = torch.randn(bsz, seq_len, dim)
    z_i_retrieval = torch.randn(bsz, dim_r)

    bank_raw = bank()
    bank_ctx = contextualizer(bank_raw)
    routing = router(v_i_global=v_i_global, contextualized_prototypes=bank_ctx)
    basis = basis_builder(h_j_tokens=h_j_tokens, contextualized_prototypes=bank_ctx)
    pairwise = surrogate_builder.build_pairwise(routing.alpha, basis.basis_bank).pairwise_surrogate
    z_hat = projector(pairwise)
    scores = scorer.score_pairwise(z_i_retrieval=z_i_retrieval, z_hat_pairwise_text=z_hat)

    diag_sur = surrogate_builder.build_diagonal(routing.alpha, basis.basis_bank).diagonal_surrogate
    exact_diag = surrogate_builder.build_exact_diagonal_teacher(
        q_summary=routing.q_summary, h_j_tokens=h_j_tokens
    ).exact_text

    assert scores.shape == (bsz, bsz)
    assert diag_sur.shape == (bsz, dim)
    assert exact_diag.shape == (bsz, dim)
