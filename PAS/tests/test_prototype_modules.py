import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if torch is not None:
    from model.prototype import (
        MaskedTokenPooler,
        PrototypeAggregator,
        PrototypeBank,
        PrototypeConditionedTextHead,
        init_mode_requires_data,
        PrototypeContextualizer,
        PrototypeLosses,
        Router,
        TokenMaskBuilder,
        TokenScorer,
        build_prototype_head,
    )
    from model.prototype.projector import MLPProjector


@unittest.skipUnless(torch is not None, 'PyTorch is required for prototype module tests.')
class PrototypeModuleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.batch_size = 3
        self.seq_len = 6
        self.feature_dim = 8
        self.num_prototypes = 5
        self.num_classes = 3
        self.token_ids = torch.tensor(
            [
                [49406, 11, 12, 49407, 0, 0],
                [49406, 21, 49407, 0, 0, 0],
                [49406, 31, 32, 33, 49407, 0],
            ],
            dtype=torch.long,
        )
        self.attention_mask = self.token_ids.ne(0)
        self.special_positions = {
            'cls': torch.zeros(self.batch_size, dtype=torch.long),
            'eos': torch.tensor([3, 2, 4], dtype=torch.long),
        }

    def _build_args(self, **overrides):
        base = dict(
            use_prototype_bank=True,
            use_image_conditioned_pooling=True,
            use_prototype_contextualization=True,
            prototype_contextualization_enabled=True,
            prototype_num_prototypes=self.num_prototypes,
            prototype_dim=self.feature_dim,
            projection_dim=4,
            projector_hidden_dim=12,
            projector_dropout=0.0,
            projector_type='mlp2',
            normalize_projector_outputs=True,
            prototype_init='normalized_random',
            prototype_init_path=None,
            prototype_init_hybrid_ratio=0.5,
            prototype_init_max_iters=50,
            prototype_init_tol=1e-4,
            prototype_init_seed=17,
            prototype_routing_type='cosine',
            prototype_temperature=0.07,
            token_scoring_type='cosine',
            token_pooling_temperature=0.07,
            normalize_for_token_scoring=True,
            token_policy='content_only',
            special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
            error_on_empty_kept_tokens=True,
            prototype_contextualization_type='self_attention',
            prototype_contextualization_residual=True,
            normalize_for_self_interaction=True,
            normalize_for_routing=True,
            proxy_temperature=0.2,
            lambda_proxy=1.0,
            use_loss_proxy_text_exact=True,
            use_loss_ret_exact=False,
            lambda_ret_exact=1.0,
            ret_exact_temperature=None,
            use_loss_align=True,
            lambda_align=0.5,
            use_loss_diag=True,
            lambda_diag=0.25,
            use_loss_support=False,
            lambda_support=0.0,
            support_min=2.0,
            use_diversity_loss=True,
            diversity_loss_weight=0.05,
            use_balancing_loss=True,
            prototype_balance_loss_weight=0.1,
            temperature=0.07,
            prototype_dead_threshold=0.01,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def _build_head(self, **overrides):
        base = dict(
            input_dim=self.feature_dim,
            num_prototypes=self.num_prototypes,
            prototype_dim=self.feature_dim,
            projector_output_dim=4,
            projector_hidden_dim=12,
            projector_dropout=0.0,
            projector_type='mlp2',
            normalize_projector_outputs=True,
            prototype_init='normalized_random',
            prototype_init_path=None,
            prototype_init_hybrid_ratio=0.5,
            prototype_init_max_iters=50,
            prototype_init_tol=1e-4,
            prototype_init_seed=17,
            routing_type='cosine',
            routing_temperature=0.07,
            token_scoring_type='cosine',
            token_temperature=0.07,
            normalize_for_token_scoring=True,
            token_policy='content_only',
            special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
            error_on_empty_kept_tokens=True,
            contextualization_enabled=True,
            contextualization_type='self_attention',
            contextualization_residual=True,
            normalize_for_self_interaction=True,
            normalize_for_routing=True,
            num_classes=self.num_classes,
            proxy_temperature=0.2,
            lambda_proxy=1.0,
            use_loss_proxy_text_exact=True,
            use_loss_ret_exact=False,
            lambda_ret_exact=1.0,
            ret_exact_temperature=None,
            use_loss_align=True,
            lambda_align=0.5,
            use_loss_diag=True,
            lambda_diag=0.25,
            use_loss_support=False,
            support_loss_weight=0.0,
            support_min=2.0,
            use_diversity_loss=True,
            diversity_loss_weight=0.05,
            use_balance_loss=True,
            balance_loss_weight=0.1,
            contrastive_temperature_init=0.07,
            learnable_contrastive_temperature=False,
            dead_prototype_threshold=0.01,
        )
        base.update(overrides)
        return PrototypeConditionedTextHead(**base)

    def test_prototype_bank_shape_and_gradient(self):
        bank = PrototypeBank(num_prototypes=self.num_prototypes, prototype_dim=self.feature_dim)
        prototypes, debug = bank(return_debug=True)
        self.assertEqual(tuple(prototypes.shape), (self.num_prototypes, self.feature_dim))
        self.assertIn('raw_prototypes', debug)
        prototypes.sum().backward()
        self.assertIsNotNone(bank.prototypes.grad)

    def _write_tensor_file(self, tensor):
        handle = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
        handle.close()
        torch.save(tensor, handle.name)
        self.addCleanup(lambda: os.path.exists(handle.name) and os.remove(handle.name))
        return handle.name

    def _write_feature_checkpoint(self, payload):
        handle = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
        handle.close()
        torch.save(payload, handle.name)
        self.addCleanup(lambda: os.path.exists(handle.name) and os.remove(handle.name))
        return handle.name

    def _assert_unit_norm_rows(self, tensor, atol=1e-5):
        self.assertTrue(torch.isfinite(tensor).all())
        row_norms = tensor.norm(dim=-1)
        torch.testing.assert_close(row_norms, torch.ones_like(row_norms), atol=atol, rtol=atol)

    def _make_clustered_features(self):
        centers = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        feature_groups = []
        for center in centers:
            noise = 0.03 * torch.randn(24, centers.size(1))
            group = torch.nn.functional.normalize(center.unsqueeze(0) + noise, dim=-1)
            feature_groups.append(group)
        features = torch.cat(feature_groups, dim=0)
        return features, centers


    def test_init_mode_requires_data_matches_supported_surface(self):
        self.assertFalse(init_mode_requires_data('normalized_random'))
        self.assertFalse(init_mode_requires_data('orthogonal_normalized_random'))
        for init_mode in (
            'sampled_image_embeddings',
            'kmeans_centroids',
            'spherical_kmeans_centroids',
            'hybrid_spherical_kmeans_random',
        ):
            self.assertTrue(init_mode_requires_data(init_mode))

    def test_sampled_image_embeddings_accepts_in_memory_features_without_path(self):
        features = torch.randn(self.num_prototypes + 4, self.feature_dim)
        bank = PrototypeBank(
            num_prototypes=self.num_prototypes,
            prototype_dim=self.feature_dim,
            init_mode='sampled_image_embeddings',
            init_features=features,
            init_seed=11,
        )
        prototypes = bank.get_prototypes()
        self._assert_unit_norm_rows(prototypes)
        normalized_features = torch.nn.functional.normalize(features, dim=-1)
        best_match = (prototypes @ normalized_features.t()).max(dim=1).values
        self.assertTrue(bool(torch.all(best_match > 0.999)))
        self.assertTrue(bank.last_init_diagnostics['auto_train_image_fallback_used'])
        self.assertEqual(bank.last_init_diagnostics['feature_count'], features.size(0))
        self.assertEqual(bank.last_init_diagnostics['clustering_strategy'], 'sampled_image_embeddings')

    def test_kmeans_centroids_accepts_in_memory_features_without_path(self):
        features, true_centers = self._make_clustered_features()
        bank = PrototypeBank(
            num_prototypes=true_centers.size(0),
            prototype_dim=true_centers.size(1),
            init_mode='kmeans_centroids',
            init_features=features,
            init_seed=77,
            init_max_iters=40,
            init_tol=1e-5,
        )
        prototypes = bank.get_prototypes()
        self._assert_unit_norm_rows(prototypes)
        cosine_to_true = prototypes @ true_centers.t()
        best_match = cosine_to_true.max(dim=1).values
        self.assertTrue(bool(torch.all(best_match > 0.9)))
        self.assertTrue(bank.last_init_diagnostics['auto_train_image_fallback_used'])
        self.assertEqual(bank.last_init_diagnostics['clustering_strategy'], 'kmeans')
        self.assertFalse(bank.last_init_diagnostics['feature_normalized_for_init'])

    def test_spherical_kmeans_centroids_accepts_in_memory_features_without_path(self):
        features, true_centers = self._make_clustered_features()
        bank_a = PrototypeBank(
            num_prototypes=true_centers.size(0),
            prototype_dim=true_centers.size(1),
            init_mode='spherical_kmeans_centroids',
            init_features=features,
            init_seed=77,
            init_max_iters=40,
            init_tol=1e-5,
        )
        bank_b = PrototypeBank(
            num_prototypes=true_centers.size(0),
            prototype_dim=true_centers.size(1),
            init_mode='spherical_kmeans_centroids',
            init_features=features,
            init_seed=77,
            init_max_iters=40,
            init_tol=1e-5,
        )
        prototypes = bank_a.get_prototypes()
        torch.testing.assert_close(prototypes, bank_b.get_prototypes())
        self._assert_unit_norm_rows(prototypes)
        cosine_to_true = prototypes @ true_centers.t()
        best_match = cosine_to_true.max(dim=1).values
        self.assertTrue(bool(torch.all(best_match > 0.9)))
        self.assertTrue(bank_a.last_init_diagnostics['auto_train_image_fallback_used'])
        self.assertTrue(bank_a.last_init_diagnostics['feature_normalized_for_init'])

    def test_hybrid_spherical_kmeans_random_accepts_in_memory_features_without_path(self):
        features, _ = self._make_clustered_features()
        hybrid_bank = PrototypeBank(
            num_prototypes=6,
            prototype_dim=features.size(1),
            init_mode='hybrid_spherical_kmeans_random',
            init_features=features,
            init_seed=91,
            init_hybrid_ratio=0.5,
            init_max_iters=40,
            init_tol=1e-5,
        )
        spherical_bank = PrototypeBank(
            num_prototypes=3,
            prototype_dim=features.size(1),
            init_mode='spherical_kmeans_centroids',
            init_features=features,
            init_seed=91,
            init_max_iters=40,
            init_tol=1e-5,
        )
        prototypes = hybrid_bank.get_prototypes()
        self._assert_unit_norm_rows(prototypes)
        self.assertEqual(hybrid_bank.last_init_diagnostics['hybrid_data_count'], 3)
        self.assertEqual(hybrid_bank.last_init_diagnostics['hybrid_random_count'], 3)
        self.assertTrue(hybrid_bank.last_init_diagnostics['auto_train_image_fallback_used'])
        torch.testing.assert_close(prototypes[:3], spherical_bank.get_prototypes(), atol=1e-5, rtol=1e-5)

    def test_orthogonal_normalized_random_init_is_deterministic_and_normalized(self):
        bank_a = PrototypeBank(
            num_prototypes=6,
            prototype_dim=self.feature_dim,
            init_mode='orthogonal_normalized_random',
            init_seed=123,
        )
        bank_b = PrototypeBank(
            num_prototypes=6,
            prototype_dim=self.feature_dim,
            init_mode='orthogonal_normalized_random',
            init_seed=123,
        )
        torch.testing.assert_close(bank_a.get_prototypes(), bank_b.get_prototypes())
        self._assert_unit_norm_rows(bank_a.get_prototypes())
        self.assertEqual(bank_a.last_init_diagnostics['cluster_iterations'], 0)

    def test_spherical_kmeans_centroids_matches_synthetic_clusters_and_is_deterministic(self):
        features, true_centers = self._make_clustered_features()
        feature_path = self._write_feature_checkpoint({'image_features': features})
        bank_a = PrototypeBank(
            num_prototypes=true_centers.size(0),
            prototype_dim=true_centers.size(1),
            init_mode='spherical_kmeans_centroids',
            init_path=feature_path,
            init_seed=77,
            init_max_iters=40,
            init_tol=1e-5,
        )
        bank_b = PrototypeBank(
            num_prototypes=true_centers.size(0),
            prototype_dim=true_centers.size(1),
            init_mode='spherical_kmeans_centroids',
            init_path=feature_path,
            init_seed=77,
            init_max_iters=40,
            init_tol=1e-5,
        )
        prototypes = bank_a.get_prototypes()
        torch.testing.assert_close(prototypes, bank_b.get_prototypes())
        self._assert_unit_norm_rows(prototypes)
        cosine_to_true = prototypes @ true_centers.t()
        best_match = cosine_to_true.max(dim=1).values
        self.assertTrue(bool(torch.all(best_match > 0.9)))
        self.assertGreaterEqual(bank_a.last_init_diagnostics['cluster_iterations'], 1)
        self.assertEqual(bank_a.last_init_diagnostics['feature_count'], features.size(0))

    def test_hybrid_spherical_kmeans_random_combines_data_and_random_components(self):
        features, _ = self._make_clustered_features()
        feature_path = self._write_feature_checkpoint({'image_features': features})
        hybrid_bank = PrototypeBank(
            num_prototypes=6,
            prototype_dim=features.size(1),
            init_mode='hybrid_spherical_kmeans_random',
            init_path=feature_path,
            init_seed=91,
            init_hybrid_ratio=0.5,
            init_max_iters=40,
            init_tol=1e-5,
        )
        spherical_bank = PrototypeBank(
            num_prototypes=3,
            prototype_dim=features.size(1),
            init_mode='spherical_kmeans_centroids',
            init_path=feature_path,
            init_seed=91,
            init_max_iters=40,
            init_tol=1e-5,
        )
        prototypes = hybrid_bank.get_prototypes()
        self._assert_unit_norm_rows(prototypes)
        self.assertEqual(hybrid_bank.last_init_diagnostics['hybrid_data_count'], 3)
        self.assertEqual(hybrid_bank.last_init_diagnostics['hybrid_random_count'], 3)
        torch.testing.assert_close(prototypes[:3], spherical_bank.get_prototypes(), atol=1e-5, rtol=1e-5)
        cosine_between_parts = torch.abs(prototypes[3:] @ prototypes[:3].t())
        self.assertTrue(bool(torch.all(cosine_between_parts < 0.99)))

    def test_external_init_modes_remain_compatible(self):
        prototypes = torch.randn(self.num_prototypes, self.feature_dim)
        path = self._write_tensor_file(prototypes)
        for init_mode in ('sampled_image_embeddings', 'kmeans_centroids'):
            bank = PrototypeBank(
                num_prototypes=self.num_prototypes,
                prototype_dim=self.feature_dim,
                init_mode=init_mode,
                init_path=path,
            )
            expected = torch.nn.functional.normalize(prototypes, dim=-1)
            torch.testing.assert_close(bank.get_prototypes(), expected)

    def test_spherical_kmeans_requires_unambiguous_feature_tensor(self):
        ambiguous_path = self._write_feature_checkpoint(
            {
                'image_features': torch.randn(8, self.feature_dim),
                'text_features': torch.randn(8, self.feature_dim),
            }
        )
        with self.assertRaisesRegex(ValueError, 'Candidate tensor keys/shapes'):
            PrototypeBank(
                num_prototypes=3,
                prototype_dim=self.feature_dim,
                init_mode='spherical_kmeans_centroids',
                init_path=ambiguous_path,
                init_seed=3,
            )

    def test_contextualizer_off_returns_original(self):
        prototypes = torch.randn(self.num_prototypes, self.feature_dim)
        contextualizer = PrototypeContextualizer(enabled=False, contextualization_type='none')
        contextualized, debug = contextualizer(prototypes, return_debug=True)
        torch.testing.assert_close(contextualized, prototypes)
        torch.testing.assert_close(debug['contextualization_weights'], torch.eye(self.num_prototypes))

    def test_contextualization_type_none_disables_contextualizer_even_when_enabled(self):
        head = self._build_head(contextualization_type='none')
        context = head.get_prototype_context(return_debug=True)
        torch.testing.assert_close(context['prototypes'], context['contextualized_prototypes'])

    def test_router_probabilities_sum_to_one(self):
        router = Router(routing_type='cosine', temperature=0.07)
        image_embeddings = torch.randn(self.batch_size, self.feature_dim)
        prototypes = torch.randn(self.num_prototypes, self.feature_dim)
        alpha, _ = router(image_embeddings, prototypes, return_debug=True)
        torch.testing.assert_close(alpha.sum(dim=-1), torch.ones(self.batch_size), atol=1e-5, rtol=1e-5)

    def test_router_effective_support_varies_with_routing_concentration(self):
        router = Router(routing_type='dot', temperature=0.1, normalize=False)
        prototypes = torch.zeros(self.num_prototypes, self.feature_dim)
        prototypes[0, 0] = 10.0
        uniform_embedding = torch.zeros(1, self.feature_dim)
        peaked_embedding = torch.zeros(1, self.feature_dim)
        peaked_embedding[0, 0] = 1.0
        _, uniform_debug = router(uniform_embedding, prototypes, return_debug=True)
        _, peaked_debug = router(peaked_embedding, prototypes, return_debug=True)
        self.assertGreater(uniform_debug['routing_effective_support'], peaked_debug['routing_effective_support'])

    def test_token_mask_builder_policies(self):
        builder = TokenMaskBuilder(
            token_policy='content_only',
            special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
        )
        content_mask = builder.build(self.token_ids, attention_mask=self.attention_mask)
        expected = torch.tensor(
            [
                [False, True, True, False, False, False],
                [False, True, False, False, False, False],
                [False, True, True, True, False, False],
            ]
        )
        torch.testing.assert_close(content_mask, expected)

    def test_masked_token_pooler_masks_invalid_positions(self):
        pooler = MaskedTokenPooler()
        scores = torch.tensor([[4.0, 1.0, -2.0, 7.0]], dtype=torch.float32)
        states = torch.randn(1, 4, self.feature_dim)
        valid_mask = torch.tensor([[True, True, False, False]])
        pooled, beta, debug = pooler(scores, states, valid_mask, return_debug=True)
        self.assertEqual(beta[0, 2].item(), 0.0)
        self.assertEqual(beta[0, 3].item(), 0.0)
        torch.testing.assert_close(beta.sum(dim=-1), torch.ones(1), atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.isfinite(pooled).all())

    def test_projector_supports_linear_variant(self):
        projector = MLPProjector(input_dim=self.feature_dim, hidden_dim=12, output_dim=4, projector_type='linear')
        outputs, debug = projector(torch.randn(self.batch_size, self.feature_dim), return_debug=True)
        self.assertEqual(tuple(outputs.shape), (self.batch_size, 4))
        self.assertEqual(debug['projector_type'], 'linear')

    def test_similarity_preparation_matches_train_and_infer_when_normalized(self):
        losses = PrototypeLosses(
            temperature_init=1.0,
            normalize_embeddings=True,
            num_classes=self.num_classes,
            embedding_dim=2,
            proxy_temperature=0.5,
        )
        image_embeddings = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
        text_embeddings = torch.tensor([[4.0, 3.0], [2.0, 1.0]])
        image_norm = torch.nn.functional.normalize(image_embeddings, dim=-1)
        text_norm = torch.nn.functional.normalize(text_embeddings, dim=-1)
        expected_matrix = (text_norm @ image_norm.t()) * losses.get_logit_scale()
        expected_paired = (text_norm * image_norm).sum(dim=-1) * losses.get_logit_scale()
        torch.testing.assert_close(losses.compute_similarity_matrix(image_embeddings, text_embeddings), expected_matrix)
        torch.testing.assert_close(losses.compute_paired_similarity(image_embeddings, text_embeddings), expected_paired)

    def test_similarity_preparation_matches_train_and_infer_when_not_normalized(self):
        losses = PrototypeLosses(
            temperature_init=1.0,
            normalize_embeddings=False,
            num_classes=self.num_classes,
            embedding_dim=2,
            proxy_temperature=0.5,
        )
        image_embeddings = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
        text_embeddings = torch.tensor([[4.0, 3.0], [2.0, 1.0]])
        expected_matrix = (text_embeddings @ image_embeddings.t()) * losses.get_logit_scale()
        expected_paired = (text_embeddings * image_embeddings).sum(dim=-1) * losses.get_logit_scale()
        torch.testing.assert_close(losses.compute_similarity_matrix(image_embeddings, text_embeddings), expected_matrix)
        torch.testing.assert_close(losses.compute_paired_similarity(image_embeddings, text_embeddings), expected_paired)

    def test_loss_module_requires_class_labels(self):
        losses = PrototypeLosses(
            temperature_init=0.07,
            normalize_embeddings=True,
            num_classes=self.num_classes,
            embedding_dim=4,
            proxy_temperature=0.2,
        )
        with self.assertRaisesRegex(ValueError, 'pids must be provided'):
            losses(torch.randn(self.batch_size, 4), torch.randn(self.batch_size, 4), torch.randn(self.batch_size, 4), pids=None)

    def test_loss_module_rejects_learnable_temperature(self):
        with self.assertRaisesRegex(ValueError, 'Learnable retrieval logit scaling'):
            PrototypeLosses(temperature_init=0.07, learnable_temperature=True, num_classes=self.num_classes, embedding_dim=4)

    def test_support_loss_matches_ipr_formula(self):
        losses = PrototypeLosses(
            temperature_init=0.07,
            normalize_embeddings=True,
            num_classes=self.num_classes,
            embedding_dim=4,
            proxy_temperature=0.2,
            use_loss_support=True,
            support_loss_weight=0.5,
            support_min=2.0,
        )
        routing = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ], dtype=torch.float32)
        expected_support = torch.tensor([1.0, 2.0], dtype=torch.float32)
        expected_loss = torch.tensor(((2.0 - 1.0) ** 2 + 0.0) / 2.0, dtype=torch.float32)
        torch.testing.assert_close(losses.effective_support(routing), expected_support)
        torch.testing.assert_close(losses.support_loss(routing), expected_loss)

    def test_support_loss_is_zero_when_effective_support_meets_threshold(self):
        losses = PrototypeLosses(
            temperature_init=0.07,
            normalize_embeddings=True,
            num_classes=self.num_classes,
            embedding_dim=4,
            proxy_temperature=0.2,
            use_loss_support=True,
            support_loss_weight=0.5,
            support_min=2.0,
        )
        routing = torch.full((self.batch_size, self.num_prototypes), 1.0 / self.num_prototypes)
        self.assertEqual(losses.support_loss(routing).item(), 0.0)

    def test_support_loss_is_positive_for_peaked_routing(self):
        losses = PrototypeLosses(
            temperature_init=0.07,
            normalize_embeddings=True,
            num_classes=self.num_classes,
            embedding_dim=4,
            proxy_temperature=0.2,
            use_loss_support=True,
            support_loss_weight=0.5,
            support_min=2.0,
        )
        routing = torch.tensor([
            [0.97, 0.01, 0.01, 0.01, 0.0],
            [0.99, 0.01, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float32)
        self.assertGreater(losses.support_loss(routing).item(), 0.0)

    def test_loss_module_reports_amortized_components(self):
        losses = PrototypeLosses(
            temperature_init=0.07,
            normalize_embeddings=True,
            num_classes=self.num_classes,
            embedding_dim=4,
            proxy_temperature=0.2,
            lambda_proxy=1.0,
            lambda_align=0.5,
            lambda_diag=0.25,
            use_loss_support=True,
            support_loss_weight=0.2,
            support_min=2.0,
            use_diversity_loss=True,
            diversity_loss_weight=0.05,
            use_balance_loss=True,
            balance_loss_weight=0.1,
        )
        image_embeddings = torch.randn(self.batch_size, 4)
        surrogate_embeddings = torch.randn(self.batch_size, 4)
        exact_embeddings = torch.randn(self.batch_size, 4)
        prototypes = torch.randn(self.num_prototypes, self.feature_dim)
        routing = torch.softmax(torch.randn(self.batch_size, self.num_prototypes), dim=-1)
        outputs = losses(image_embeddings, surrogate_embeddings, exact_embeddings, pids=torch.tensor([0, 1, 2]), prototypes=prototypes, routing_weights=routing)
        expected_total = (
            outputs['loss_proxy_weighted']
            + outputs['loss_align_weighted']
            + outputs['loss_diag_weighted']
            + outputs['loss_support_weighted']
            + outputs['loss_diversity_weighted']
            + outputs['loss_balance_weighted']
        )
        torch.testing.assert_close(outputs['loss_total'], expected_total)

    def test_loss_module_exact_proxy_supervision_reaches_exact_embedding(self):
        losses = PrototypeLosses(
            temperature_init=0.07,
            normalize_embeddings=True,
            num_classes=self.num_classes,
            embedding_dim=4,
            proxy_temperature=0.2,
        )
        image_embeddings = torch.randn(self.batch_size, 4, requires_grad=True)
        surrogate_embeddings = torch.randn(self.batch_size, 4, requires_grad=True)
        exact_embeddings = torch.randn(self.batch_size, 4, requires_grad=True)
        outputs = losses(image_embeddings, surrogate_embeddings, exact_embeddings, pids=torch.tensor([0, 1, 2]))
        outputs['loss_total'].backward()
        self.assertIsNotNone(surrogate_embeddings.grad)
        self.assertIsNotNone(image_embeddings.grad)
        self.assertIsNotNone(exact_embeddings.grad)


    def test_loss_module_align_and_diag_can_be_disabled_explicitly(self):
        losses = PrototypeLosses(
            temperature_init=0.07,
            normalize_embeddings=True,
            num_classes=self.num_classes,
            embedding_dim=4,
            proxy_temperature=0.2,
            use_loss_align=False,
            lambda_align=0.5,
            use_loss_diag=False,
            lambda_diag=0.25,
        )
        outputs = losses(
            torch.randn(self.batch_size, 4),
            torch.randn(self.batch_size, 4),
            torch.randn(self.batch_size, 4),
            pids=torch.tensor([0, 1, 2]),
        )
        self.assertEqual(outputs['loss_align'].item(), 0.0)
        self.assertEqual(outputs['loss_diag'].item(), 0.0)
        self.assertEqual(outputs['loss_align_weighted'].item(), 0.0)
        self.assertEqual(outputs['loss_diag_weighted'].item(), 0.0)
        self.assertEqual(outputs['loss_support'].item(), 0.0)
        self.assertEqual(outputs['loss_support_weighted'].item(), 0.0)

    def test_loss_module_exact_proxy_can_be_disabled_to_restore_detached_anchor_behavior(self):
        losses = PrototypeLosses(
            temperature_init=0.07,
            normalize_embeddings=True,
            num_classes=self.num_classes,
            embedding_dim=4,
            proxy_temperature=0.2,
            use_loss_proxy_text_exact=False,
        )
        image_embeddings = torch.randn(self.batch_size, 4, requires_grad=True)
        surrogate_embeddings = torch.randn(self.batch_size, 4, requires_grad=True)
        exact_embeddings = torch.randn(self.batch_size, 4, requires_grad=True)
        outputs = losses(image_embeddings, surrogate_embeddings, exact_embeddings, pids=torch.tensor([0, 1, 2]))
        outputs['loss_total'].backward()
        self.assertIsNotNone(surrogate_embeddings.grad)
        self.assertIsNotNone(image_embeddings.grad)
        self.assertIsNone(exact_embeddings.grad)


    def test_exact_retrieval_loss_prefers_stronger_diagonal_logits(self):
        losses = PrototypeLosses(
            temperature_init=0.07,
            normalize_embeddings=True,
            num_classes=self.num_classes,
            embedding_dim=4,
            proxy_temperature=0.2,
            use_loss_proxy_image=False,
            use_loss_proxy_text=False,
            use_loss_proxy_text_exact=False,
            use_loss_align=False,
            use_loss_diag=False,
            use_loss_ret_exact_image=True,
            use_diversity_loss=False,
            use_balance_loss=False,
            balance_loss_weight=0.0,
        )
        strong = torch.tensor(
            [
                [5.0, 0.5, 0.1],
                [0.4, 4.5, 0.2],
                [0.3, 0.6, 4.0],
            ],
            dtype=torch.float32,
        )
        weak = torch.tensor(
            [
                [0.5, 1.4, 0.8],
                [1.1, 0.4, 1.2],
                [1.0, 0.9, 0.3],
            ],
            dtype=torch.float32,
        )
        self.assertLess(losses.exact_retrieval_loss(strong)['loss'].item(), losses.exact_retrieval_loss(weak)['loss'].item())
        self.assertLess(losses.exact_retrieval_loss(strong, transpose=True)['loss'].item(), losses.exact_retrieval_loss(weak, transpose=True)['loss'].item())

    def test_head_exact_retrieval_loss_exposes_pairwise_logits(self):
        head = self._build_head(use_loss_ret_exact=True)
        image_embeddings = torch.randn(self.batch_size, self.feature_dim, requires_grad=True)
        text_states = torch.randn(self.batch_size, self.seq_len, self.feature_dim, requires_grad=True)
        outputs = head(
            image_embeddings=image_embeddings,
            text_token_states=text_states,
            token_ids=self.token_ids,
            pids=torch.tensor([0, 1, 2], dtype=torch.long),
            attention_mask=self.attention_mask,
            special_token_positions=self.special_positions,
            return_debug=False,
        )
        self.assertEqual(tuple(outputs['exact_pairwise_logits'].shape), (self.batch_size, self.batch_size))
        self.assertIn('loss_ret_exact', outputs['losses'])
        self.assertTrue(torch.isfinite(outputs['losses']['loss_ret_exact']))

    def test_head_exact_retrieval_loss_drives_exact_branch_gradients(self):
        head = self._build_head(
            use_loss_proxy_image=False,
            use_loss_proxy_text=False,
            use_loss_proxy_text_exact=False,
            use_loss_align=False,
            use_loss_diag=False,
            use_loss_ret_exact=True,
            use_diversity_loss=False,
            use_balance_loss=False,
            balance_loss_weight=0.0,
        )
        image_embeddings = torch.randn(self.batch_size, self.feature_dim, requires_grad=True)
        text_states = torch.randn(self.batch_size, self.seq_len, self.feature_dim, requires_grad=True)
        outputs = head(
            image_embeddings=image_embeddings,
            text_token_states=text_states,
            token_ids=self.token_ids,
            pids=torch.tensor([0, 1, 2], dtype=torch.long),
            attention_mask=self.attention_mask,
            special_token_positions=self.special_positions,
            return_debug=False,
        )
        outputs['exact_pairwise_logits'].retain_grad()
        outputs['losses']['loss_total'].backward()
        self.assertIsNotNone(outputs['exact_pairwise_logits'].grad)
        text_projector_grad = sum(
            parameter.grad.detach().abs().sum().item()
            for name, parameter in head.named_parameters()
            if name.startswith('text_projector') and parameter.grad is not None
        )
        self.assertGreater(text_projector_grad, 0.0)

    def test_exact_retrieval_loss_supports_independent_image_and_text_weights(self):
        losses = PrototypeLosses(
            temperature_init=0.07,
            normalize_embeddings=True,
            num_classes=self.num_classes,
            embedding_dim=4,
            proxy_temperature=0.2,
            use_loss_proxy_image=False,
            use_loss_proxy_text=False,
            use_loss_proxy_text_exact=False,
            use_loss_align=False,
            use_loss_diag=False,
            use_loss_ret_exact_image=True,
            use_loss_ret_exact_text=True,
            lambda_ret_exact_image=2.0,
            lambda_ret_exact_text=0.5,
            use_diversity_loss=False,
            use_balance_loss=False,
            balance_loss_weight=0.0,
        )
        logits = torch.tensor(
            [
                [3.0, 0.2, 0.1],
                [0.3, 2.5, 0.4],
                [0.1, 0.4, 2.8],
            ],
            dtype=torch.float32,
        )
        image_embeddings = torch.randn(self.batch_size, 4)
        surrogate_embeddings = torch.randn(self.batch_size, 4)
        exact_embeddings = torch.randn(self.batch_size, 4)
        outputs = losses(
            image_embeddings,
            surrogate_embeddings,
            exact_embeddings,
            pids=torch.tensor([0, 1, 2]),
            exact_pairwise_logits=logits,
        )
        expected = (2.0 * outputs['loss_ret_exact_image']) + (0.5 * outputs['loss_ret_exact_text'])
        self.assertAlmostEqual(outputs['loss_ret_exact_weighted'].item(), expected.item(), places=6)

    def test_loss_module_proxy_parameters_exist(self):
        losses = PrototypeLosses(
            temperature_init=0.07,
            normalize_embeddings=True,
            num_classes=self.num_classes,
            embedding_dim=4,
            proxy_temperature=0.2,
        )
        self.assertEqual(tuple(losses.class_proxies.shape), (self.num_classes, 4))

    def test_build_head_returns_prototype_conditioned_head(self):
        head = build_prototype_head(self._build_args(), input_dim=self.feature_dim, num_classes=self.num_classes)
        self.assertIsInstance(head, PrototypeConditionedTextHead)

    def test_build_text_basis_bank_shape_and_masks(self):
        head = self._build_head()
        text_states = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        context = head.get_prototype_context(return_debug=False)
        basis_outputs = head.build_text_basis_bank(
            text_token_states=text_states,
            token_ids=self.token_ids,
            contextualized_prototypes=context['contextualized_prototypes'],
            attention_mask=self.attention_mask,
            special_token_positions=self.special_positions,
            return_debug=True,
        )
        self.assertEqual(tuple(basis_outputs['basis_bank'].shape), (self.batch_size, self.num_prototypes, self.feature_dim))
        keep_mask = basis_outputs['token_keep_mask']
        basis_weights = basis_outputs['basis_token_weights']
        for batch_index in range(self.batch_size):
            masked_weights = basis_weights[batch_index][:, ~keep_mask[batch_index]]
            if masked_weights.numel() > 0:
                torch.testing.assert_close(masked_weights, torch.zeros_like(masked_weights))
        self.assertTrue(torch.isfinite(basis_outputs['basis_bank']).all())

    def test_surrogate_reconstruction_matches_manual_weighted_sum(self):
        head = self._build_head()
        alpha = torch.tensor([[0.7, 0.3], [0.1, 0.9]], dtype=torch.float32)
        basis_bank = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        surrogate = head.reconstruct_surrogate_text(alpha, basis_bank)
        expected = torch.stack([
            0.7 * basis_bank[0, 0] + 0.3 * basis_bank[0, 1],
            0.1 * basis_bank[1, 0] + 0.9 * basis_bank[1, 1],
        ])
        torch.testing.assert_close(surrogate, expected)

    def test_basis_bank_path_avoids_pairwise_batch_expansion(self):
        head = self._build_head()
        text_states = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        context = head.get_prototype_context(return_debug=False)
        basis_outputs = head.build_text_basis_bank(
            text_token_states=text_states,
            token_ids=self.token_ids,
            contextualized_prototypes=context['contextualized_prototypes'],
            attention_mask=self.attention_mask,
            special_token_positions=self.special_positions,
            return_debug=True,
        )
        for value in basis_outputs.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                self.assertFalse(value.shape[:2] == (self.batch_size, self.batch_size))

    def test_prototype_head_end_to_end_shapes_losses_and_masks(self):
        head = self._build_head()
        image_embeddings = torch.randn(self.batch_size, self.feature_dim, requires_grad=True)
        text_states = torch.randn(self.batch_size, self.seq_len, self.feature_dim, requires_grad=True)
        outputs = head(
            image_embeddings=image_embeddings,
            text_token_states=text_states,
            token_ids=self.token_ids,
            pids=torch.tensor([0, 1, 2], dtype=torch.long),
            attention_mask=self.attention_mask,
            special_token_positions=self.special_positions,
            return_debug=True,
        )
        self.assertEqual(tuple(outputs['routing_weights'].shape), (self.batch_size, self.num_prototypes))
        self.assertEqual(tuple(outputs['basis_bank'].shape), (self.batch_size, self.num_prototypes, self.feature_dim))
        self.assertEqual(tuple(outputs['surrogate_pooled_text'].shape), (self.batch_size, self.feature_dim))
        self.assertEqual(tuple(outputs['exact_pooled_text'].shape), (self.batch_size, self.feature_dim))
        self.assertEqual(tuple(outputs['surrogate_text_projected'].shape), (self.batch_size, 4))
        self.assertEqual(tuple(outputs['exact_text_projected'].shape), (self.batch_size, 4))
        self.assertIn('loss_proxy', outputs['losses'])
        self.assertIn('loss_align', outputs['losses'])
        self.assertIn('loss_diag', outputs['losses'])
        self.assertTrue(torch.isfinite(outputs['losses']['loss_total']))
        outputs['losses']['loss_total'].backward()
        self.assertIsNotNone(head.prototype_bank.prototypes.grad)
        self.assertIsNotNone(head.losses.class_proxies.grad)

    def test_head_exact_proxy_supervision_reaches_exact_text_embedding(self):
        head = self._build_head()
        image_embeddings = torch.randn(self.batch_size, self.feature_dim, requires_grad=True)
        text_states = torch.randn(self.batch_size, self.seq_len, self.feature_dim, requires_grad=True)
        outputs = head(
            image_embeddings=image_embeddings,
            text_token_states=text_states,
            token_ids=self.token_ids,
            pids=torch.tensor([0, 1, 2], dtype=torch.long),
            attention_mask=self.attention_mask,
            special_token_positions=self.special_positions,
            return_debug=False,
        )
        outputs['surrogate_text_projected'].retain_grad()
        outputs['exact_text_projected'].retain_grad()
        outputs['losses']['loss_total'].backward()
        self.assertIsNotNone(outputs['surrogate_text_projected'].grad)
        self.assertIsNotNone(outputs['exact_text_projected'].grad)


    def test_head_exact_proxy_can_be_disabled_to_restore_detached_exact_path(self):
        head = self._build_head(use_loss_proxy_text_exact=False)
        image_embeddings = torch.randn(self.batch_size, self.feature_dim, requires_grad=True)
        text_states = torch.randn(self.batch_size, self.seq_len, self.feature_dim, requires_grad=True)
        outputs = head(
            image_embeddings=image_embeddings,
            text_token_states=text_states,
            token_ids=self.token_ids,
            pids=torch.tensor([0, 1, 2], dtype=torch.long),
            attention_mask=self.attention_mask,
            special_token_positions=self.special_positions,
            return_debug=False,
        )
        outputs['surrogate_text_projected'].retain_grad()
        outputs['exact_text_projected'].retain_grad()
        outputs['losses']['loss_total'].backward()
        self.assertIsNotNone(outputs['surrogate_text_projected'].grad)
        self.assertIsNone(outputs['exact_text_projected'].grad)

    def test_compute_approximate_pairwise_similarity_runs(self):
        head = self._build_head()
        image_projected = torch.randn(2, 4)
        routing_weights = torch.softmax(torch.randn(2, self.num_prototypes), dim=-1)
        basis_bank = torch.randn(3, self.num_prototypes, self.feature_dim)
        similarity = head.compute_approximate_pairwise_similarity(image_projected, routing_weights, basis_bank, image_chunk_size=1, text_chunk_size=2)
        self.assertEqual(tuple(similarity.shape), (3, 2))
        self.assertTrue(torch.isfinite(similarity).all())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()




