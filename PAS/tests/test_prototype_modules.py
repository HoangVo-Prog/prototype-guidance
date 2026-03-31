import os
import sys
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
            use_diversity_loss=True,
            diversity_loss_weight=0.05,
            use_balancing_loss=True,
            prototype_balance_loss_weight=0.1,
            temperature=0.07,
            learn_logit_scale=True,
            prototype_dead_threshold=0.01,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def _build_head(self, **overrides):
        return PrototypeConditionedTextHead(
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
            use_diversity_loss=True,
            diversity_loss_weight=0.05,
            use_balance_loss=True,
            balance_loss_weight=0.1,
            contrastive_temperature_init=0.07,
            learnable_contrastive_temperature=True,
            dead_prototype_threshold=0.01,
            **overrides,
        )

    def test_prototype_bank_shape_and_gradient(self):
        bank = PrototypeBank(num_prototypes=self.num_prototypes, prototype_dim=self.feature_dim)
        prototypes, debug = bank(return_debug=True)
        self.assertEqual(tuple(prototypes.shape), (self.num_prototypes, self.feature_dim))
        self.assertIn('raw_prototypes', debug)
        self.assertEqual(debug['prototype_init_mode'], 'normalized_random')
        loss = prototypes.sum()
        loss.backward()
        self.assertIsNotNone(bank.prototypes.grad)
        self.assertEqual(tuple(bank.prototypes.grad.shape), (self.num_prototypes, self.feature_dim))

    def test_contextualizer_off_returns_original(self):
        prototypes = torch.randn(self.num_prototypes, self.feature_dim)
        contextualizer = PrototypeContextualizer(enabled=False, contextualization_type='none')
        contextualized, debug = contextualizer(prototypes, return_debug=True)
        torch.testing.assert_close(contextualized, prototypes)
        torch.testing.assert_close(debug['contextualization_weights'], torch.eye(self.num_prototypes))

    def test_contextualizer_weights_sum_to_one(self):
        prototypes = torch.randn(self.num_prototypes, self.feature_dim)
        contextualizer = PrototypeContextualizer(enabled=True, contextualization_type='self_attention')
        contextualized, debug = contextualizer(prototypes, return_debug=True)
        self.assertEqual(tuple(contextualized.shape), (self.num_prototypes, self.feature_dim))
        row_sums = debug['contextualization_weights'].sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isfinite(contextualized).all())

    def test_contextualizer_reports_fixed_single_step_contract(self):
        prototypes = torch.randn(self.num_prototypes, self.feature_dim)
        contextualizer = PrototypeContextualizer(enabled=True, contextualization_type='self_attention')
        contextualized, debug = contextualizer(prototypes, return_debug=True)
        self.assertEqual(tuple(contextualized.shape), (self.num_prototypes, self.feature_dim))
        self.assertEqual(debug['contextualization_num_layers'], 1)
        self.assertTrue(torch.isfinite(contextualized).all())


    def test_router_probabilities_sum_to_one(self):
        router = Router(routing_type='cosine', temperature=0.07)
        image_embeddings = torch.randn(self.batch_size, self.feature_dim)
        prototypes = torch.randn(self.num_prototypes, self.feature_dim)
        alpha, debug = router(image_embeddings, prototypes, return_debug=True)
        self.assertEqual(tuple(alpha.shape), (self.batch_size, self.num_prototypes))
        torch.testing.assert_close(alpha.sum(dim=-1), torch.ones(self.batch_size), atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isfinite(debug['routing_similarity']).all())
        self.assertTrue(torch.isfinite(debug['alpha_logits']).all())

    def test_aggregator_changes_with_routing_weights(self):
        prototypes = torch.arange(self.num_prototypes * self.feature_dim, dtype=torch.float32).view(self.num_prototypes, self.feature_dim)
        aggregator = PrototypeAggregator()
        alpha_a = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        alpha_b = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
        summary_a = aggregator(alpha_a, prototypes)
        summary_b = aggregator(alpha_b, prototypes)
        self.assertFalse(torch.allclose(summary_a, summary_b))

    def test_token_scorer_shape_and_query_sensitivity(self):
        scorer = TokenScorer(scoring_type='cosine', temperature=0.07)
        token_states = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        query_a = torch.randn(self.batch_size, self.feature_dim)
        query_b = query_a + 0.5
        scores_a = scorer(query_a, token_states)
        scores_b = scorer(query_b, token_states)
        self.assertEqual(tuple(scores_a.shape), (self.batch_size, self.seq_len))
        self.assertFalse(torch.allclose(scores_a, scores_b))
        self.assertTrue(torch.isfinite(scores_a).all())

    def test_token_mask_builder_policies(self):
        content_builder = TokenMaskBuilder(
            token_policy='content_only',
            special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
        )
        content_mask, content_debug = content_builder.build(self.token_ids, attention_mask=self.attention_mask, return_debug=True)
        expected_content = torch.tensor(
            [
                [False, True, True, False, False, False],
                [False, True, False, False, False, False],
                [False, True, True, True, False, False],
            ]
        )
        torch.testing.assert_close(content_mask, expected_content)
        torch.testing.assert_close(content_debug['token_valid_mask'], self.attention_mask)
        torch.testing.assert_close(content_debug['token_keep_mask'], expected_content)

        all_builder = TokenMaskBuilder(
            token_policy='content_plus_special',
            special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
        )
        all_mask = all_builder.build(self.token_ids, attention_mask=self.attention_mask)
        torch.testing.assert_close(all_mask, self.attention_mask)

        eos_builder = TokenMaskBuilder(
            token_policy='eos_only',
            special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
        )
        eos_mask = eos_builder.build(self.token_ids, attention_mask=self.attention_mask)
        expected_eos = torch.tensor(
            [
                [False, False, False, True, False, False],
                [False, False, True, False, False, False],
                [False, False, False, False, True, False],
            ]
        )
        torch.testing.assert_close(eos_mask, expected_eos)

    def test_token_mask_builder_uses_configured_ids_not_positions_or_argmax(self):
        token_ids = torch.tensor([[11, 12, 7, 13, 5, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[True, True, True, True, True, False]])
        builder = TokenMaskBuilder(
            token_policy='content_only',
            special_token_ids={'bos_token_id': 7, 'eos_token_id': 5, 'pad_token_id': 0},
        )
        token_keep_mask = builder.build(token_ids, attention_mask=attention_mask)
        expected = torch.tensor([[True, True, False, True, False, False]])
        torch.testing.assert_close(token_keep_mask, expected)

        eos_only_builder = TokenMaskBuilder(
            token_policy='eos_only',
            special_token_ids={'bos_token_id': 7, 'eos_token_id': 5, 'pad_token_id': 0},
        )
        eos_only_mask = eos_only_builder.build(token_ids, attention_mask=attention_mask)
        expected_eos = torch.tensor([[False, False, False, False, True, False]])
        torch.testing.assert_close(eos_only_mask, expected_eos)

    def test_eos_only_keeps_only_final_valid_eos(self):
        token_ids = torch.tensor([[49406, 10, 49407, 11, 49407, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[True, True, True, True, True, False]])
        builder = TokenMaskBuilder(
            token_policy='eos_only',
            special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
        )
        token_keep_mask, debug = builder.build(token_ids, attention_mask=attention_mask, return_debug=True)
        expected = torch.tensor([[False, False, False, False, True, False]])
        torch.testing.assert_close(token_keep_mask, expected)
        torch.testing.assert_close(debug['special_token_positions']['eos'], torch.tensor([4]))

    def test_build_valid_mask_uses_final_eos_when_attention_mask_is_missing(self):
        token_ids = torch.tensor([[49406, 10, 49407, 11, 49407, 0]], dtype=torch.long)
        builder = TokenMaskBuilder(
            token_policy='content_only',
            special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
        )
        token_valid_mask = builder.build_valid_mask(token_ids)
        expected = torch.tensor([[True, True, True, True, True, False]])
        torch.testing.assert_close(token_valid_mask, expected)

    def test_token_mask_builder_requires_explicit_special_token_metadata(self):
        token_ids = torch.tensor([[49406, 10, 49407, 0]], dtype=torch.long)
        builder = TokenMaskBuilder(
            token_policy='content_only',
            special_token_ids={'pad_token_id': 0},
        )
        with self.assertRaises(ValueError):
            builder.build(token_ids, attention_mask=torch.tensor([[True, True, True, False]]))

    def test_token_mask_builder_raises_when_no_content_token_remains(self):
        token_ids = torch.tensor([[49406, 49407, 0, 0]], dtype=torch.long)
        builder = TokenMaskBuilder(
            token_policy='content_only',
            special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
        )
        with self.assertRaises(ValueError):
            builder.build(token_ids, attention_mask=token_ids.ne(0))

    def test_token_mask_builder_requires_required_special_tokens_per_sample(self):
        token_ids = torch.tensor(
            [
                [49406, 10, 49407, 0],
                [20, 21, 22, 0],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [
                [True, True, True, False],
                [True, True, True, False],
            ]
        )
        builder = TokenMaskBuilder(
            token_policy='content_only',
            special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
        )
        with self.assertRaises(ValueError):
            builder.build(token_ids, attention_mask=attention_mask)

    def test_masked_token_pooler_masks_invalid_positions(self):
        pooler = MaskedTokenPooler()
        scores = torch.tensor([[4.0, 1.0, -2.0, 7.0]], dtype=torch.float32)
        states = torch.randn(1, 4, self.feature_dim)
        valid_mask = torch.tensor([[True, True, False, False]])
        pooled, beta, debug = pooler(scores, states, valid_mask, return_debug=True)
        self.assertEqual(tuple(pooled.shape), (1, self.feature_dim))
        self.assertEqual(tuple(beta.shape), (1, 4))
        self.assertEqual(beta[0, 2].item(), 0.0)
        self.assertEqual(beta[0, 3].item(), 0.0)
        torch.testing.assert_close(beta.sum(dim=-1), torch.ones(1), atol=1e-6, rtol=1e-6)
        self.assertEqual(debug['beta_logits_masked'][0, 2].item(), float('-inf'))
        self.assertEqual(debug['beta_logits_masked'][0, 3].item(), float('-inf'))

    def test_masked_token_pooler_stable_with_large_logits(self):
        pooler = MaskedTokenPooler()
        scores = torch.tensor([[1000.0, 999.0, -1000.0]], dtype=torch.float32)
        states = torch.randn(1, 3, self.feature_dim)
        valid_mask = torch.tensor([[True, True, False]])
        pooled, beta = pooler(scores, states, valid_mask)
        self.assertTrue(torch.isfinite(beta).all())
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertEqual(beta[0, 2].item(), 0.0)

    def test_projector_receives_gradients_and_exposes_raw_outputs(self):
        projector = MLPProjector(input_dim=self.feature_dim, hidden_dim=12, output_dim=4)
        inputs = torch.randn(self.batch_size, self.feature_dim, requires_grad=True)
        outputs, debug = projector(inputs, return_debug=True)
        loss = outputs.sum()
        loss.backward()
        self.assertIsNotNone(projector.net[0].weight.grad)
        self.assertIsNotNone(projector.net[3].weight.grad)
        self.assertIn('projected_features_raw', debug)

    def test_projector_supports_linear_variant(self):
        projector = MLPProjector(input_dim=self.feature_dim, hidden_dim=12, output_dim=4, projector_type='linear')
        outputs, debug = projector(torch.randn(self.batch_size, self.feature_dim), return_debug=True)
        self.assertEqual(tuple(outputs.shape), (self.batch_size, 4))
        self.assertEqual(debug['projector_type'], 'linear')

    def test_loss_module_reports_raw_and_weighted_components(self):
        losses = PrototypeLosses(
            temperature_init=0.07,
            use_diversity_loss=True,
            diversity_loss_weight=0.05,
            use_balance_loss=True,
            balance_loss_weight=0.1,
        )
        image_embeddings = torch.randn(self.batch_size, 4)
        text_embeddings = torch.randn(self.batch_size, 4)
        prototypes = torch.randn(self.num_prototypes, self.feature_dim)
        routing = torch.softmax(torch.randn(self.batch_size, self.num_prototypes), dim=-1)
        outputs = losses(image_embeddings, text_embeddings, prototypes=prototypes, routing_weights=routing, return_debug=True)
        expected_total = outputs['loss_infonce'] + (0.05 * outputs['loss_diversity']) + (0.1 * outputs['loss_balance'])
        torch.testing.assert_close(outputs['loss_total'], expected_total)
        torch.testing.assert_close(outputs['loss_diversity_weighted'], 0.05 * outputs['loss_diversity'])
        torch.testing.assert_close(outputs['loss_balance_weighted'], 0.1 * outputs['loss_balance'])

    def test_loss_module_can_disable_logit_scale_learning(self):
        losses = PrototypeLosses(temperature_init=0.07, learnable_temperature=False)
        self.assertEqual(list(losses.named_parameters()), [])
        self.assertTrue(hasattr(losses, 'logit_scale'))

    def test_build_head_requires_balance_flag_and_weight_to_agree(self):
        args = self._build_args(use_balancing_loss=False, prototype_balance_loss_weight=0.1)
        with self.assertRaises(ValueError):
            build_prototype_head(args, input_dim=self.feature_dim)

    def test_build_head_returns_prototype_conditioned_head(self):
        head = build_prototype_head(self._build_args(), input_dim=self.feature_dim)
        self.assertIsInstance(head, PrototypeConditionedTextHead)

    def test_prototype_head_end_to_end_shapes_losses_and_masks(self):
        head = self._build_head()
        image_embeddings = torch.randn(self.batch_size, self.feature_dim, requires_grad=True)
        text_states = torch.randn(self.batch_size, self.seq_len, self.feature_dim, requires_grad=True)
        outputs = head(
            image_embeddings=image_embeddings,
            text_token_states=text_states,
            token_ids=self.token_ids,
            attention_mask=self.attention_mask,
            special_token_positions=self.special_positions,
            return_debug=True,
        )
        self.assertEqual(tuple(outputs['routing_weights'].shape), (self.batch_size, self.num_prototypes))
        self.assertEqual(tuple(outputs['summary'].shape), (self.batch_size, self.feature_dim))
        self.assertEqual(tuple(outputs['token_scores'].shape), (self.batch_size, self.seq_len))
        self.assertEqual(tuple(outputs['token_weights'].shape), (self.batch_size, self.seq_len))
        self.assertEqual(tuple(outputs['pooled_text'].shape), (self.batch_size, self.feature_dim))
        self.assertEqual(tuple(outputs['image_projected'].shape), (self.batch_size, 4))
        self.assertEqual(tuple(outputs['image_projected_raw'].shape), (self.batch_size, 4))
        self.assertEqual(tuple(outputs['text_projected'].shape), (self.batch_size, 4))
        self.assertEqual(tuple(outputs['text_projected_raw'].shape), (self.batch_size, 4))
        torch.testing.assert_close(outputs['routing_weights'].sum(dim=-1), torch.ones(self.batch_size), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(outputs['token_weights'].sum(dim=-1), torch.ones(self.batch_size), atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.equal(outputs['token_weights'][~outputs['token_keep_mask']], torch.zeros_like(outputs['token_weights'][~outputs['token_keep_mask']])))
        self.assertTrue(torch.equal(outputs['beta'][~outputs['token_keep_mask']], torch.zeros_like(outputs['beta'][~outputs['token_keep_mask']])))
        self.assertTrue(torch.isfinite(outputs['losses']['loss_total']))
        self.assertIn('prototype_usage_entropy', outputs['metrics'])
        self.assertIn('prototype_usage', outputs['debug'])
        self.assertIn('token_valid_mask', outputs['debug'])
        self.assertIn('token_keep_mask', outputs['debug'])
        self.assertIn('beta_logits_masked', outputs['debug'])
        outputs['losses']['loss_total'].backward()
        self.assertIsNotNone(head.prototype_bank.prototypes.grad)
        self.assertIsNotNone(head.image_projector.net[0].weight.grad)
        self.assertIsNotNone(head.text_projector.net[0].weight.grad)

    def test_frozen_prototypes_do_not_receive_gradients(self):
        head = self._build_head()
        head.prototype_bank.prototypes.requires_grad_(False)
        image_embeddings = torch.randn(self.batch_size, self.feature_dim, requires_grad=True)
        text_states = torch.randn(self.batch_size, self.seq_len, self.feature_dim, requires_grad=True)
        outputs = head(
            image_embeddings=image_embeddings,
            text_token_states=text_states,
            token_ids=self.token_ids,
            attention_mask=self.attention_mask,
            special_token_positions=self.special_positions,
            return_debug=False,
        )
        outputs['losses']['loss_total'].backward()
        self.assertIsNone(head.prototype_bank.prototypes.grad)
        self.assertIsNotNone(head.image_projector.net[0].weight.grad)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()






