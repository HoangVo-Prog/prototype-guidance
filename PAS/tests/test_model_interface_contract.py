import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment-dependent
    torch = None
    nn = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

IMPORT_ERROR = None
if torch is not None:
    try:
        from model.build import PASModel
        from model.interfaces import EncoderOutput
        from utils.metrics import Evaluator
    except Exception as exc:  # pragma: no cover - environment-dependent
        PASModel = None
        EncoderOutput = None
        Evaluator = None
        IMPORT_ERROR = exc


if nn is not None:
    class DummyCLIPBackbone(nn.Module):
        def __init__(self, embed_dim=8, image_shape=(3, 4, 4), vocab_size=50010, text_length=77):
            super().__init__()
            self.embed_dim = embed_dim
            self.image_input_dim = image_shape[0] * image_shape[1] * image_shape[2]
            self.visual = nn.Linear(self.image_input_dim, embed_dim)
            self.transformer = nn.Linear(embed_dim, embed_dim)
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.positional_embedding = nn.Parameter(torch.zeros(text_length, embed_dim))
            self.ln_final = nn.LayerNorm(embed_dim)
            self.text_projection = nn.Parameter(torch.eye(embed_dim))

        def encode_image_intermediates(self, image, return_all=False, average_attn_weights=True):
            del return_all, average_attn_weights
            flat = image.view(image.size(0), -1).float()
            cls_token = self.visual(flat)
            aux_token = torch.tanh(cls_token)
            tokens = torch.stack([cls_token, aux_token], dim=1)
            return {
                'projected_tokens': tokens,
                'pre_projection_tokens': tokens,
                'attention_weights': None,
            }

        def encode_text_intermediates(self, text, return_all=False, average_attn_weights=True):
            del return_all, average_attn_weights
            embedded = self.token_embedding(text.long())
            positional = self.positional_embedding[:text.size(1)].unsqueeze(0)
            hidden = self.transformer(embedded + positional)
            hidden = self.ln_final(hidden)
            projected = hidden @ self.text_projection
            return {
                'projected_tokens': projected,
                'pre_projection_tokens': hidden,
                'attention_weights': None,
            }
else:
    DummyCLIPBackbone = None


@unittest.skipUnless(
    torch is not None and PASModel is not None and EncoderOutput is not None and Evaluator is not None,
    f'Interface tests require torch and repo runtime imports: {IMPORT_ERROR}',
)
class ModelInterfaceContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.patch_build = mock.patch(
            'model.build.build_CLIP_from_openai_pretrained',
            side_effect=lambda *args, **kwargs: (DummyCLIPBackbone(), {'embed_dim': 8, 'vision_layers': 1, 'transformer_width': 8}),
        )
        self.patch_build.start()
        self.addCleanup(self.patch_build.stop)

        self.batch_size = 3
        self.seq_len = 6
        self.num_prototypes = 4
        self.prototype_dim = 8
        self.projection_dim = 4
        self.images = torch.randn(self.batch_size, 3, 4, 4)
        self.caption_ids = torch.tensor(
            [
                [49406, 11, 12, 49407, 0, 0],
                [49406, 21, 49407, 0, 0, 0],
                [49406, 31, 32, 33, 49407, 0],
            ],
            dtype=torch.long,
        )
        self.pids = torch.tensor([0, 1, 2], dtype=torch.long)
        self.expected_eos = torch.tensor([3, 2, 4], dtype=torch.long)
        self.text_loader = [
            (
                torch.tensor([0, 1], dtype=torch.long),
                torch.tensor(
                    [
                        [49406, 11, 12, 49407, 0, 0],
                        [49406, 21, 49407, 0, 0, 0],
                    ],
                    dtype=torch.long,
                ),
            ),
            (
                torch.tensor([2], dtype=torch.long),
                torch.tensor(
                    [[49406, 31, 32, 33, 49407, 0]],
                    dtype=torch.long,
                ),
            ),
        ]
        self.image_loader = [
            (torch.tensor([0, 1], dtype=torch.long), self.images[:2]),
            (torch.tensor([2], dtype=torch.long), self.images[2:]),
        ]

    def _build_args(self, **overrides):
        base = dict(
            pretrain_choice='ViT-B/16',
            img_size=(4, 4),
            stride_size=1,
            model_name='PAS',
            model_variant='pas_contract',
            image_backbone='dummy_visual',
            text_backbone='dummy_text',
            embedding_dim=8,
            projection_dim=self.projection_dim,
            projector_hidden_dim=8,
            projector_dropout=0.0,
            projector_type='mlp2',
            normalize_projector_outputs=True,
            backbone_precision='fp32',
            prototype_precision='fp32',
            temperature=0.07,
            proxy_temperature=0.2,
            lambda_proxy=1.0,
            use_loss_proxy_text_exact=True,
            lambda_align=0.5,
            lambda_diag=0.25,
            use_loss_support=False,
            lambda_support=0.0,
            support_min=2.0,
            text_length=77,
            vocab_size=50010,
            use_prototype_bank=True,
            use_image_conditioned_pooling=True,
            use_prototype_contextualization=True,
            prototype_contextualization_enabled=True,
            return_debug_outputs=True,
            prototype_num_prototypes=self.num_prototypes,
            prototype_dim=self.prototype_dim,
            prototype_init='normalized_random',
            prototype_init_path=None,
            prototype_routing_type='cosine',
            prototype_temperature=0.07,
            prototype_contextualization_type='self_attention',
            prototype_contextualization_residual=True,
            normalize_for_self_interaction=True,
            normalize_for_routing=True,
            use_balancing_loss=True,
            prototype_balance_loss_weight=0.1,
            prototype_dead_threshold=0.005,
            use_diversity_loss=True,
            diversity_loss_weight=0.01,
            token_policy='content_only',
            token_scoring_type='cosine',
            normalize_for_token_scoring=True,
            token_pooling_temperature=0.07,
            special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
            error_on_empty_kept_tokens=True,
            freeze_image_backbone=True,
            freeze_text_backbone=True,
            prototype_eval_image_chunk_size=2,
            prototype_eval_text_chunk_size=2,
            retrieval_scorer='exact',
            training=True,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def _build_model(self, **overrides):
        return PASModel(self._build_args(**overrides), num_classes=3)

    def test_extract_image_features_contract(self):
        model = self._build_model()
        outputs = model.extract_image_features(self.images)
        self.assertIsInstance(outputs, EncoderOutput)
        self.assertEqual(tuple(outputs.projected_tokens.shape), (self.batch_size, 2, self.prototype_dim))
        self.assertEqual(tuple(outputs.projected_pooled.shape), (self.batch_size, self.prototype_dim))
        self.assertEqual(tuple(outputs.pre_projection_tokens.shape), (self.batch_size, 2, self.prototype_dim))
        self.assertEqual(tuple(outputs.pre_projection_pooled.shape), (self.batch_size, self.prototype_dim))
        self.assertEqual(outputs.pooling_mode, 'cls')
        self.assertIsNone(outputs.token_mask)
        self.assertEqual(outputs.special_token_positions, {})
        self.assertEqual(outputs.metadata['encoder'], 'image')
        self.assertEqual(outputs.metadata['backbone'], 'dummy_visual')
        torch.testing.assert_close(outputs.projected_pooled, outputs.projected_tokens[:, 0, :])
        torch.testing.assert_close(outputs.pre_projection_pooled, outputs.pre_projection_tokens[:, 0, :])

    def test_extract_text_features_contract(self):
        model = self._build_model()
        outputs = model.extract_text_features(self.caption_ids)
        self.assertIsInstance(outputs, EncoderOutput)
        self.assertEqual(tuple(outputs.projected_tokens.shape), (self.batch_size, self.seq_len, self.prototype_dim))
        self.assertEqual(tuple(outputs.projected_pooled.shape), (self.batch_size, self.prototype_dim))
        self.assertEqual(tuple(outputs.pre_projection_tokens.shape), (self.batch_size, self.seq_len, self.prototype_dim))
        self.assertEqual(tuple(outputs.pre_projection_pooled.shape), (self.batch_size, self.prototype_dim))
        self.assertEqual(tuple(outputs.token_mask.shape), (self.batch_size, self.seq_len))
        self.assertEqual(outputs.pooling_mode, 'image_conditioned')
        self.assertEqual(outputs.metadata['encoder'], 'text')
        self.assertIn('eos', outputs.special_token_positions)
        self.assertIn('cls', outputs.special_token_positions)
        torch.testing.assert_close(outputs.special_token_positions['eos'], self.expected_eos)
        batch_indices = torch.arange(self.batch_size)
        torch.testing.assert_close(outputs.projected_pooled, outputs.projected_tokens[batch_indices, self.expected_eos])
        torch.testing.assert_close(outputs.pre_projection_pooled, outputs.pre_projection_tokens[batch_indices, self.expected_eos])

    def test_retrieval_encoder_contracts(self):
        model = self._build_model()
        image_features = model.encode_image_for_retrieval(self.images)
        self.assertEqual(set(image_features.keys()), {'image_projected', 'summary', 'routing_weights'})
        self.assertEqual(tuple(image_features['image_projected'].shape), (self.batch_size, self.projection_dim))
        self.assertEqual(tuple(image_features['summary'].shape), (self.batch_size, self.prototype_dim))
        self.assertEqual(tuple(image_features['routing_weights'].shape), (self.batch_size, self.num_prototypes))
        torch.testing.assert_close(
            image_features['routing_weights'].sum(dim=-1),
            torch.ones(self.batch_size, dtype=image_features['routing_weights'].dtype),
            atol=1e-5,
            rtol=1e-5,
        )

        text_features = model.encode_text_for_retrieval(self.caption_ids)
        self.assertEqual(tuple(text_features['text_token_states'].shape), (self.batch_size, self.seq_len, self.prototype_dim))
        self.assertEqual(tuple(text_features['token_ids'].shape), (self.batch_size, self.seq_len))
        self.assertEqual(tuple(text_features['attention_mask'].shape), (self.batch_size, self.seq_len))
        torch.testing.assert_close(text_features['token_ids'], self.caption_ids)
        torch.testing.assert_close(text_features['special_token_positions']['eos'], self.expected_eos)

        basis_features = model.encode_text_basis_for_retrieval(self.caption_ids)
        self.assertEqual(set(basis_features.keys()), {'basis_bank'})
        self.assertEqual(tuple(basis_features['basis_bank'].shape), (self.batch_size, self.num_prototypes, self.prototype_dim))

    def test_exact_similarity_entrypoint_uses_deployed_pairwise_pipeline(self):
        model = self._build_model().eval()
        image_features = model.encode_image_for_retrieval(self.images[:2])
        text_features = model.encode_text_for_retrieval(self.caption_ids)
        with mock.patch.object(model.prototype_head, 'compute_pairwise_similarity', wraps=model.prototype_head.compute_pairwise_similarity) as exact_mock:
            with mock.patch.object(model.prototype_head, 'compute_approximate_pairwise_similarity', wraps=model.prototype_head.compute_approximate_pairwise_similarity) as approx_mock:
                similarity = model.compute_retrieval_similarity(image_features, text_features)
        self.assertEqual(tuple(similarity.shape), (self.batch_size, 2))
        self.assertTrue(torch.isfinite(similarity).all())
        self.assertGreater(exact_mock.call_count, 0)
        self.assertEqual(approx_mock.call_count, 0)

    def test_optional_approximate_similarity_entrypoint_is_separate(self):
        model = self._build_model().eval()
        image_features = model.encode_image_for_retrieval(self.images[:2])
        basis_features = model.encode_text_basis_for_retrieval(self.caption_ids)
        with mock.patch.object(model.prototype_head, 'compute_pairwise_similarity', wraps=model.prototype_head.compute_pairwise_similarity) as exact_mock:
            with mock.patch.object(model.prototype_head, 'compute_approximate_pairwise_similarity', wraps=model.prototype_head.compute_approximate_pairwise_similarity) as approx_mock:
                similarity = model.compute_approximate_retrieval_similarity(image_features, basis_features)
        self.assertEqual(tuple(similarity.shape), (self.batch_size, 2))
        self.assertTrue(torch.isfinite(similarity).all())
        self.assertEqual(exact_mock.call_count, 0)
        self.assertGreater(approx_mock.call_count, 0)

    def test_forward_contract_covers_amortized_pipeline_dimensions(self):
        model = self._build_model()
        outputs = model(
            {'images': self.images, 'caption_ids': self.caption_ids, 'pids': self.pids},
            return_debug=False,
        )
        required_keys = {
            'loss_total',
            'loss_proxy',
            'loss_proxy_text_exact',
            'loss_align',
            'loss_diag',
            'loss_support',
            'loss_diversity',
            'loss_balance',
            'proxy_temperature',
            'use_loss_proxy_text_exact',
            'use_loss_support',
            'lambda_support',
            'retrieval_temperature',
            'logit_scale',
            'alpha',
            'z_v',
            'z_t_hat_diag',
            'z_t_exact_diag',
            'debug',
        }
        self.assertTrue(required_keys.issubset(outputs.keys()))
        self.assertEqual(tuple(outputs['alpha'].shape), (self.batch_size, self.num_prototypes))
        self.assertEqual(tuple(outputs['z_v'].shape), (self.batch_size, self.projection_dim))
        self.assertEqual(tuple(outputs['z_t_hat_diag'].shape), (self.batch_size, self.projection_dim))
        self.assertEqual(tuple(outputs['z_t_exact_diag'].shape), (self.batch_size, self.projection_dim))
        self.assertTrue(torch.isfinite(outputs['loss_total']))
        for key in ('prototype_usage_entropy', 'routing_entropy', 'token_pool_entropy', 'q_norm'):
            self.assertIn(key, outputs['debug'])

    def test_forward_debug_contract_exposes_canonical_internal_tensors(self):
        model = self._build_model()
        outputs = model(
            {'images': self.images, 'caption_ids': self.caption_ids, 'pids': self.pids},
            return_debug=True,
        )
        debug = outputs['debug']
        for key in ('alpha', 'beta', 'Q', 'Theta_v', 'Theta_tilde', 'basis_bank', 'T_hat_pool', 'T_exact_pool', 'Z_v', 'Z_t', 'Z_t_exact', 'text_exact_proxy_logits'):
            self.assertIn(key, debug)
        self.assertEqual(tuple(debug['alpha'].shape), (self.batch_size, self.num_prototypes))
        self.assertEqual(tuple(debug['beta'].shape), (self.batch_size, self.seq_len))
        self.assertEqual(tuple(debug['Q'].shape), (self.batch_size, self.prototype_dim))
        self.assertEqual(tuple(debug['Theta_v'].shape), (self.num_prototypes, self.prototype_dim))
        self.assertEqual(tuple(debug['Theta_tilde'].shape), (self.num_prototypes, self.prototype_dim))
        self.assertEqual(tuple(debug['basis_bank'].shape), (self.batch_size, self.num_prototypes, self.prototype_dim))
        self.assertEqual(tuple(debug['T_hat_pool'].shape), (self.batch_size, self.prototype_dim))
        self.assertEqual(tuple(debug['T_exact_pool'].shape), (self.batch_size, self.prototype_dim))
        self.assertEqual(tuple(debug['Z_v'].shape), (self.batch_size, self.projection_dim))
        self.assertEqual(tuple(debug['Z_t'].shape), (self.batch_size, self.projection_dim))
        self.assertEqual(tuple(debug['Z_t_exact'].shape), (self.batch_size, self.projection_dim))
        self.assertNotIn('pairwise_similarity', debug)
        self.assertNotIn('pairwise_text_bank', debug)
        if 'basis_token_scores' in debug:
            self.assertEqual(tuple(debug['basis_token_scores'].shape), (self.batch_size, self.num_prototypes, self.seq_len))
        if 'basis_token_weights' in debug:
            self.assertEqual(tuple(debug['basis_token_weights'].shape), (self.batch_size, self.num_prototypes, self.seq_len))

    def test_no_prototype_bank_uses_direct_image_conditioned_pooling(self):
        model = self._build_model(use_prototype_bank=False, retrieval_scorer='exact').eval()
        self.assertFalse(hasattr(model.prototype_head, 'prototype_bank'))

        image_features = model.encode_image_for_retrieval(self.images)
        self.assertEqual(set(image_features.keys()), {'image_projected', 'summary', 'routing_weights'})
        self.assertEqual(tuple(image_features['routing_weights'].shape), (self.batch_size, 0))
        self.assertEqual(tuple(image_features['summary'].shape), (self.batch_size, self.prototype_dim))

        outputs = model(
            {'images': self.images, 'caption_ids': self.caption_ids, 'pids': self.pids},
            return_debug=True,
        )
        self.assertEqual(tuple(outputs['alpha'].shape), (self.batch_size, 0))
        self.assertEqual(tuple(outputs['debug']['Theta_v'].shape), (0, self.prototype_dim))
        self.assertEqual(tuple(outputs['debug']['Theta_tilde'].shape), (0, self.prototype_dim))
        self.assertEqual(tuple(outputs['debug']['basis_bank'].shape), (self.batch_size, 0, self.prototype_dim))
        self.assertIn('direct_image_conditioned_pooling', outputs['debug'])
        self.assertTrue(torch.isfinite(outputs['loss_total']))

        text_features = model.encode_text_for_retrieval(self.caption_ids)
        similarity = model.compute_retrieval_similarity(image_features, text_features)
        self.assertEqual(tuple(similarity.shape), (self.batch_size, self.batch_size))
        self.assertTrue(torch.isfinite(similarity).all())

    def test_no_prototype_bank_rejects_approximate_retrieval(self):
        model = self._build_model(use_prototype_bank=False, retrieval_scorer='exact').eval()
        with self.assertRaisesRegex(RuntimeError, 'use_prototype_bank=false'):
            model.encode_text_basis_for_retrieval(self.caption_ids)
        with self.assertRaisesRegex(ValueError, 'retrieval_scorer=approximate'):
            self._build_model(use_prototype_bank=False, retrieval_scorer='approximate')
    def test_named_optimizer_groups_contract_exposes_required_group_names(self):
        model = self._build_model(freeze_image_backbone=False, freeze_text_backbone=False)
        groups = model.named_optimizer_groups()
        for key in ('prototype_bank', 'projectors', 'class_proxies', 'image_backbone', 'text_backbone', 'other'):
            self.assertIn(key, groups)
        self.assertGreater(len(groups['prototype_bank']), 0)
        self.assertGreater(len(groups['projectors']), 0)
        self.assertGreater(len(groups['class_proxies']), 0)
        self.assertGreater(len(groups['image_backbone']), 0)
        self.assertGreater(len(groups['text_backbone']), 0)
        self.assertTrue(any(name == 'prototype_head.losses.class_proxies' for name, _ in groups['class_proxies']))

    def test_evaluator_defaults_to_exact_retrieval_pipeline(self):
        model = self._build_model(retrieval_scorer='exact').eval()
        evaluator = Evaluator(self.image_loader, self.text_loader, self._build_args(retrieval_scorer='exact'))
        with mock.patch.object(model, 'compute_retrieval_similarity', wraps=model.compute_retrieval_similarity) as exact_mock:
            with mock.patch.object(model, 'compute_approximate_retrieval_similarity', wraps=model.compute_approximate_retrieval_similarity) as approx_mock:
                top1 = evaluator.eval(model)
        self.assertTrue(torch.isfinite(torch.tensor(top1)))
        self.assertGreater(exact_mock.call_count, 0)
        self.assertEqual(approx_mock.call_count, 0)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

