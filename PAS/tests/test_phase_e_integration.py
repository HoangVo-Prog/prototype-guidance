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
        from model.build import PASModel, build_model
        from solver.build import build_optimizer
        from utils.metrics import Evaluator
    except Exception as exc:  # pragma: no cover - environment-dependent
        PASModel = None
        build_model = None
        build_optimizer = None
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
            flat = image.view(image.size(0), -1).float()
            cls_token = self.visual(flat)
            aux_token = torch.tanh(cls_token)
            tokens = torch.stack([cls_token, aux_token], dim=1)
            attention = [torch.eye(tokens.size(1), device=image.device).unsqueeze(0).repeat(image.size(0), 1, 1)] if return_all else None
            return {
                'projected_tokens': tokens,
                'pre_projection_tokens': tokens,
                'attention_weights': attention,
            }

        def encode_text_intermediates(self, text, return_all=False, average_attn_weights=True):
            embedded = self.token_embedding(text.long())
            positional = self.positional_embedding[:text.size(1)].unsqueeze(0)
            hidden = self.transformer(embedded + positional)
            hidden = self.ln_final(hidden)
            projected = hidden @ self.text_projection
            attention = [torch.eye(text.size(1), device=text.device).unsqueeze(0).repeat(text.size(0), 1, 1)] if return_all else None
            return {
                'projected_tokens': projected,
                'pre_projection_tokens': hidden,
                'attention_weights': attention,
            }
else:
    DummyCLIPBackbone = None


@unittest.skipUnless(
    torch is not None and PASModel is not None and Evaluator is not None and build_optimizer is not None and build_model is not None,
    f'Integration tests require torch and repo runtime imports: {IMPORT_ERROR}',
)
class PhaseEIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.patch_build = mock.patch(
            'model.build.build_CLIP_from_openai_pretrained',
            side_effect=lambda *args, **kwargs: (DummyCLIPBackbone(), {'embed_dim': 8, 'vision_layers': 1, 'transformer_width': 8}),
        )
        self.patch_build.start()
        self.addCleanup(self.patch_build.stop)

        self.batch = {
            'images': torch.randn(4, 3, 4, 4),
            'caption_ids': torch.tensor(
                [
                    [49406, 11, 12, 49407, 0, 0],
                    [49406, 21, 22, 49407, 0, 0],
                    [49406, 31, 32, 49407, 0, 0],
                    [49406, 41, 42, 49407, 0, 0],
                ],
                dtype=torch.long,
            ),
            'pids': torch.tensor([0, 1, 0, 1], dtype=torch.long),
            'image_pids': torch.tensor([0, 1, 0, 1], dtype=torch.long),
            'caption_pids': torch.tensor([0, 1, 0, 1], dtype=torch.long),
        }
        self.text_loader = [
            (
                torch.tensor([0, 1], dtype=torch.long),
                torch.tensor(
                    [
                        [49406, 11, 12, 49407, 0, 0],
                        [49406, 21, 22, 49407, 0, 0],
                    ],
                    dtype=torch.long,
                ),
            ),
            (
                torch.tensor([0, 1], dtype=torch.long),
                torch.tensor(
                    [
                        [49406, 31, 32, 49407, 0, 0],
                        [49406, 41, 42, 49407, 0, 0],
                    ],
                    dtype=torch.long,
                ),
            ),
        ]
        self.img_loader = [
            (torch.tensor([0, 1], dtype=torch.long), torch.randn(2, 3, 4, 4)),
            (torch.tensor([0, 1], dtype=torch.long), torch.randn(2, 3, 4, 4)),
        ]

    def _build_args(self, freeze_backbones=True, **overrides):
        base = dict(
            pretrain_choice='ViT-B/16',
            img_size=(4, 4),
            stride_size=1,
            model_name='PAS',
            model_variant='pas_test',
            image_backbone='dummy_visual',
            text_backbone='dummy_text',
            embedding_dim=8,
            projection_dim=4,
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
            use_loss_ret_exact=False,
            lambda_ret_exact=1.0,
            ret_exact_temperature=None,
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
            prototype_num_prototypes=4,
            prototype_dim=8,
            prototype_init='normalized_random',
            prototype_init_path=None,
            prototype_routing_type='cosine',
            prototype_temperature=0.07,
            prototype_contextualization_type='self_attention',
            prototype_contextualization_residual=True,
            normalize_for_self_interaction=True,
            normalize_for_routing=True,
            use_balancing_loss=False,
            prototype_balance_loss_weight=0.0,
            prototype_dead_threshold=0.005,
            use_diversity_loss=True,
            diversity_loss_weight=0.01,
            token_policy='content_only',
            token_scoring_type='cosine',
            normalize_for_token_scoring=True,
            token_pooling_temperature=0.07,
            special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
            error_on_empty_kept_tokens=True,
            freeze_image_backbone=freeze_backbones,
            freeze_text_backbone=freeze_backbones,
            prototype_eval_image_chunk_size=2,
            prototype_eval_text_chunk_size=2,
            retrieval_scorer='exact',
            optimizer='AdamW',
            lr=0.01,
            lr_prototype_bank=0.02,
            lr_projectors=0.04,
            lr_class_proxies=0.03,
            lr_image_backbone=0.001,
            lr_text_backbone=0.001,
            weight_decay=0.01,
            weight_decay_prototype_bank=0.02,
            weight_decay_projectors=0.04,
            weight_decay_class_proxies=0.07,
            weight_decay_image_backbone=0.05,
            weight_decay_text_backbone=0.06,
            momentum=0.9,
            alpha=0.9,
            beta=0.999,
            training=True,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_forward_returns_structured_amortized_losses_and_lightweight_debug_metrics(self):
        model = PASModel(self._build_args(), num_classes=2)
        outputs = model(self.batch, return_debug=False)
        for key in ('loss_total', 'loss_proxy', 'loss_proxy_text_exact', 'loss_ret_exact', 'loss_align', 'loss_diag', 'loss_support', 'loss_diversity', 'loss_balance', 'debug'):
            self.assertIn(key, outputs)
        for key in (
            'prototype_usage_entropy',
            'routing_entropy',
            'token_pool_entropy',
            'q_norm',
            'image_surrogate_positive_cosine_mean',
            'image_surrogate_hardest_negative_cosine_mean',
            'image_exact_positive_cosine_mean',
            'image_exact_hardest_negative_cosine_mean',
        ):
            self.assertIn(key, outputs['debug'])
        self.assertTrue(torch.isfinite(outputs['loss_total']))


    def test_forward_with_exact_retrieval_loss_exposes_pairwise_logits(self):
        model = PASModel(self._build_args(
            use_loss_proxy_image=False,
            use_loss_proxy_text=False,
            use_loss_proxy_text_exact=False,
            use_loss_align=False,
            use_loss_diag=False,
            use_loss_ret_exact=True,
            use_diversity_loss=False,
            prototype_balance_loss_weight=0.0,
            use_balancing_loss=False,
        ), num_classes=2)
        outputs = model(self.batch, return_debug=False)
        self.assertIn('loss_ret_exact', outputs)
        self.assertIn('exact_pairwise_logits', outputs)
        self.assertEqual(tuple(outputs['exact_pairwise_logits'].shape), (4, 4))
        self.assertGreaterEqual(outputs['loss_ret_exact'].item(), 0.0)


    def test_forward_can_disable_proxy_losses_for_heldout_validation_batches(self):
        model = PASModel(self._build_args(use_loss_ret_exact=True), num_classes=2)
        heldout_batch = dict(self.batch)
        heldout_batch['pids'] = torch.tensor([10, 11, 10, 11], dtype=torch.long)
        heldout_batch['image_pids'] = heldout_batch['pids']
        heldout_batch['caption_pids'] = heldout_batch['pids']
        outputs = model(heldout_batch, return_debug=False, disable_proxy_losses=True)
        self.assertEqual(outputs['loss_proxy'].item(), 0.0)
        self.assertTrue(torch.isfinite(outputs['loss_total']))
        self.assertGreaterEqual(outputs['loss_ret_exact'].item(), 0.0)

    def test_forward_with_full_debug_exposes_surrogate_and_exact_tensors(self):
        model = PASModel(self._build_args(), num_classes=2)
        outputs = model(self.batch, return_debug=True)
        for key in ('alpha', 'Q', 'Theta_v', 'Theta_tilde', 'basis_bank', 'Z_t', 'Z_t_exact', 'text_exact_proxy_logits'):
            self.assertIn(key, outputs['debug'])
        self.assertTrue(torch.isfinite(outputs['loss_total']))

    def test_stage1_support_recipe_runs_with_diag_support_and_diversity_only(self):
        model = PASModel(
            self._build_args(
                use_loss_proxy_image=False,
                use_loss_proxy_text=False,
                use_loss_proxy_text_exact=False,
                lambda_proxy=0.0,
                use_loss_align=False,
                lambda_align=0.0,
                use_loss_support=True,
                lambda_support=0.1,
                support_min=2.0,
                use_balancing_loss=False,
                prototype_balance_loss_weight=0.0,
                use_diversity_loss=True,
                diversity_loss_weight=0.01,
            ),
            num_classes=2,
        )
        outputs = model(self.batch, return_debug=False)
        self.assertIn('loss_support', outputs)
        self.assertEqual(outputs['use_loss_support'].item(), 1.0)
        self.assertAlmostEqual(outputs['lambda_support'].item(), 0.1, places=6)
        self.assertEqual(outputs['loss_balance'].item(), 0.0)
        self.assertTrue(torch.isfinite(outputs['loss_total']))

    def test_forward_requires_pids_for_proxy_training(self):
        model = PASModel(self._build_args(), num_classes=2)
        batch = dict(self.batch)
        batch.pop('pids')
        with self.assertRaisesRegex(KeyError, r"batch\['pids'\].*proxy objective"):
            model(batch)

    def test_forward_requires_num_classes_for_training(self):
        with self.assertRaisesRegex(ValueError, r'num_classes > 0'):
            PASModel(self._build_args(), num_classes=0)

    def test_forward_rejects_mismatched_caption_pids(self):
        model = PASModel(self._build_args(), num_classes=2)
        batch = dict(self.batch)
        batch['caption_pids'] = torch.tensor([1, 1, 0, 1], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, r'Batch label mismatch'):
            model(batch)

    def test_optimizer_groups_follow_named_group_contract(self):
        args = self._build_args(freeze_backbones=False)
        model = PASModel(args, num_classes=2)
        optimizer = build_optimizer(args, model)
        groups = {group['name']: group for group in optimizer.param_groups}
        self.assertEqual(groups['prototype_bank']['lr'], args.lr_prototype_bank)
        self.assertEqual(groups['projectors']['lr'], args.lr_projectors)
        self.assertEqual(groups['class_proxies']['lr'], args.lr_class_proxies)
        self.assertEqual(groups['class_proxies']['weight_decay'], args.weight_decay_class_proxies)
        self.assertEqual(groups['image_backbone']['lr'], args.lr_image_backbone)
        self.assertEqual(groups['text_backbone']['lr'], args.lr_text_backbone)
        self.assertNotIn('logit_scale', groups)


    def test_freeze_proxy_removes_class_proxies_from_trainable_optimizer_groups(self):
        args = self._build_args(freeze_proxy=True)
        model = PASModel(args, num_classes=2)
        self.assertFalse(model.prototype_head.losses.class_proxies.requires_grad)
        optimizer = build_optimizer(args, model)
        groups = {group['name']: group for group in optimizer.param_groups}
        self.assertNotIn('class_proxies', groups)


    def test_freeze_prototype_removes_prototype_bank_from_trainable_optimizer_groups(self):
        args = self._build_args(freeze_prototype=True)
        model = PASModel(args, num_classes=2)
        prototype_parameters = list(model.prototype_head.prototype_bank.parameters())
        self.assertTrue(prototype_parameters)
        self.assertTrue(all(not parameter.requires_grad for parameter in prototype_parameters))
        optimizer = build_optimizer(args, model)
        groups = {group['name']: group for group in optimizer.param_groups}
        self.assertNotIn('prototype_bank', groups)

    def test_build_model_respects_prototype_precision_setting(self):
        model_fp32 = build_model(self._build_args(prototype_precision='fp32'), num_classes=2)
        prototype_dtypes_fp32 = {parameter.dtype for parameter in model_fp32.prototype_head.parameters()}
        self.assertEqual(prototype_dtypes_fp32, {torch.float32})

        model_fp16 = build_model(self._build_args(prototype_precision='fp16', amp=True, amp_dtype='fp16'), num_classes=2)
        prototype_dtypes_fp16 = {parameter.dtype for parameter in model_fp16.prototype_head.parameters()}
        self.assertEqual(prototype_dtypes_fp16, {torch.float16})

    def test_build_model_respects_backbone_precision_setting(self):
        model_fp16 = build_model(self._build_args(backbone_precision='fp16'), num_classes=2)
        self.assertEqual(model_fp16.base_model.visual.weight.dtype, torch.float16)
        model_fp32 = build_model(self._build_args(backbone_precision='fp32'), num_classes=2)
        self.assertEqual(model_fp32.base_model.visual.weight.dtype, torch.float32)

    def test_unfrozen_fp16_backbone_requires_amp(self):
        with self.assertRaisesRegex(ValueError, r'Unfrozen fp16 backbone training requires training\.amp=true'):
            build_model(self._build_args(freeze_backbones=False, backbone_precision='fp16', amp=False), num_classes=2)

    def test_fp16_backbone_rejects_bf16_amp(self):
        with self.assertRaisesRegex(ValueError, r'model\.backbone_precision=fp16 requires training\.amp_dtype=fp16'):
            build_model(self._build_args(backbone_precision='fp16', amp=True, amp_dtype='bf16'), num_classes=2)

    def test_fp16_prototype_requires_amp(self):
        with self.assertRaisesRegex(ValueError, r'model\.prototype_precision=fp16 requires training\.amp=true'):
            build_model(self._build_args(prototype_precision='fp16', amp=False), num_classes=2)

    def test_fp16_prototype_rejects_bf16_amp(self):
        with self.assertRaisesRegex(ValueError, r'model\.prototype_precision=fp16 requires training\.amp_dtype=fp16'):
            build_model(self._build_args(prototype_precision='fp16', amp=True, amp_dtype='bf16'), num_classes=2)

    def test_build_model_requires_normalized_projector_outputs(self):
        with self.assertRaisesRegex(ValueError, r'normalize_projector_outputs=true'):
            build_model(self._build_args(normalize_projector_outputs=False), num_classes=2)

    def test_build_model_requires_num_classes_for_eval_construction_too(self):
        with self.assertRaisesRegex(ValueError, r'num_classes > 0'):
            build_model(self._build_args(training=False), num_classes=0)


    def test_random_init_without_path_skips_automatic_feature_fallback(self):
        with mock.patch.object(PASModel, '_extract_train_image_embeddings', wraps=PASModel._extract_train_image_embeddings) as extract_mock:
            PASModel(self._build_args(prototype_init='normalized_random'), num_classes=2)
        self.assertEqual(extract_mock.call_count, 0)

    def test_data_driven_init_with_path_preserves_existing_behavior_and_skips_fallback(self):
        prototypes = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        with mock.patch.object(PASModel, '_extract_train_image_embeddings', wraps=PASModel._extract_train_image_embeddings) as extract_mock:
            with mock.patch('model.prototype.prototype_bank.PrototypeBank._load_external_prototypes', return_value=prototypes):
                build_model(
                    self._build_args(prototype_init='sampled_image_embeddings', prototype_init_path='features.pt'),
                    num_classes=2,
                )
        self.assertEqual(extract_mock.call_count, 0)

    def test_missing_path_data_driven_init_uses_train_image_fallback_once(self):
        fallback_features = torch.randn(12, 8)
        train_loader = SimpleNamespace(dataset=SimpleNamespace(dataset=[(0, 0, 'unused.jpg', 'caption')]))
        with mock.patch.object(PASModel, '_extract_train_image_embeddings', return_value=fallback_features) as extract_mock:
            model = build_model(
                self._build_args(prototype_init='sampled_image_embeddings', prototype_init_path=None),
                num_classes=2,
                train_loader=train_loader,
            )
        self.assertEqual(extract_mock.call_count, 1)
        diagnostics = model.prototype_head.prototype_bank.last_init_diagnostics
        self.assertTrue(diagnostics['auto_train_image_fallback_used'])
        self.assertEqual(diagnostics['feature_count'], fallback_features.size(0))
        self.assertEqual(diagnostics['clustering_strategy'], 'sampled_image_embeddings')

    def test_missing_path_data_driven_init_requires_train_loader(self):
        with self.assertRaisesRegex(ValueError, r'requires train image embeddings'):
            build_model(
                self._build_args(prototype_init='kmeans_centroids', prototype_init_path=None),
                num_classes=2,
            )

    def test_tiny_overfit_reduces_loss(self):
        model = PASModel(self._build_args(), num_classes=2)
        optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.05)
        losses = []
        model.train()
        for _ in range(8):
            optimizer.zero_grad()
            outputs = model(self.batch)
            loss = outputs['loss_total']
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
        self.assertLess(min(losses[1:]), losses[0])

    def test_encode_text_basis_for_retrieval_returns_basis_bank(self):
        model = PASModel(self._build_args(), num_classes=2)
        basis_features = model.encode_text_basis_for_retrieval(self.batch['caption_ids'][:2])
        self.assertEqual(tuple(basis_features['basis_bank'].shape), (2, 4, 8))

    def test_retrieval_text_interface_reuses_training_text_state_family(self):
        model = PASModel(self._build_args(), num_classes=2)
        text = self.batch['caption_ids'][:2]
        training_text = model.extract_text_features(text)
        retrieval_text = model.encode_text_for_retrieval(text)
        torch.testing.assert_close(retrieval_text['text_token_states'], model._resolve_text_states(training_text))
        torch.testing.assert_close(retrieval_text['attention_mask'], training_text.token_mask)

    def test_evaluator_runs_exact_end_to_end(self):
        args = self._build_args(retrieval_scorer='exact')
        model = PASModel(args, num_classes=2)
        evaluator = Evaluator(self.img_loader, self.text_loader, args)
        top1 = evaluator.eval(model.eval())
        self.assertTrue(torch.isfinite(torch.tensor(top1)))
        self.assertIn('val/pas/R1', evaluator.latest_metrics)
        self.assertIn('val/debug/eval_positive_exact_cosine_mean', evaluator.latest_metrics)
        self.assertIn('val/debug/eval_hardest_negative_exact_cosine_mean', evaluator.latest_metrics)
        self.assertIn('val/debug/eval_exact_margin_mean', evaluator.latest_metrics)

    def test_evaluator_runs_approximate_end_to_end(self):
        args = self._build_args(retrieval_scorer='approximate')
        model = PASModel(args, num_classes=2)
        evaluator = Evaluator(self.img_loader, self.text_loader, args)
        top1 = evaluator.eval(model.eval())
        self.assertTrue(torch.isfinite(torch.tensor(top1)))
        self.assertIn('val/pas/R1', evaluator.latest_metrics)

    def test_evaluator_default_uses_exact_scorer(self):
        args = self._build_args(retrieval_scorer='exact')
        model = PASModel(args, num_classes=2)
        evaluator = Evaluator(self.img_loader, self.text_loader, args)
        with mock.patch.object(model, 'compute_retrieval_similarity', wraps=model.compute_retrieval_similarity) as exact_mock:
            with mock.patch.object(model, 'compute_approximate_retrieval_similarity', wraps=model.compute_approximate_retrieval_similarity) as approx_mock:
                evaluator.eval(model.eval())
        self.assertGreater(exact_mock.call_count, 0)
        self.assertEqual(approx_mock.call_count, 0)

    def test_optional_approximate_scorer_is_not_default(self):
        args = self._build_args(retrieval_scorer='approximate')
        model = PASModel(args, num_classes=2)
        evaluator = Evaluator(self.img_loader, self.text_loader, args)
        with mock.patch.object(model, 'compute_retrieval_similarity', wraps=model.compute_retrieval_similarity) as exact_mock:
            with mock.patch.object(model, 'compute_approximate_retrieval_similarity', wraps=model.compute_approximate_retrieval_similarity) as approx_mock:
                evaluator.eval(model.eval())
        self.assertEqual(exact_mock.call_count, 0)
        self.assertGreater(approx_mock.call_count, 0)

    def test_exact_and_approximate_similarity_both_available(self):
        model = PASModel(self._build_args(), num_classes=2).eval()
        images = self.batch['images'][:2]
        text = self.batch['caption_ids'][:2]
        image_features = model.encode_image_for_retrieval(images)
        text_features = model.encode_text_for_retrieval(text)
        text_basis_features = model.encode_text_basis_for_retrieval(text)
        exact_similarity = model.compute_retrieval_similarity(image_features, text_features)
        approximate_similarity = model.compute_approximate_retrieval_similarity(image_features, text_basis_features)
        self.assertEqual(tuple(exact_similarity.shape), (2, 2))
        self.assertEqual(tuple(approximate_similarity.shape), (2, 2))
        self.assertTrue(torch.isfinite(exact_similarity).all())
        self.assertTrue(torch.isfinite(approximate_similarity).all())

    def test_inference_similarity_depends_on_prototype_bank(self):
        model = PASModel(self._build_args(), num_classes=2).eval()
        images = self.batch['images'][:2]
        text = self.batch['caption_ids'][:2]
        image_features_a = model.encode_image_for_retrieval(images)
        text_features = model.encode_text_for_retrieval(text)
        similarity_a = model.compute_retrieval_similarity(image_features_a, text_features)
        with torch.no_grad():
            replacement_bank = torch.randn_like(model.prototype_head.prototype_bank.prototypes)
            replacement_bank = torch.nn.functional.normalize(replacement_bank, dim=-1)
            model.prototype_head.prototype_bank.prototypes.copy_(replacement_bank)
        image_features_b = model.encode_image_for_retrieval(images)
        similarity_b = model.compute_retrieval_similarity(image_features_b, text_features)
        self.assertFalse(torch.allclose(similarity_a, similarity_b))

    def test_build_model_accepts_supported_vit_choices(self):
        for pretrain_choice in ('ViT-B/16', 'ViT-B/32', 'ViT-L/14'):
            model = build_model(self._build_args(pretrain_choice=pretrain_choice), num_classes=2)
            self.assertEqual(model.embed_dim, 8)

    def test_build_model_rejects_unsupported_resnet_choices(self):
        with self.assertRaisesRegex(ValueError, r'Supported `pretrain_choice` values'):
            build_model(self._build_args(pretrain_choice='RN50'), num_classes=2)

    def test_build_model_rejects_incompatible_text_width_contract(self):
        with mock.patch(
            'model.build.build_CLIP_from_openai_pretrained',
            side_effect=lambda *args, **kwargs: (DummyCLIPBackbone(), {'embed_dim': 8, 'vision_layers': 1, 'transformer_width': 7}),
        ):
            with self.assertRaisesRegex(ValueError, r'transformer_width == embed_dim'):
                build_model(self._build_args(), num_classes=2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()




