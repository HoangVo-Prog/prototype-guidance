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
            backbone_precision='fp16',
            prototype_precision='fp32',
            temperature=0.07,
            learn_logit_scale=True,
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
            optimizer='AdamW',
            lr=0.01,
            lr_prototype_bank=0.02,
            lr_projectors=0.04,
            lr_logit_scale=0.005,
            lr_image_backbone=0.001,
            lr_text_backbone=0.001,
            weight_decay=0.01,
            weight_decay_prototype_bank=0.02,
            weight_decay_projectors=0.04,
            weight_decay_logit_scale=0.0,
            weight_decay_image_backbone=0.05,
            weight_decay_text_backbone=0.06,
            momentum=0.9,
            alpha=0.9,
            beta=0.999,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_forward_returns_structured_losses_and_lightweight_debug_metrics(self):
        model = PASModel(self._build_args(), num_classes=2)
        outputs = model(self.batch, return_debug=False)
        for key in ('loss_total', 'loss_infonce', 'loss_diversity', 'loss_balance', 'debug'):
            self.assertIn(key, outputs)
        for key in ('prototype_usage_entropy', 'routing_entropy', 'token_pool_entropy', 'q_norm'):
            self.assertIn(key, outputs['debug'])
        self.assertTrue(torch.isfinite(outputs['loss_total']))

    def test_forward_with_full_debug_exposes_canonical_tensors(self):
        model = PASModel(self._build_args(), num_classes=2)
        outputs = model(self.batch, return_debug=True)
        for key in ('alpha', 'beta', 'Q', 'Theta_v', 'Theta_tilde', 'token_valid_mask', 'token_keep_mask', 'beta_logits_masked', 'Z_v_raw', 'Z_t_raw'):
            self.assertIn(key, outputs['debug'])
        self.assertTrue(torch.isfinite(outputs['loss_total']))

    def test_freeze_policy_freezes_backbones_and_keeps_prototype_head_trainable(self):
        model = PASModel(self._build_args(freeze_backbones=True), num_classes=2)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.base_model.visual.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.base_model.transformer.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.base_model.token_embedding.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.prototype_head.parameters()))

    def test_optimizer_groups_follow_named_group_contract(self):
        args = self._build_args(freeze_backbones=False, backbone_precision='fp32')
        model = PASModel(args, num_classes=2)
        optimizer = build_optimizer(args, model)
        groups = {group['name']: group for group in optimizer.param_groups}
        self.assertEqual(groups['prototype_bank']['lr'], args.lr_prototype_bank)
        self.assertNotIn('contextualizer', groups)
        self.assertEqual(groups['projectors']['lr'], args.lr_projectors)
        self.assertEqual(groups['logit_scale']['lr'], args.lr_logit_scale)
        self.assertEqual(groups['image_backbone']['lr'], args.lr_image_backbone)
        self.assertEqual(groups['text_backbone']['lr'], args.lr_text_backbone)
        self.assertEqual(groups['prototype_bank']['weight_decay'], args.weight_decay_prototype_bank)
        self.assertEqual(groups['projectors']['weight_decay'], args.weight_decay_projectors)
        self.assertEqual(groups['logit_scale']['weight_decay'], args.weight_decay_logit_scale)

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
            build_model(self._build_args(backbone_precision='fp32', prototype_precision='fp16', amp=True, amp_dtype='bf16'), num_classes=2)

    def test_evaluator_runs_end_to_end(self):
        args = self._build_args()
        model = PASModel(args, num_classes=2)
        evaluator = Evaluator(self.img_loader, self.text_loader, args)
        top1 = evaluator.eval(model.eval())
        self.assertTrue(torch.isfinite(torch.tensor(top1)))
        self.assertIn('val/pas/R1', evaluator.latest_metrics)

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

    def test_model_supports_non_matching_prototype_dim(self):
        model = PASModel(self._build_args(prototype_dim=6, projector_hidden_dim=10), num_classes=2)
        outputs = model(self.batch, return_debug=True)
        self.assertEqual(tuple(outputs['debug']['Q'].shape), (4, 6))
        self.assertEqual(tuple(outputs['debug']['Theta_v'].shape), (4, 6))
        self.assertEqual(tuple(outputs['debug']['text_tokens'].shape), (4, 6, 8))
        self.assertTrue(torch.isfinite(outputs['loss_total']))

    def test_retrieval_similarity_uses_the_same_logit_scale_family_as_training(self):
        model = PASModel(self._build_args(), num_classes=2).eval()
        images = self.batch['images'][:2]
        text = self.batch['caption_ids'][:2]
        image_features = model.encode_image_for_retrieval(images)
        text_features = model.encode_text_for_retrieval(text)

        with torch.no_grad():
            model.prototype_head.losses.logit_scale.copy_(torch.log(torch.tensor(2.0)))
            similarity_a = model.compute_retrieval_similarity(image_features, text_features)
            model.prototype_head.losses.logit_scale.copy_(torch.log(torch.tensor(4.0)))
            similarity_b = model.compute_retrieval_similarity(image_features, text_features)

        torch.testing.assert_close(similarity_b, similarity_a * 2.0, atol=1e-5, rtol=1e-5)

    def test_embedding_dim_mismatch_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, r'model\.embedding_dim must match the CLIP backbone feature dimension'):
            PASModel(self._build_args(embedding_dim=7), num_classes=2)

    def test_retrieval_text_interface_reuses_training_text_state_family(self):
        model = PASModel(self._build_args(), num_classes=2)
        text = self.batch['caption_ids'][:2]
        training_text = model.extract_text_features(text)
        retrieval_text = model.encode_text_for_retrieval(text)
        torch.testing.assert_close(retrieval_text['text_token_states'], model._resolve_text_states(training_text))
        torch.testing.assert_close(retrieval_text['attention_mask'], training_text.token_mask)
        self.assertEqual(set(retrieval_text['special_token_positions'].keys()), set(training_text.special_token_positions.keys()))

    def test_forward_requires_pids_for_identity_aware_training(self):
        model = PASModel(self._build_args(), num_classes=2)
        batch = dict(self.batch)
        batch.pop('pids')
        with self.assertRaisesRegex(KeyError, r"batch\['pids'\]"):
            model(batch)

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
