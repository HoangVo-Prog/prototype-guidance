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
        self._dummy_clip_builder = lambda *args, **kwargs: (
            DummyCLIPBackbone(),
            {'embed_dim': 8, 'vision_layers': 1, 'transformer_width': 8},
        )

        self.patch_build = mock.patch(
            'model.build.build_CLIP_from_openai_pretrained',
            side_effect=self._dummy_clip_builder,
        )
        self.patch_build.start()
        self.addCleanup(self.patch_build.stop)

        # Host-only runtime (clip/itself) resolves CLIP builder from the original
        # adapter module object returned by get_original_itself_components(), so we
        # patch that object directly to keep tests independent from external weights.
        from model.hosts.itself import get_original_itself_components
        components = get_original_itself_components()
        self.patch_original_components_build = mock.patch.object(
            components.model_build,
            'build_CLIP_from_openai_pretrained',
            side_effect=self._dummy_clip_builder,
        )
        self.patch_original_components_build.start()
        self.addCleanup(self.patch_original_components_build.stop)

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

    def _build_args(self, stage='stage1', finetune='', **overrides):
        stage_freezes = {
            'stage0': (False, False, True),
            'stage1': (True, True, False),
            'stage2': (False, False, True),
            'joint': (False, False, False),
        }
        freeze_image_backbone, freeze_text_backbone, freeze_prototype_side = stage_freezes[stage]
        base = dict(
            pretrain_choice='ViT-B/16',
            img_size=(4, 4),
            stride_size=1,
            model_name='PAS',
            model_variant='pas_test',
            training_mode='pas',
            host_type='clip',
            image_backbone='dummy_visual',
            text_backbone='dummy_text',
            embedding_dim=8,
            projection_dim=4,
            projector_hidden_dim=8,
            projector_dropout=0.0,
            projector_type='mlp2',
            normalize_projector_outputs=True,
            use_custom_projector=True,
            backbone_precision='fp32',
            prototype_precision='fp32',
            temperature=0.07,
            proxy_temperature=0.2,
            use_host_loss=True,
            lambda_host=1.0,
            itself_loss_names='tal+cid',
            itself_only_global=False,
            itself_select_ratio=0.4,
            itself_grab_embed_dim=16,
            itself_score_weight_global=0.68,
            itself_tau=0.015,
            itself_margin=0.1,
            itself_return_all=True,
            itself_topk_type='mean',
            itself_layer_index=-1,
            itself_average_attn_weights=True,
            itself_modify_k=False,
            lambda_proxy=1.0,
            lambda_proxy_image=1.0,
            lambda_proxy_text=1.0,
            lambda_proxy_text_exact=1.0,
            retrieval_mode='surrogate_i2t',
            use_loss_proxy_image=True,
            use_loss_proxy_text=True,
            use_loss_proxy_text_exact=True,
            use_loss_ret=True,
            lambda_ret=1.0,
            use_loss_align=False,
            lambda_align=0.0,
            use_loss_diag=True,
            lambda_diag=1.0,
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
            training_stage=stage,
            freeze_image_backbone=freeze_image_backbone,
            freeze_text_backbone=freeze_text_backbone,
            freeze_prototype_side=freeze_prototype_side,
            freeze_host_projectors=(stage == 'stage1'),
            finetune=finetune,
            prototype_eval_image_chunk_size=2,
            prototype_eval_text_chunk_size=2,
            retrieval_scorer='exact',
            optimizer='AdamW',
            lr=0.01,
            lr_prototype_bank=0.02,
            lr_projectors=0.04,
            lr_prototype_routing=0.05,
            lr_prototype_pooling=0.06,
            lr_prototype_contextualization=0.07,
            lr_class_proxies=0.03,
            lr_image_backbone=0.001,
            lr_text_backbone=0.001,
            weight_decay=0.01,
            weight_decay_prototype_bank=0.02,
            weight_decay_projectors=0.04,
            weight_decay_prototype_routing=0.08,
            weight_decay_prototype_pooling=0.09,
            weight_decay_prototype_contextualization=0.11,
            weight_decay_class_proxies=0.07,
            weight_decay_image_backbone=0.05,
            weight_decay_text_backbone=0.06,
            momentum=0.9,
            alpha=0.9,
            beta=0.999,
            lr_factor=5.0,
            bias_lr_factor=2.0,
            weight_decay_bias=0.0,
            optimizer_eps=1e-8,
            training=True,
            amp=False,
            amp_dtype='fp16',
            num_workers=0,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_vanilla_clip_mode_runs_with_exact_eval(self):
        model = PASModel(
            self._build_args(
                training_mode='vanilla_clip',
                use_prototype_bank=False,
                use_image_conditioned_pooling=False,
                use_custom_projector=False,
                token_policy='eos_only',
                retrieval_mode='clip_bidirectional',
                use_loss_proxy_image=False,
                use_loss_proxy_text=False,
                use_loss_proxy_text_exact=False,
                use_loss_align=False,
                use_loss_diag=False,
                use_loss_support=False,
                use_balancing_loss=False,
                use_diversity_loss=False,
                retrieval_scorer='exact',
            ),
            num_classes=2,
        )
        outputs = model(self.batch, return_debug=False)
        self.assertIn('loss_ret', outputs)
        self.assertEqual(tuple(outputs['alpha'].shape), (4, 0))
        evaluator = Evaluator(self.img_loader, self.text_loader, self._build_args(
            training_mode='vanilla_clip',
            use_prototype_bank=False,
            use_image_conditioned_pooling=False,
            use_custom_projector=False,
            token_policy='eos_only',
            retrieval_mode='clip_bidirectional',
            use_loss_proxy_image=False,
            use_loss_proxy_text=False,
            use_loss_proxy_text_exact=False,
            use_loss_align=False,
            use_loss_diag=False,
            use_loss_support=False,
            use_balancing_loss=False,
            use_diversity_loss=False,
            retrieval_scorer='exact',
        ))
        self.assertTrue(torch.isfinite(torch.tensor(evaluator.eval(model.eval()))))

    def test_itself_host_only_mode_runs_with_exact_eval(self):
        args = self._build_args(
            stage='stage1',
            host_type='itself',
            use_prototype_branch=False,
            use_prototype_bank=False,
            use_image_conditioned_pooling=False,
            retrieval_scorer='exact',
            use_loss_proxy_image=False,
            use_loss_proxy_text=False,
            use_loss_proxy_text_exact=False,
            use_loss_align=False,
            use_loss_diag=False,
            use_loss_support=False,
            use_balancing_loss=False,
            use_diversity_loss=False,
        )
        model = PASModel(args, num_classes=2)
        outputs = model(self.batch, return_debug=False)
        self.assertTrue('loss_host' in outputs or 'tal_loss' in outputs or 'cid_loss' in outputs)
        if 'alpha' in outputs:
            self.assertEqual(tuple(outputs['alpha'].shape), (4, 0))
        evaluator = Evaluator(self.img_loader, self.text_loader, args)
        self.assertTrue(torch.isfinite(torch.tensor(evaluator.eval(model.eval()))))

    def test_itself_host_plus_prototype_mode_runs_with_exact_eval(self):
        args = self._build_args(stage='stage1', host_type='itself', retrieval_scorer='exact')
        model = PASModel(args, num_classes=2)
        outputs = model(self.batch, return_debug=False)
        self.assertIn('loss_host', outputs)
        self.assertIn('loss_proto_total', outputs)
        self.assertEqual(tuple(outputs['alpha'].shape), (4, 4))
        evaluator = Evaluator(self.img_loader, self.text_loader, args)
        self.assertTrue(torch.isfinite(torch.tensor(evaluator.eval(model.eval()))))

    def test_stage0_itself_optimizer_matches_original_policy(self):
        args = self._build_args(
            stage='stage0',
            host_type='itself',
            use_prototype_branch=False,
            use_prototype_bank=False,
            use_image_conditioned_pooling=False,
            use_loss_proxy_image=False,
            use_loss_proxy_text=False,
            use_loss_proxy_text_exact=False,
            use_loss_align=False,
            use_loss_diag=False,
            use_loss_support=False,
            use_balancing_loss=False,
            use_diversity_loss=False,
            optimizer='Adam',
            lr=1e-5,
            weight_decay=4e-5,
            lr_factor=5.0,
            bias_lr_factor=2.0,
            weight_decay_bias=0.0,
            optimizer_eps=1e-3,
        )
        model = PASModel(args, num_classes=2)
        optimizer = build_optimizer(args, model)

        self.assertAlmostEqual(optimizer.defaults['eps'], 1e-3)
        groups_by_name = {group['name']: group for group in optimizer.param_groups}
        self.assertAlmostEqual(groups_by_name['stage0_itself::classifier']['lr'], 5e-5)
        self.assertAlmostEqual(groups_by_name['stage0_itself::classifier_bias']['lr'], 5e-5)
        self.assertAlmostEqual(groups_by_name['stage0_itself::classifier_bias']['weight_decay'], 0.0)
        self.assertAlmostEqual(groups_by_name['stage0_itself::visual_embedding']['lr'], 1e-3)
        self.assertAlmostEqual(groups_by_name['stage0_itself::visual_embedding_bias']['lr'], 1e-3)
        self.assertAlmostEqual(groups_by_name['stage0_itself::visual_embedding_bias']['weight_decay'], 0.0)
        self.assertAlmostEqual(groups_by_name['stage0_itself::base']['lr'], 1e-5)
        self.assertAlmostEqual(groups_by_name['stage0_itself::base']['weight_decay'], 4e-5)

        optimizer_param_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group['params']
        }
        model_param_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        self.assertSetEqual(optimizer_param_ids, model_param_ids)

    def test_pas_optimizer_uses_module_specific_lr_groups(self):
        args = self._build_args(stage='stage1')
        model = PASModel(args, num_classes=2)
        optimizer = build_optimizer(args, model)

        groups_by_name = {group['name']: group for group in optimizer.param_groups}
        self.assertAlmostEqual(groups_by_name['prototype_bank']['lr'], 0.02)
        self.assertAlmostEqual(groups_by_name['prototype_bank']['weight_decay'], 0.02)
        self.assertAlmostEqual(groups_by_name['prototype_projectors']['lr'], 0.04)
        self.assertAlmostEqual(groups_by_name['prototype_projectors']['weight_decay'], 0.04)
        self.assertAlmostEqual(groups_by_name['prototype_pooling']['lr'], 0.06)
        self.assertAlmostEqual(groups_by_name['prototype_pooling']['weight_decay'], 0.09)
        self.assertAlmostEqual(groups_by_name['class_proxies']['lr'], 0.03)
        self.assertAlmostEqual(groups_by_name['class_proxies']['weight_decay'], 0.07)

        optimizer_param_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group['params']
        }
        model_param_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        self.assertSetEqual(optimizer_param_ids, model_param_ids)

    def test_forward_returns_surrogate_retrieval_outputs(self):
        model = PASModel(self._build_args(stage='stage1'), num_classes=2)
        outputs = model(self.batch, return_debug=False)
        self.assertIn('loss_ret', outputs)
        self.assertIn('surrogate_pairwise_logits', outputs)
        self.assertEqual(tuple(outputs['surrogate_pairwise_logits'].shape), (4, 4))
        self.assertTrue(torch.isfinite(outputs['loss_total']))

    def test_surrogate_pairwise_logits_receive_gradients(self):
        model = PASModel(self._build_args(stage='stage1'), num_classes=2)
        outputs = model(self.batch, return_debug=False)
        outputs['surrogate_pairwise_logits'].retain_grad()
        outputs['loss_total'].backward()
        self.assertIsNotNone(outputs['surrogate_pairwise_logits'].grad)

    def test_stage2_allows_missing_checkpoint(self):
        model = PASModel(self._build_args(stage='stage2', finetune=''), num_classes=2)
        self.assertIsNotNone(model)

    def test_stage2_freezes_prototype_side(self):
        model = PASModel(self._build_args(stage='stage2', finetune='runs/stage1/best.pth'), num_classes=2)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.prototype_head.parameters()))
        optimizer = build_optimizer(self._build_args(stage='stage2', finetune='runs/stage1/best.pth'), model)
        groups = {group['name']: group for group in optimizer.param_groups}
        self.assertNotIn('prototype_bank', groups)
        self.assertNotIn('prototype_projectors', groups)
        self.assertNotIn('prototype_routing', groups)
        self.assertNotIn('prototype_pooling', groups)
        self.assertNotIn('prototype_contextualization', groups)
        self.assertNotIn('class_proxies', groups)
        self.assertIn('image_backbone', groups)
        self.assertIn('text_backbone', groups)

    def test_evaluator_runs_exact_and_approximate(self):
        exact_args = self._build_args(stage='stage1', retrieval_scorer='exact')
        exact_model = PASModel(exact_args, num_classes=2)
        exact_evaluator = Evaluator(self.img_loader, self.text_loader, exact_args)
        self.assertTrue(torch.isfinite(torch.tensor(exact_evaluator.eval(exact_model.eval()))))
        self.assertIn('val/retrieval/R1', exact_evaluator.latest_metrics)

        approx_args = self._build_args(stage='stage1', retrieval_scorer='approximate')
        approx_model = PASModel(approx_args, num_classes=2)
        approx_evaluator = Evaluator(self.img_loader, self.text_loader, approx_args)
        self.assertTrue(torch.isfinite(torch.tensor(approx_evaluator.eval(approx_model.eval()))))
        self.assertIn('val/retrieval/R1', approx_evaluator.latest_metrics)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
