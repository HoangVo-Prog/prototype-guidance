import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment-dependent
    torch = None
    class _NNPlaceholder:  # pragma: no cover - test import guard
        class Module:
            pass
    nn = _NNPlaceholder

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if torch is not None:
    import model.build as model_build
    from model.fusion import ResidualScoreFusion
    from model.interface_contract import HostExportPolicy, HostPluginInterface, build_host_plugin_interface
    from model.plug_and_play import PASRuntimeModel
    from model.runtime_modes import RUNTIME_MODE_CALIBRATION_ONLY
    from model.runtime_modes import (
        RUNTIME_MODE_FUSED_EXTERNAL,
        RUNTIME_MODE_HOST_ONLY,
        RUNTIME_MODE_JOINT_TRAINING,
        RUNTIME_MODE_PROTOTYPE_ONLY,
        resolve_runtime_mode_from_args,
    )
    from processor.processor import _apply_runtime_mode_trainability
    from solver.build import build_optimizer
    from utils.metrics import Evaluator
    from utils.modular_checkpoint import ModularCheckpointManager


@unittest.skipUnless(torch is not None, 'Torch is required for structural split runtime tests.')
class StructuralSplitRuntimeTests(unittest.TestCase):
    class _DummyTokenMaskBuilder:
        def build_valid_mask(self, token_ids):
            return torch.ones_like(token_ids, dtype=torch.bool)

        def get_special_token_positions(self, token_ids, attention_mask=None):
            del attention_mask
            eos = token_ids.argmax(dim=-1)
            cls = torch.zeros_like(eos)
            return {'eos': eos, 'cls': cls}

    class _DummyBaseModel(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = int(dim)
            self.visual = nn.Linear(dim, dim, bias=False)
            self.transformer = nn.Linear(dim, dim, bias=False)
            self.token_embedding = nn.Embedding(128, dim)
            self.positional_embedding = nn.Parameter(torch.zeros(77, dim))
            self.ln_final = nn.LayerNorm(dim)
            self.text_projection = nn.Parameter(torch.eye(dim))

        def encode_image_intermediates(self, image, return_all=False, average_attn_weights=True):
            del return_all, average_attn_weights
            projected = self.visual(image.float())
            local = projected * 0.5
            projected_tokens = torch.stack([projected, local], dim=1)
            return {
                'projected_tokens': projected_tokens,
                'pre_projection_tokens': projected_tokens.clone(),
                'attention_weights': None,
            }

        def encode_text_intermediates(self, text, return_all=False, average_attn_weights=True):
            del return_all, average_attn_weights
            embedded = self.token_embedding(text.long())
            projected_tokens = self.transformer(embedded)
            return {
                'projected_tokens': projected_tokens,
                'pre_projection_tokens': projected_tokens.clone(),
                'attention_weights': None,
            }

    class _DummyHostHead(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.image_projector = nn.Linear(dim, dim, bias=False)
            self.text_projector = nn.Linear(dim, dim, bias=False)
            self.use_host_loss = True

        def encode_image_branch(self, image_output, return_debug=False):
            del return_debug
            image_projected = self.image_projector(image_output.projected_pooled)
            routing_weights = torch.softmax(image_projected[:, :2], dim=-1)
            return {
                'image_projected': image_projected,
                'image_projected_raw': image_projected,
                'summary': image_output.projected_pooled,
                'routing_weights': routing_weights,
            }

        def encode_text_branch(self, text_output, token_ids, return_debug=False, current_step=None):
            del token_ids, return_debug, current_step
            text_projected = self.text_projector(text_output.projected_pooled)
            return {
                'text_projected': text_projected,
                'text_projected_raw': text_projected,
                'surrogate_text_projected': text_projected,
                'surrogate_text_projected_raw': text_projected,
                'exact_text_projected': text_projected,
                'exact_text_projected_raw': text_projected,
            }

        def compute_similarity_matrix(self, image_features, text_features):
            image = image_features.get('host_image_projected', image_features['image_projected'])
            text = text_features.get('host_text_projected', text_features['text_projected'])
            return text @ image.t()

        def forward(self, image_output, text_output, token_ids, pids=None, return_debug=False, current_step=None):
            del pids, return_debug, current_step
            image_features = self.encode_image_branch(image_output, return_debug=False)
            text_features = self.encode_text_branch(text_output, token_ids, return_debug=False)
            pairwise = image_features['image_projected'] @ text_features['text_projected'].t()
            loss = pairwise.diag().mean()
            losses = {
                'loss_total': loss,
                'loss_ret': loss,
                'loss_ret_i2t': loss,
                'loss_ret_t2i': loss,
                'loss_cid': loss.new_zeros(()),
                'retrieval_temperature': loss.new_tensor(0.07),
                'logit_scale': loss.new_tensor(1.0 / 0.07),
            }
            outputs = dict(image_features)
            outputs.update(text_features)
            outputs['surrogate_pairwise_logits'] = pairwise
            outputs['host_similarity_logits'] = pairwise
            outputs['losses'] = losses
            outputs['metrics'] = {}
            return outputs

    class _DummyPrototypeHead(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = int(dim)
            self.image_projector = nn.Linear(dim, dim, bias=False)
            self.text_projector = nn.Linear(dim, dim, bias=False)
            self.prototype_bank = nn.Parameter(torch.randn(2, dim))
            self.text_pool_query = nn.Parameter(torch.randn(dim))
            self.last_inputs = {}

        def _loss_dict(self, reference, loss_total):
            zero = reference.new_zeros(())
            return {
                'loss_total': loss_total,
                'loss_proxy': zero,
                'loss_proxy_image': zero,
                'loss_proxy_text': zero,
                'loss_proxy_text_exact': zero,
                'loss_ret': loss_total,
                'loss_align': zero,
                'loss_dir': zero,
                'loss_gap': zero,
                'loss_sup': zero,
                'loss_diag': zero,
                'loss_support': zero,
                'loss_diversity': zero,
                'loss_balance': zero,
                'loss_proxy_image_weighted': zero,
                'loss_proxy_text_weighted': zero,
                'loss_proxy_text_exact_weighted': zero,
                'loss_proxy_weighted': zero,
                'loss_ret_weighted': loss_total,
                'loss_weight_ret': zero,
                'loss_weight_ret_weighted': zero,
                'loss_align_weighted': zero,
                'loss_dir_weighted': zero,
                'loss_gap_weighted': zero,
                'loss_sup_weighted': zero,
                'loss_diag_weighted': zero,
                'loss_support_weighted': zero,
                'loss_diversity_weighted': zero,
                'loss_balance_weighted': zero,
                'lambda_proxy': zero,
                'lambda_proxy_image': zero,
                'lambda_proxy_text': zero,
                'lambda_proxy_text_exact': zero,
                'use_loss_proxy_text_exact': zero,
                'use_loss_ret': reference.new_tensor(1.0),
                'lambda_ret': reference.new_tensor(1.0),
                'use_loss_weight_ret': zero,
                'lambda_weight_ret': zero,
                'weight_ret_margin_delta': zero,
                'weight_ret_tau': zero,
                'weight_ret_detach_host': zero,
                'weight_ret_normalize_mean_one': zero,
                'lambda_align': zero,
                'use_loss_dir': zero,
                'lambda_dir': zero,
                'use_loss_gap': zero,
                'lambda_gap': zero,
                'use_loss_sup': zero,
                'lambda_sup': zero,
                'lambda_diag': zero,
                'use_loss_support': zero,
                'lambda_support': zero,
                'lambda_div': zero,
                'lambda_bal': zero,
                'prototype_gap_margin': zero,
                'prototype_support_target': zero,
                'proxy_temperature': zero,
                'diag_temperature': zero,
                'retrieval_temperature': reference.new_tensor(0.07),
                'logit_scale': reference.new_tensor(1.0 / 0.07),
                'debug_metrics': {},
            }

        def encode_image_branch(self, image_embeddings, image_local_tokens=None, return_debug=False):
            del image_local_tokens, return_debug
            image_projected = self.image_projector(image_embeddings)
            routing_weights = torch.softmax(image_projected[:, :2], dim=-1)
            return {
                'image_embedding': image_embeddings,
                'summary': image_embeddings,
                'routing_weights': routing_weights,
                'image_projected': image_projected,
                'image_projected_raw': image_projected,
                'image_proxy_features': image_embeddings,
                'router_debug': {},
                'image_projector_debug': {'projected_features_raw': image_projected},
                'debug': {},
            }

        def get_prototype_context(self, return_debug=False):
            del return_debug
            return {
                'prototypes': self.prototype_bank,
                'contextualized_prototypes': self.prototype_bank,
                'contextualizer_debug': {},
                'debug': {},
            }

        def build_text_basis_bank(
            self,
            text_token_states,
            token_ids,
            contextualized_prototypes,
            attention_mask=None,
            special_token_positions=None,
            return_debug=False,
            prepared_text=None,
        ):
            del token_ids, attention_mask, special_token_positions, return_debug, prepared_text
            batch = text_token_states.size(0)
            basis_bank = contextualized_prototypes.unsqueeze(0).expand(batch, -1, -1)
            return {'basis_bank': basis_bank}

        def compute_pairwise_similarity(
            self,
            image_projected,
            summaries,
            text_token_states,
            token_ids,
            pids=None,
            attention_mask=None,
            special_token_positions=None,
            image_chunk_size=32,
            text_chunk_size=128,
        ):
            del summaries, token_ids, pids, attention_mask, special_token_positions, image_chunk_size, text_chunk_size
            text = self.text_projector(text_token_states[:, 0, :])
            return text @ image_projected.t()

        def compute_approximate_pairwise_similarity(
            self,
            image_projected,
            routing_weights,
            basis_bank,
            image_chunk_size=32,
            text_chunk_size=128,
        ):
            del image_chunk_size, text_chunk_size
            # [I, P] x [T, P, D] -> [T, I, D]
            surrogate = torch.einsum('ip,tpd->tid', routing_weights, basis_bank)
            expanded_image = image_projected.unsqueeze(0).expand(surrogate.size(0), -1, -1)
            return (surrogate * expanded_image).sum(dim=-1)

        def pool_text_with_summary(
            self,
            summaries,
            text_token_states,
            token_ids,
            attention_mask=None,
            special_token_positions=None,
            return_debug=False,
        ):
            del summaries, token_ids, attention_mask, special_token_positions, return_debug
            pooled = text_token_states[:, 0, :]
            text_projected = self.text_projector(pooled)
            return {
                'text_projected': text_projected,
                'text_projected_raw': text_projected,
            }

        def forward(
            self,
            image_embeddings,
            text_token_states,
            token_ids,
            image_local_tokens=None,
            pids=None,
            attention_mask=None,
            special_token_positions=None,
            host_pairwise_logits=None,
            return_debug=False,
            disable_proxy_losses=False,
        ):
            del pids, attention_mask, special_token_positions, return_debug, disable_proxy_losses
            self.last_inputs = {
                'image_embeddings': image_embeddings,
                'text_token_states': text_token_states,
                'token_ids': token_ids,
                'image_local_tokens': image_local_tokens,
                'host_pairwise_logits': host_pairwise_logits,
            }
            image_projected = self.image_projector(image_embeddings)
            text_projected = self.text_projector(text_token_states[:, 0, :])
            routing_weights = torch.softmax(image_projected[:, :2], dim=-1)
            basis_bank = self.prototype_bank.unsqueeze(0).expand(text_projected.size(0), -1, -1)
            pairwise = image_projected @ text_projected.t()
            loss_total = pairwise.diag().mean()
            losses = self._loss_dict(image_projected, loss_total)
            return {
                'routing_weights': routing_weights,
                'summary': image_embeddings,
                'prototypes': self.prototype_bank,
                'contextualized_prototypes': self.prototype_bank,
                'basis_bank': basis_bank,
                'token_valid_mask': torch.ones_like(token_ids, dtype=torch.bool),
                'token_keep_mask': torch.ones_like(token_ids, dtype=torch.bool),
                'beta_logits_masked': torch.zeros_like(token_ids, dtype=image_projected.dtype),
                'exact_token_weights': torch.zeros_like(token_ids, dtype=image_projected.dtype),
                'surrogate_pooled_text': text_projected,
                'exact_pooled_text': text_projected,
                'surrogate_text_projected': text_projected,
                'surrogate_text_projected_raw': text_projected,
                'exact_text_projected': text_projected,
                'exact_text_projected_raw': text_projected,
                'image_projected': image_projected,
                'image_projected_raw': image_projected,
                'losses': losses,
                'metrics': {},
                'debug': {},
                'surrogate_pairwise_logits': pairwise,
            }

    class _DummyLegacyModel(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.base_model = StructuralSplitRuntimeTests._DummyBaseModel(dim)
            self.host_head = StructuralSplitRuntimeTests._DummyHostHead(dim)
            self.prototype_head = StructuralSplitRuntimeTests._DummyPrototypeHead(dim)
            self.token_mask_builder = StructuralSplitRuntimeTests._DummyTokenMaskBuilder()
            self.fusion_module = ResidualScoreFusion(
                enabled=True,
                lambda_host=0.6,
                lambda_prototype=0.4,
                coefficient_source='fixed',
            )
            self.host_type = 'clip'
            self.lambda_host = 1.0
            self.use_host_loss = True
            self.use_prototype_branch = True
            self.use_prototype_bank = True
            self.use_image_conditioned_pooling = True
            self.return_debug_outputs = False
            self.fusion_enabled = True
            self.fusion_lambda_host = 0.6
            self.fusion_lambda_prototype = 0.4
            self.fusion_coefficient = 0.4
            self.prototype_dim = dim
            self.image_backbone = 'clip_visual'
            self.text_backbone = 'clip_text_transformer'
            self.backbone_precision = 'fp32'
            self.prototype_precision = 'fp32'
            self.itself_return_all = False
            self.itself_average_attn_weights = True
            self.prototype_eval_image_chunk_size = 16
            self.prototype_eval_text_chunk_size = 16

    def _runtime_args(self, runtime_mode='joint_training', training=True):
        return SimpleNamespace(
            runtime_mode=runtime_mode,
            training=training,
            use_prototype_branch=True,
            use_prototype_bank=True,
            use_image_conditioned_pooling=True,
            host_type='clip',
            pretrain_choice='ViT-B/16',
            img_size=(4, 4),
            stride_size=4,
            image_backbone='clip_visual',
            text_backbone='clip_text_transformer',
            backbone_precision='fp32',
            prototype_precision='fp32',
            return_debug_outputs=False,
            fusion_enabled=True,
            fusion_lambda_host=0.6,
            fusion_lambda_prototype=0.4,
            fusion_coefficient_source='fixed',
            lambda_host=1.0,
            output_dir='.',
            log_debug_metrics=True,
            amp=False,
            amp_dtype='fp16',
            retrieval_scorer='exact',
            retrieval_metrics=['R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum'],
            fusion_eval_subsets=None,
            config_data={},
            run_name='unit-test',
            training_stage='joint',
            optimizer='AdamW',
            lr=1e-3,
            lr_prototype_bank=1e-3,
            lr_projectors=1e-3,
            lr_prototype_routing=1e-3,
            lr_prototype_pooling=1e-3,
            lr_prototype_contextualization=1e-3,
            lr_host_projectors=1e-3,
            lr_class_proxies=1e-3,
            lr_image_backbone=0.0,
            lr_text_backbone=0.0,
            weight_decay=1e-2,
            weight_decay_prototype_bank=1e-2,
            weight_decay_projectors=1e-2,
            weight_decay_prototype_routing=1e-2,
            weight_decay_prototype_pooling=1e-2,
            weight_decay_prototype_contextualization=1e-2,
            weight_decay_host_projectors=1e-2,
            weight_decay_class_proxies=1e-2,
            weight_decay_image_backbone=0.0,
            weight_decay_text_backbone=0.0,
            alpha=0.9,
            beta=0.999,
            momentum=0.9,
            optimizer_eps=1e-8,
        )

    def _batch(self, requires_grad=True):
        images = torch.randn(2, 4, requires_grad=requires_grad)
        caption_ids = torch.tensor([[1, 2, 3], [1, 3, 2]], dtype=torch.long)
        pids = torch.tensor([0, 1], dtype=torch.long)
        return {'images': images, 'caption_ids': caption_ids, 'pids': pids}

    @staticmethod
    def _snapshot_params(named_params, prefix):
        return {
            name: param.detach().clone()
            for name, param in named_params
            if name.startswith(prefix)
        }

    @staticmethod
    def _assert_params_unchanged(testcase, before, model, prefix):
        for name, parameter in model.named_parameters():
            if not name.startswith(prefix):
                continue
            testcase.assertTrue(torch.allclose(parameter.detach(), before[name]))

    def test_interface_contract_detach_and_host_logit_guard(self):
        image_embeddings = torch.randn(2, 4, requires_grad=True)
        text_states = torch.randn(2, 3, 4, requires_grad=True)
        token_ids = torch.tensor([[1, 2, 3], [1, 3, 2]], dtype=torch.long)
        host_logits = torch.randn(2, 2, requires_grad=True)
        with self.assertRaisesRegex(ValueError, 'forbids exporting host_pairwise_logits'):
            build_host_plugin_interface(
                image_embeddings=image_embeddings,
                text_token_states=text_states,
                token_ids=token_ids,
                attention_mask=torch.ones_like(token_ids, dtype=torch.bool),
                special_token_positions={'eos': token_ids.argmax(dim=-1)},
                image_local_tokens=torch.randn(2, 2, 4, requires_grad=True),
                host_pairwise_logits=host_logits,
                policy=HostExportPolicy(detach=True, allow_host_pairwise_logits=False, include_image_local_tokens=True),
                metadata={'runtime_mode': 'prototype_only'},
            )
        interface = build_host_plugin_interface(
            image_embeddings=image_embeddings,
            text_token_states=text_states,
            token_ids=token_ids,
            attention_mask=torch.ones_like(token_ids, dtype=torch.bool),
            special_token_positions={'eos': token_ids.argmax(dim=-1)},
            image_local_tokens=torch.randn(2, 2, 4, requires_grad=True),
            host_pairwise_logits=host_logits,
            policy=HostExportPolicy(detach=True, allow_host_pairwise_logits=True, include_image_local_tokens=True),
            metadata={'runtime_mode': 'prototype_only'},
        )
        self.assertFalse(interface.image_embeddings.requires_grad)
        self.assertFalse(interface.text_token_states.requires_grad)
        self.assertFalse(interface.host_pairwise_logits.requires_grad)

    def test_interface_contract_validation_rejects_missing_artifact_and_wrong_version(self):
        image_embeddings = torch.randn(2, 4)
        text_states = torch.randn(2, 3, 4)
        token_ids = torch.tensor([[1, 2, 3], [1, 3, 2]], dtype=torch.long)
        bad_missing_eos = HostPluginInterface(
            image_embeddings=image_embeddings,
            text_token_states=text_states,
            token_ids=token_ids,
            attention_mask=torch.ones_like(token_ids, dtype=torch.bool),
            special_token_positions={},
            image_local_tokens=torch.randn(2, 2, 4),
            host_pairwise_logits=None,
            version='host_export_v1',
            metadata={},
        )
        with self.assertRaisesRegex(ValueError, 'must include `eos`'):
            bad_missing_eos.validate()

        bad_version = HostPluginInterface(
            image_embeddings=image_embeddings,
            text_token_states=text_states,
            token_ids=token_ids,
            attention_mask=torch.ones_like(token_ids, dtype=torch.bool),
            special_token_positions={'eos': token_ids.argmax(dim=-1)},
            image_local_tokens=torch.randn(2, 2, 4),
            host_pairwise_logits=None,
            version='host_export_v999',
            metadata={},
        )
        with self.assertRaisesRegex(ValueError, 'Unsupported HostPluginInterface.version'):
            bad_version.validate()

    def test_runtime_mode_router_defaults(self):
        host_args = SimpleNamespace(runtime_mode='auto', use_prototype_branch=False)
        proto_args = SimpleNamespace(runtime_mode='auto', use_prototype_branch=True)
        self.assertEqual(resolve_runtime_mode_from_args(host_args, for_training=True), 'host_only')
        self.assertEqual(resolve_runtime_mode_from_args(proto_args, for_training=True), 'joint_training')
        self.assertEqual(resolve_runtime_mode_from_args(proto_args, for_training=False), 'fused_external')

    def test_host_only_runtime_disables_prototype_loss_path(self):
        args = self._runtime_args(runtime_mode=RUNTIME_MODE_HOST_ONLY, training=True)
        with patch('model.plug_and_play.legacy_pas.build_model', return_value=self._DummyLegacyModel(dim=4)):
            model = PASRuntimeModel(args=args, num_classes=2, train_loader=None)
        outputs = model(self._batch(requires_grad=True))
        self.assertEqual(outputs['debug']['runtime_mode'], RUNTIME_MODE_HOST_ONLY)
        self.assertTrue(torch.allclose(outputs['loss_total'], outputs['loss_host_weighted']))
        self.assertTrue(torch.allclose(outputs['loss_proto_total'], torch.zeros_like(outputs['loss_proto_total'])))
        self.assertIsNone(outputs['surrogate_pairwise_logits'])

    def test_runtime_external_mode_detaches_and_forbids_host_logits(self):
        args = self._runtime_args(runtime_mode=RUNTIME_MODE_PROTOTYPE_ONLY, training=True)
        with patch('model.plug_and_play.legacy_pas.build_model', return_value=self._DummyLegacyModel(dim=4)):
            model = PASRuntimeModel(args=args, num_classes=2, train_loader=None)
        outputs = model(self._batch(requires_grad=True))
        self.assertEqual(outputs['debug']['runtime_mode'], RUNTIME_MODE_PROTOTYPE_ONLY)
        self.assertIsNone(model.prototype_head.last_inputs['host_pairwise_logits'])
        self.assertFalse(model.prototype_head.last_inputs['image_embeddings'].requires_grad)
        self.assertTrue(torch.allclose(outputs['loss_total'], outputs['loss_proto_total']))

    def test_runtime_joint_mode_keeps_host_to_prototype_coupling(self):
        args = self._runtime_args(runtime_mode=RUNTIME_MODE_JOINT_TRAINING, training=True)
        with patch('model.plug_and_play.legacy_pas.build_model', return_value=self._DummyLegacyModel(dim=4)):
            model = PASRuntimeModel(args=args, num_classes=2, train_loader=None)
        outputs = model(self._batch(requires_grad=True))
        expected = (model.lambda_host * outputs['loss_host']) + outputs['loss_proto_total']
        self.assertTrue(torch.allclose(outputs['loss_total'], expected))
        self.assertIsNotNone(model.prototype_head.last_inputs['host_pairwise_logits'])
        self.assertTrue(model.prototype_head.last_inputs['image_embeddings'].requires_grad)

    def test_external_mode_optimizer_excludes_host_and_host_params_stay_fixed(self):
        args = self._runtime_args(runtime_mode=RUNTIME_MODE_PROTOTYPE_ONLY, training=True)
        with patch('model.plug_and_play.legacy_pas.build_model', return_value=self._DummyLegacyModel(dim=4)):
            model = PASRuntimeModel(args=args, num_classes=2, train_loader=None)
        _apply_runtime_mode_trainability(model, RUNTIME_MODE_PROTOTYPE_ONLY, __import__('logging').getLogger('test.structural_split'))
        optimizer = build_optimizer(args, model)
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
        self.assertTrue(all(not name.startswith('host_core.') for name in trainable_names))
        host_before = self._snapshot_params(model.named_parameters(), 'host_core.')
        batch = self._batch(requires_grad=False)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        outputs['loss_total'].backward()
        for name, parameter in model.named_parameters():
            if not name.startswith('host_core.'):
                continue
            if parameter.grad is not None:
                self.assertTrue(torch.allclose(parameter.grad.detach(), torch.zeros_like(parameter.grad.detach())))
        optimizer.step()
        self._assert_params_unchanged(self, host_before, model, 'host_core.')

    def test_calibration_only_trains_composer_only(self):
        args = self._runtime_args(runtime_mode=RUNTIME_MODE_CALIBRATION_ONLY, training=True)
        with patch('model.plug_and_play.legacy_pas.build_model', return_value=self._DummyLegacyModel(dim=4)):
            model = PASRuntimeModel(args=args, num_classes=2, train_loader=None)
        _apply_runtime_mode_trainability(model, RUNTIME_MODE_CALIBRATION_ONLY, __import__('logging').getLogger('test.structural_split'))
        optimizer = build_optimizer(args, model)
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
        self.assertTrue(trainable_names)
        self.assertTrue(all(name.startswith('composer.') for name in trainable_names))
        host_before = self._snapshot_params(model.named_parameters(), 'host_core.')
        prototype_before = self._snapshot_params(model.named_parameters(), 'prototype_plugin.')
        composer_before = self._snapshot_params(model.named_parameters(), 'composer.')
        optimizer.zero_grad(set_to_none=True)
        outputs = model(self._batch(requires_grad=False), disable_proxy_losses=True)
        self.assertGreater(float(outputs['loss_composer_calibration'].detach().item()), 0.0)
        self.assertTrue(torch.allclose(outputs['loss_total'], outputs['loss_composer_calibration']))
        outputs['loss_total'].backward()
        optimizer.step()
        self._assert_params_unchanged(self, host_before, model, 'host_core.')
        self._assert_params_unchanged(self, prototype_before, model, 'prototype_plugin.')
        any_composer_changed = False
        for name, parameter in model.named_parameters():
            if not name.startswith('composer.'):
                continue
            if not torch.allclose(parameter.detach(), composer_before[name]):
                any_composer_changed = True
                break
        self.assertTrue(any_composer_changed)

    def test_build_router_prefers_structural_split_for_prototype_modes(self):
        args = self._runtime_args(runtime_mode=RUNTIME_MODE_JOINT_TRAINING, training=True)
        with patch('model.build.build_structural_split_model', return_value='split') as split_builder:
            result = model_build.build_model(args=args, num_classes=2, train_loader=None)
        self.assertEqual(result, 'split')
        self.assertEqual(split_builder.call_count, 1)

    def test_smoke_fused_eval_and_component_checkpoint_save(self):
        args = self._runtime_args(runtime_mode=RUNTIME_MODE_FUSED_EXTERNAL, training=False)
        args.config_data = {
            'checkpointing': {
                'groups': {
                    'host': {'enabled': True},
                    'prototype_bank': {'enabled': True},
                    'prototype_projector': {'enabled': True},
                    'fusion': {'enabled': True},
                },
                'save': {
                    'dir': None,
                    'save_latest': True,
                    'save_best': False,
                    'keep_last_n': 1,
                    'artifacts': {
                        'host': {'enabled': True, 'filename_latest': 'host_latest.pth', 'filename_best': 'host_best.pth'},
                        'prototype_bank': {'enabled': True, 'filename_latest': 'prototype_bank_latest.pth', 'filename_best': 'prototype_bank_best.pth'},
                        'prototype_projector': {'enabled': True, 'filename_latest': 'prototype_projector_latest.pth', 'filename_best': 'prototype_projector_best.pth'},
                        'fusion': {'enabled': True, 'filename_latest': 'fusion_latest.pth', 'filename_best': 'fusion_best.pth'},
                    },
                },
                'load': {'enabled': False, 'strict': True, 'sources': {}},
                'authority_validation': {
                    'enabled': True,
                    'strict': True,
                    'warn_only': False,
                    'allow_fallback_row_name_classification': True,
                },
            }
        }
        with patch('model.plug_and_play.legacy_pas.build_model', return_value=self._DummyLegacyModel(dim=4)):
            model = PASRuntimeModel(args=args, num_classes=2, train_loader=None)
        model.eval()

        txt_loader = [
            (torch.tensor([0], dtype=torch.long), torch.tensor([[1, 2, 3]], dtype=torch.long)),
            (torch.tensor([1], dtype=torch.long), torch.tensor([[1, 3, 2]], dtype=torch.long)),
        ]
        img_loader = [
            (torch.tensor([0], dtype=torch.long), torch.randn(1, 4)),
            (torch.tensor([1], dtype=torch.long), torch.randn(1, 4)),
        ]
        evaluator = Evaluator(img_loader=img_loader, txt_loader=txt_loader, args=args)
        top1 = evaluator.eval(model)
        self.assertIn('display_row', evaluator.latest_authority)
        self.assertIn('source_row', evaluator.latest_authority)
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = __import__('logging').getLogger('test.structural_split')
            manager_args = self._runtime_args(runtime_mode=RUNTIME_MODE_FUSED_EXTERNAL, training=False)
            manager_args.config_data = args.config_data
            manager = ModularCheckpointManager(args=manager_args, save_dir=tmpdir, logger=logger)
            manager.save_latest(
                model=model,
                epoch=1,
                global_step=1,
                metric_value=float(top1),
                metric_row=evaluator.latest_metrics.get('val/top1_source_row'),
                metric_display_row=evaluator.latest_metrics.get('val/top1_display_row'),
                metric_source_row=evaluator.latest_metrics.get('val/top1_source_row'),
                authority_context=evaluator.latest_authority,
            )
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'host_latest.pth')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'prototype_bank_latest.pth')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'prototype_projector_latest.pth')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'fusion_latest.pth')))

    def test_checkpoint_compatibility_rejects_incompatible_interface_version(self):
        args = self._runtime_args(runtime_mode=RUNTIME_MODE_FUSED_EXTERNAL, training=False)
        with patch('model.plug_and_play.legacy_pas.build_model', return_value=self._DummyLegacyModel(dim=4)):
            model = PASRuntimeModel(args=args, num_classes=2, train_loader=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_cfg = {
                'checkpointing': {
                    'groups': {
                        'host': {'enabled': False},
                        'prototype_bank': {'enabled': True},
                        'prototype_projector': {'enabled': False},
                        'fusion': {'enabled': False},
                    },
                    'save': {
                        'dir': tmpdir,
                        'save_latest': True,
                        'save_best': False,
                        'keep_last_n': 1,
                        'artifacts': {
                            'host': {'enabled': False, 'filename_latest': 'host_latest.pth', 'filename_best': 'host_best.pth'},
                            'prototype_bank': {'enabled': True, 'filename_latest': 'prototype_bank_latest.pth', 'filename_best': 'prototype_bank_best.pth'},
                            'prototype_projector': {'enabled': False, 'filename_latest': 'prototype_projector_latest.pth', 'filename_best': 'prototype_projector_best.pth'},
                            'fusion': {'enabled': False, 'filename_latest': 'fusion_latest.pth', 'filename_best': 'fusion_best.pth'},
                        },
                    },
                    'load': {'enabled': False, 'strict': True, 'sources': {}},
                    'authority_validation': {'enabled': True, 'strict': True, 'warn_only': False, 'allow_fallback_row_name_classification': True},
                }
            }
            manager_args = self._runtime_args(runtime_mode=RUNTIME_MODE_FUSED_EXTERNAL, training=False)
            manager_args.config_data = save_cfg
            manager = ModularCheckpointManager(args=manager_args, save_dir=tmpdir, logger=__import__('logging').getLogger('test.structural_split'))
            manager.save_latest(
                model=model,
                epoch=1,
                global_step=1,
                metric_value=10.0,
                metric_row='prototype-t2i',
                metric_display_row='prototype-t2i',
                metric_source_row='prototype-t2i',
                authority_context={
                    'display_row': 'prototype-t2i',
                    'source_row': 'prototype-t2i',
                    'row_roles': {'prototype-t2i': 'prototype'},
                    'row_metrics': {'prototype-t2i': {'R1': 10.0}},
                    'candidates': {'host': None, 'prototype': 'prototype-t2i', 'fused': None},
                },
            )
            path = os.path.join(tmpdir, 'prototype_bank_latest.pth')
            payload = torch.load(path, map_location='cpu')
            payload['metadata']['host_export_interface_version'] = 'host_export_v999'
            payload.setdefault('metadata', {}).setdefault('compatibility', {})['host_export_interface_version'] = 'host_export_v999'
            torch.save(payload, path)

            load_cfg = {
                'checkpointing': {
                    'groups': save_cfg['checkpointing']['groups'],
                    'save': save_cfg['checkpointing']['save'],
                    'load': {
                        'enabled': True,
                        'strict': True,
                        'sources': {
                            'host': {'enabled': False, 'path': None},
                            'prototype_bank': {'enabled': True, 'path': path},
                            'prototype_projector': {'enabled': False, 'path': None},
                            'fusion': {'enabled': False, 'path': None},
                        },
                    },
                    'authority_validation': save_cfg['checkpointing']['authority_validation'],
                }
            }
            load_args = self._runtime_args(runtime_mode=RUNTIME_MODE_FUSED_EXTERNAL, training=False)
            load_args.config_data = load_cfg
            load_manager = ModularCheckpointManager(args=load_args, save_dir=tmpdir, logger=__import__('logging').getLogger('test.structural_split'))
            with self.assertRaisesRegex(RuntimeError, 'interface version incompatible'):
                load_manager.load_configured_groups(model)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
