import os
import sys
import tempfile
import unittest

import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.config import load_yaml_config
from utils.options import get_args


class ConfigSurfaceTests(unittest.TestCase):
    def _write_config(self, payload):
        handle = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
        with handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        self.addCleanup(lambda: os.path.exists(handle.name) and os.remove(handle.name))
        return handle.name

    def test_removed_masking_knobs_are_rejected(self):
        path = self._write_config({'text_pooling': {'exclude_special_tokens': True}})
        with self.assertRaisesRegex(ValueError, 'text_pooling.exclude_special_tokens'):
            load_yaml_config(None, path)

    def test_removed_pooling_mode_is_rejected(self):
        path = self._write_config({'model': {'pooling_mode': 'image_conditioned'}})
        with self.assertRaisesRegex(ValueError, 'model.pooling_mode'):
            load_yaml_config(None, path)

    def test_removed_contextualizer_optimizer_surface_is_rejected(self):
        path = self._write_config({'optimizer': {'lr_contextualizer': 1e-3}})
        with self.assertRaisesRegex(ValueError, 'optimizer.lr_contextualizer'):
            load_yaml_config(None, path)

    def test_removed_logit_scale_surface_is_rejected(self):
        path = self._write_config({'model': {'learn_logit_scale': True}})
        with self.assertRaisesRegex(ValueError, 'model.learn_logit_scale'):
            load_yaml_config(None, path)

    def test_removed_logit_scale_optimizer_surface_is_rejected(self):
        path = self._write_config({'optimizer': {'lr_logit_scale': 1e-4}})
        with self.assertRaisesRegex(ValueError, 'optimizer.lr_logit_scale'):
            load_yaml_config(None, path)

    def test_removed_sparse_routing_knob_fails_loudly(self):
        path = self._write_config({'prototype': {'assignment_sparse': True}})
        with self.assertRaisesRegex(ValueError, 'prototype.assignment_sparse'):
            load_yaml_config(None, path)

    def test_invalid_enum_value_fails_at_config_load_time(self):
        path = self._write_config({'prototype': {'routing_type': 'consine'}})
        with self.assertRaisesRegex(ValueError, 'prototype.routing_type'):
            load_yaml_config(None, path)

    def test_invalid_retrieval_metric_fails_at_config_load_time(self):
        path = self._write_config({'evaluation': {'retrieval_metrics': ['R1', 'bad_metric']}})
        with self.assertRaisesRegex(ValueError, 'evaluation.retrieval_metrics'):
            load_yaml_config(None, path)

    def test_invalid_retrieval_scorer_fails_at_config_load_time(self):
        path = self._write_config({'evaluation': {'retrieval_scorer': 'surrogate_default'}})
        with self.assertRaisesRegex(ValueError, 'evaluation.retrieval_scorer'):
            load_yaml_config(None, path)

    def test_invalid_cli_enum_fails_during_arg_parsing_validation(self):
        with self.assertRaisesRegex(ValueError, 'prototype_routing_type'):
            get_args(['--routing_similarity', 'consine'])

    def test_cli_flags_override_config_file_values(self):
        path = self._write_config(
            {
                'training': {
                    'batch_size': 32,
                    'epochs': 60,
                    'amp': False,
                },
                'logging': {
                    'use_wandb': True,
                    'project': 'from_config',
                },
                'model': {
                    'projection_dim': 256,
                },
            }
        )
        args = get_args([
            '--config_file', path,
            '--batch_size', '7',
            '--epochs', '11',
            '--amp', 'true',
            '--use_wandb', 'false',
            '--wandb_project', 'from_cli',
            '--projection_dim', '128',
        ])
        self.assertEqual(args.batch_size, 7)
        self.assertEqual(args.num_epoch, 11)
        self.assertTrue(args.amp)
        self.assertFalse(args.use_wandb)
        self.assertEqual(args.wandb_project, 'from_cli')
        self.assertEqual(args.projection_dim, 128)

    def test_nohup_flag_parses_from_cli(self):
        args = get_args(['--nohup'])
        self.assertTrue(args.nohup)

    def test_new_prototype_init_surface_loads(self):
        path = self._write_config(
            {
                'experiment': {
                    'seed': 19,
                },
                'prototype': {
                    'prototype_init': 'hybrid_spherical_kmeans_random',
                    'prototype_init_path': '/path/to/features.pt',
                    'prototype_init_hybrid_ratio': 0.25,
                    'prototype_init_max_iters': 11,
                    'prototype_init_tol': 5e-4,
                    'prototype_init_seed': 13,
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertEqual(args.prototype_init, 'hybrid_spherical_kmeans_random')
        self.assertEqual(args.prototype_init_path, '/path/to/features.pt')
        self.assertEqual(args.prototype_init_hybrid_ratio, 0.25)
        self.assertEqual(args.prototype_init_max_iters, 11)
        self.assertEqual(args.prototype_init_tol, 5e-4)
        self.assertEqual(args.prototype_init_seed, 13)

    def test_valid_runtime_surface_loads_special_token_ids_precision_and_amp_knobs(self):
        path = self._write_config(
            {
                'model': {
                    'projector_type': 'linear',
                    'normalize_projector_outputs': True,
                    'backbone_precision': 'fp32',
                    'prototype_precision': 'fp32',
                },
                'prototype': {
                    'normalize_for_self_interaction': True,
                    'normalize_for_routing': True,
                },
                'loss': {
                    'use_balancing_loss': True,
                    'balance_loss_weight': 0.1,
                    'use_diversity_loss': True,
                    'diversity_loss_weight': 0.01,
                    'lambda_proxy': 1.0,
                    'use_loss_proxy_image': True,
                    'use_loss_proxy_text': True,
                    'use_loss_proxy_text_exact': True,
                    'use_loss_align': False,
                    'lambda_align': 0.5,
                    'use_loss_diag': False,
                    'lambda_diag': 0.25,
                    'use_loss_support': True,
                    'lambda_support': 0.15,
                    'support_min': 2.5,
                },
                'text_pooling': {
                    'token_policy': 'content_only',
                    'scoring_type': 'cosine',
                    'normalize_for_token_scoring': True,
                    'token_temperature': 0.07,
                    'special_token_ids': {
                        'bos_token_id': 49406,
                        'eos_token_id': 49407,
                        'pad_token_id': 0,
                    },
                    'error_on_empty_kept_tokens': True,
                },
                'training': {
                    'amp': True,
                    'amp_dtype': 'bf16',
                    'proxy_temperature': 0.2,
                },
                'optimizer': {
                    'lr_class_proxies': 0.003,
                    'weight_decay_class_proxies': 0.02,
                },
                'evaluation': {
                    'retrieval_scorer': 'approximate',
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertEqual(args.projector_type, 'linear')
        self.assertTrue(args.normalize_projector_outputs)
        self.assertEqual(args.backbone_precision, 'fp32')
        self.assertEqual(args.prototype_precision, 'fp32')
        self.assertTrue(args.normalize_for_self_interaction)
        self.assertTrue(args.normalize_for_routing)
        self.assertTrue(args.use_balancing_loss)
        self.assertEqual(args.prototype_balance_loss_weight, 0.1)
        self.assertTrue(args.normalize_for_token_scoring)
        self.assertEqual(args.special_token_ids['bos_token_id'], 49406)
        self.assertTrue(args.error_on_empty_kept_tokens)
        self.assertTrue(args.amp)
        self.assertEqual(args.amp_dtype, 'bf16')
        self.assertEqual(args.proxy_temperature, 0.2)
        self.assertEqual(args.lambda_proxy, 1.0)
        self.assertFalse(args.use_loss_align)
        self.assertEqual(args.lambda_align, 0.5)
        self.assertFalse(args.use_loss_diag)
        self.assertEqual(args.lambda_diag, 0.25)
        self.assertTrue(args.use_loss_support)
        self.assertEqual(args.lambda_support, 0.15)
        self.assertEqual(args.support_min, 2.5)
        self.assertEqual(args.lr_class_proxies, 0.003)
        self.assertEqual(args.weight_decay_class_proxies, 0.02)
        self.assertEqual(args.retrieval_scorer, 'approximate')

    def test_new_align_and_diag_disable_flags_parse_from_cli(self):
        args = get_args(['--use_loss_align', 'false', '--use_loss_diag', 'false'])
        self.assertFalse(args.use_loss_align)
        self.assertFalse(args.use_loss_diag)

    def test_support_loss_flags_parse_from_cli(self):
        args = get_args(['--use_loss_support', 'true', '--lambda_support', '0.2', '--support_min', '3.0'])
        self.assertTrue(args.use_loss_support)
        self.assertEqual(args.lambda_support, 0.2)
        self.assertEqual(args.support_min, 3.0)

    def test_support_loss_disabled_config_remains_backward_compatible(self):
        path = self._write_config(
            {
                'loss': {
                    'use_loss_support': False,
                    'lambda_support': 0.0,
                    'support_min': 2.0,
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertFalse(args.use_loss_support)
        self.assertEqual(args.lambda_support, 0.0)
        self.assertEqual(args.support_min, 2.0)

    def test_legacy_split_loss_config_still_loads_for_backward_compat(self):
        path = self._write_config(
            {
                'prototype': {
                    'use_balancing_loss': True,
                    'balance_loss_weight': 0.1,
                    'use_diversity_loss': False,
                    'diversity_loss_weight': 0.0,
                },
                'training': {
                    'lambda_proxy': 1.25,
                    'use_loss_proxy_image': True,
                    'use_loss_proxy_text': False,
                    'use_loss_proxy_text_exact': True,
                    'lambda_align': 0.2,
                    'lambda_diag': 0.3,
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertTrue(args.use_balancing_loss)
        self.assertEqual(args.prototype_balance_loss_weight, 0.1)
        self.assertFalse(args.use_diversity_loss)
        self.assertEqual(args.diversity_loss_weight, 0.0)
        self.assertEqual(args.lambda_proxy, 1.25)
        self.assertFalse(args.use_loss_proxy_text)
        self.assertEqual(args.lambda_align, 0.2)
        self.assertEqual(args.lambda_diag, 0.3)

    def test_authoritative_contextualization_flag_overrides_legacy_alias(self):
        path = self._write_config(
            {
                'model': {
                    'use_prototype_contextualization': True,
                },
                'prototype': {
                    'contextualization_enabled': False,
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertFalse(args.prototype_contextualization_enabled)

    def test_legacy_contextualization_alias_applies_only_when_authoritative_flag_is_missing(self):
        path = self._write_config(
            {
                'model': {
                    'use_prototype_contextualization': False,
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertFalse(args.prototype_contextualization_enabled)

    def test_cli_authoritative_contextualization_flag_cannot_be_reenabled_by_legacy_alias(self):
        path = self._write_config(
            {
                'model': {
                    'use_prototype_contextualization': True,
                },
            }
        )
        args = get_args(['--config_file', path, '--prototype_contextualization_enabled', 'false'])
        self.assertFalse(args.prototype_contextualization_enabled)



if __name__ == '__main__':  # pragma: no cover
    unittest.main()

