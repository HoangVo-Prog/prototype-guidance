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

    def test_valid_runtime_surface_loads_special_token_ids_precision_and_amp_knobs(self):
        path = self._write_config(
            {
                'model': {
                    'projector_type': 'linear',
                    'normalize_projector_outputs': True,
                    'learn_logit_scale': False,
                    'backbone_precision': 'fp32',
                    'prototype_precision': 'fp32',
                },
                'prototype': {
                    'normalize_for_self_interaction': True,
                    'normalize_for_routing': True,
                    'use_balancing_loss': True,
                    'balance_loss_weight': 0.1,
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
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertEqual(args.projector_type, 'linear')
        self.assertTrue(args.normalize_projector_outputs)
        self.assertFalse(args.learn_logit_scale)
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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
