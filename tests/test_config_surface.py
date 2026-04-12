import json
import os
import sys
import tempfile
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from utils.config import load_yaml_config
    from utils.options import get_args
    CONFIG_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    load_yaml_config = None
    get_args = None
    CONFIG_IMPORT_ERROR = exc


@unittest.skipUnless(load_yaml_config is not None and get_args is not None, f'Config tests require yaml/config runtime imports: {CONFIG_IMPORT_ERROR}')
class ConfigSurfaceTests(unittest.TestCase):
    def _write_config(self, payload):
        handle = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
        with handle:
            handle.write(json.dumps(payload))
        self.addCleanup(lambda: os.path.exists(handle.name) and os.remove(handle.name))
        return handle.name

    def test_removed_exact_retrieval_config_is_rejected(self):
        path = self._write_config({'loss': {'use_loss_ret_exact_text': True}})
        with self.assertRaisesRegex(ValueError, 'use_loss_ret_exact_text'):
            load_yaml_config(None, path)

    def test_removed_exact_retrieval_cli_flag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'row-wise surrogate image-to-text retrieval'):
            get_args(['--use_loss_ret_exact'])

    def test_invalid_retrieval_scorer_fails_at_config_load_time(self):
        path = self._write_config({'evaluation': {'retrieval_scorer': 'surrogate_default'}})
        with self.assertRaisesRegex(ValueError, 'evaluation.retrieval_scorer'):
            load_yaml_config(None, path)

    def test_fusion_explicit_lambdas_require_unit_sum(self):
        path = self._write_config(
            {
                'fusion': {
                    'lambda_host': 0.8,
                    'lambda_prototype': 0.3,
                },
            }
        )
        with self.assertRaisesRegex(ValueError, 'fusion.lambda_host'):
            load_yaml_config(None, path)

    def test_fusion_eval_subsets_validate_unit_sum(self):
        path = self._write_config(
            {
                'fusion': {
                    'lambda_host': 1.0,
                    'lambda_prototype': 0.0,
                    'eval_subsets': [
                        {'name': 'host_only', 'lambda_host': 1.0, 'lambda_prototype': 0.0},
                        {'name': 'bad_pair', 'lambda_host': 0.8, 'lambda_prototype': 0.3},
                    ],
                },
            }
        )
        with self.assertRaisesRegex(ValueError, 'fusion.eval_subsets'):
            load_yaml_config(None, path)

    def test_fusion_legacy_coefficient_alias_still_loads(self):
        path = self._write_config(
            {
                'fusion': {
                    'coefficient': 0.6,
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertAlmostEqual(args.fusion_lambda_host, 1.0)
        self.assertAlmostEqual(args.fusion_lambda_prototype, 0.6)

    def test_stage_must_be_known_enum(self):
        path = self._write_config({'training': {'stage': 'warmup'}})
        with self.assertRaisesRegex(ValueError, 'training.stage'):
            load_yaml_config(None, path)

    def test_training_log_debug_metrics_alias_is_accepted(self):
        path = self._write_config({'training': {'log_debug_metrics': True}})
        args = get_args(['--config_file', path])
        self.assertTrue(args.log_debug_metrics)

    def test_diag_temperature_alias_is_accepted(self):
        path = self._write_config({'objectives': {'objectives': {'diag_temperature': 0.15}}})
        args = get_args(['--config_file', path])
        self.assertAlmostEqual(args.diag_temperature, 0.15)

    def test_stage2_allows_missing_finetune_checkpoint(self):
        path = self._write_config(
            {
                'loss': {
                    'use_loss_ret': True,
                    'lambda_ret': 1.0,
                    'use_loss_diag': True,
                    'lambda_diag': 1.0,
                },
                'training': {
                    'stage': 'stage2',
                    'freeze_image_backbone': False,
                    'freeze_text_backbone': False,
                    'freeze_prototype_side': True,
                    'finetune': '',
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertEqual(args.training_stage, 'stage2')
        self.assertEqual(args.finetune, '')

    def test_stage1_allows_arbitrary_freeze_policy(self):
        path = self._write_config(
            {
                'loss': {
                    'use_loss_ret': True,
                    'lambda_ret': 1.0,
                    'use_loss_diag': True,
                    'lambda_diag': 1.0,
                },
                'training': {
                    'stage': 'stage1',
                    'freeze_image_backbone': False,
                    'freeze_text_backbone': True,
                    'freeze_prototype_side': False,
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertEqual(args.training_stage, 'stage1')
        self.assertFalse(args.freeze_image_backbone)
        self.assertTrue(args.freeze_text_backbone)
        self.assertFalse(args.freeze_prototype_side)

    def test_no_prototype_bank_allows_approximate_config_surface(self):
        path = self._write_config(
            {
                'model': {
                    'use_prototype_bank': False,
                    'use_image_conditioned_pooling': False,
                },
                'evaluation': {
                    'retrieval_scorer': 'approximate',
                },
            }
        )
        config = load_yaml_config(None, path)
        self.assertFalse(config['model']['use_prototype_bank'])
        self.assertFalse(config['model']['use_image_conditioned_pooling'])
        self.assertEqual(config['evaluation']['retrieval_scorer'], 'approximate')

    def test_vanilla_clip_requires_eos_only_and_bidirectional_loss(self):
        path = self._write_config(
            {
                'model': {
                    'training_mode': 'vanilla_clip',
                    'use_prototype_bank': False,
                    'use_image_conditioned_pooling': False,
                    'use_custom_projector': False,
                },
                'text_pooling': {
                    'token_policy': 'content_only',
                },
                'loss': {
                    'use_loss_ret': True,
                    'retrieval_mode': 'surrogate_i2t',
                    'use_loss_proxy_image': False,
                    'use_loss_proxy_text': False,
                    'use_loss_proxy_text_exact': False,
                    'use_loss_align': False,
                    'use_loss_diag': False,
                    'use_loss_support': False,
                    'use_balancing_loss': False,
                    'use_diversity_loss': False,
                },
                'evaluation': {
                    'retrieval_scorer': 'exact',
                },
            }
        )
        with self.assertRaisesRegex(ValueError, 'vanilla_clip requires text_pooling.token_policy=eos_only'):
            load_yaml_config(None, path)

    def test_clip_bidirectional_requires_vanilla_mode(self):
        path = self._write_config(
            {
                'model': {
                    'training_mode': 'pas',
                },
                'loss': {
                    'retrieval_mode': 'clip_bidirectional',
                },
            }
        )
        with self.assertRaisesRegex(ValueError, 'clip_bidirectional is only supported'):
            load_yaml_config(None, path)

    def test_vanilla_clip_surface_loads(self):
        path = self._write_config(
            {
                'model': {
                    'training_mode': 'vanilla_clip',
                    'use_prototype_bank': False,
                    'use_image_conditioned_pooling': False,
                    'use_custom_projector': False,
                },
                'text_pooling': {
                    'token_policy': 'eos_only',
                },
                'loss': {
                    'use_loss_ret': True,
                    'retrieval_mode': 'clip_bidirectional',
                    'lambda_ret': 1.0,
                    'use_loss_proxy_image': False,
                    'use_loss_proxy_text': False,
                    'use_loss_proxy_text_exact': False,
                    'use_loss_align': False,
                    'use_loss_diag': False,
                    'use_loss_support': False,
                    'use_balancing_loss': False,
                    'use_diversity_loss': False,
                },
                'evaluation': {
                    'retrieval_scorer': 'exact',
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertEqual(args.training_mode, 'vanilla_clip')
        self.assertFalse(args.use_prototype_bank)
        self.assertFalse(args.use_image_conditioned_pooling)
        self.assertFalse(args.use_custom_projector)
        self.assertEqual(args.token_policy, 'eos_only')
        self.assertEqual(args.retrieval_mode, 'clip_bidirectional')


    def test_itself_host_only_surface_loads(self):
        path = self._write_config(
            {
                'model': {
                    'training_mode': 'pas',
                    'use_prototype_branch': False,
                    'use_prototype_bank': False,
                    'use_image_conditioned_pooling': False,
                },
                'host': {
                    'type': 'itself',
                    'itself_loss_names': 'tal+cid',
                    'itself_return_all': True,
                },
                'loss': {
                    'retrieval_mode': 'surrogate_i2t',
                },
                'evaluation': {
                    'retrieval_scorer': 'exact',
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertEqual(args.host_type, 'itself')
        self.assertFalse(args.use_prototype_branch)
        self.assertFalse(args.use_prototype_bank)
        self.assertFalse(args.use_image_conditioned_pooling)
        self.assertTrue(args.itself_return_all)

    def test_itself_host_plus_prototype_surface_loads(self):
        path = self._write_config(
            {
                'model': {
                    'training_mode': 'pas',
                    'use_prototype_branch': True,
                    'use_prototype_bank': True,
                    'use_image_conditioned_pooling': True,
                },
                'host': {
                    'type': 'itself',
                    'itself_loss_names': 'tal+cid',
                    'itself_return_all': True,
                },
                'loss': {
                    'retrieval_mode': 'surrogate_i2t',
                },
                'evaluation': {
                    'retrieval_scorer': 'exact',
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertEqual(args.host_type, 'itself')
        self.assertTrue(args.use_prototype_branch)
        self.assertTrue(args.use_prototype_bank)
        self.assertTrue(args.use_image_conditioned_pooling)

    def test_vanilla_clip_rejects_itself_host(self):
        path = self._write_config(
            {
                'model': {
                    'training_mode': 'vanilla_clip',
                    'use_prototype_bank': False,
                    'use_image_conditioned_pooling': False,
                },
                'host': {
                    'type': 'itself',
                },
                'text_pooling': {
                    'token_policy': 'eos_only',
                },
                'loss': {
                    'use_loss_ret': True,
                    'retrieval_mode': 'clip_bidirectional',
                },
                'evaluation': {
                    'retrieval_scorer': 'exact',
                },
            }
        )
        with self.assertRaisesRegex(ValueError, 'requires host.type=clip'):
            load_yaml_config(None, path)

    def test_prototype_bank_requires_image_conditioned_pooling(self):
        path = self._write_config(
            {
                'model': {
                    'use_prototype_bank': True,
                    'use_image_conditioned_pooling': False,
                },
            }
        )
        with self.assertRaisesRegex(ValueError, 'use_prototype_bank=true'):
            load_yaml_config(None, path)

    def test_stage1_allows_loss_ablation_flags(self):
        path = self._write_config(
            {
                'loss': {
                    'use_loss_ret': False,
                    'lambda_ret': 1.0,
                    'use_loss_diag': False,
                    'lambda_diag': 1.0,
                    'use_loss_align': False,
                    'lambda_align': 0.0,
                },
                'training': {
                    'stage': 'stage1',
                    'freeze_image_backbone': True,
                    'freeze_text_backbone': True,
                    'freeze_prototype_side': False,
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertFalse(args.use_loss_ret)
        self.assertFalse(args.use_loss_diag)
    def test_valid_stage1_surface_loads(self):
        path = self._write_config(
            {
                'loss': {
                    'use_loss_ret': True,
                    'lambda_ret': 1.25,
                    'use_loss_diag': True,
                    'lambda_diag': 0.75,
                    'use_loss_align': False,
                    'lambda_align': 0.0,
                },
                'training': {
                    'stage': 'stage1',
                    'freeze_image_backbone': True,
                    'freeze_text_backbone': True,
                    'freeze_prototype_side': False,
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertEqual(args.training_stage, 'stage1')
        self.assertTrue(args.use_loss_ret)
        self.assertEqual(args.lambda_ret, 1.25)
        self.assertTrue(args.use_loss_diag)
        self.assertEqual(args.lambda_diag, 0.75)
        self.assertTrue(args.freeze_image_backbone)
        self.assertTrue(args.freeze_text_backbone)
        self.assertFalse(args.freeze_prototype_side)

    def test_valid_stage2_surface_loads(self):
        path = self._write_config(
            {
                'loss': {
                    'use_loss_ret': True,
                    'lambda_ret': 1.0,
                    'use_loss_diag': True,
                    'lambda_diag': 1.0,
                },
                'training': {
                    'stage': 'stage2',
                    'freeze_image_backbone': False,
                    'freeze_text_backbone': False,
                    'freeze_prototype_side': True,
                    'finetune': 'runs/stage1/best.pth',
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertEqual(args.training_stage, 'stage2')
        self.assertFalse(args.freeze_image_backbone)
        self.assertFalse(args.freeze_text_backbone)
        self.assertTrue(args.freeze_prototype_side)
        self.assertEqual(args.finetune, 'runs/stage1/best.pth')

    def test_freeze_schedule_surface_loads(self):
        path = self._write_config(
            {
                'model': {
                    'use_prototype_branch': True,
                    'use_prototype_bank': True,
                    'use_image_conditioned_pooling': True,
                },
                'training': {
                    'epochs': 8,
                    'freeze_schedule': [
                        {
                            'name': 'warmup',
                            'epoch_start': 1,
                            'epoch_end': 2,
                            'trainable_groups': ['prototype_bank', 'prototype_projector', 'routing', 'fusion'],
                            'frozen_groups': ['host_backbone', 'host_retrieval'],
                            'lr_overrides': {'prototype_bank': 1e-4},
                            'loss_weights': {'lambda_host': 0.0, 'lambda_ret': 1.0},
                        },
                        {
                            'name': 'joint',
                            'epoch_start': 3,
                            'epoch_end': 8,
                            'trainable_groups': ['host_backbone', 'host_retrieval', 'prototype_projector', 'fusion'],
                            'frozen_groups': ['prototype_bank'],
                            'lr_overrides': {'host_backbone': 1e-5},
                            'loss_weights': {'lambda_host': 1.0},
                        },
                    ],
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertIsInstance(args.freeze_schedule, list)
        self.assertEqual(args.freeze_schedule[0]['name'], 'warmup')
        self.assertEqual(args.freeze_schedule[1]['name'], 'joint')

    def test_freeze_schedule_rejects_unknown_group(self):
        path = self._write_config(
            {
                'training': {
                    'epochs': 3,
                    'freeze_schedule': [
                        {
                            'name': 'bad',
                            'epoch_start': 1,
                            'epoch_end': 3,
                            'trainable_groups': ['prototype_bank', 'unknown_group'],
                        }
                    ],
                },
            }
        )
        with self.assertRaisesRegex(ValueError, 'Unsupported module group'):
            load_yaml_config(None, path)

    def test_freeze_schedule_rejects_overlaps(self):
        path = self._write_config(
            {
                'training': {
                    'epochs': 5,
                    'freeze_schedule': [
                        {'name': 'p1', 'epoch_start': 1, 'epoch_end': 3},
                        {'name': 'p2', 'epoch_start': 3, 'epoch_end': 5},
                    ],
                },
            }
        )
        with self.assertRaisesRegex(ValueError, 'overlapping phases'):
            load_yaml_config(None, path)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()


