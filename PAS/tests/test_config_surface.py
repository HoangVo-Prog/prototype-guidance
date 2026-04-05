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

    def test_stage_must_be_known_enum(self):
        path = self._write_config({'training': {'stage': 'warmup'}})
        with self.assertRaisesRegex(ValueError, 'training.stage'):
            load_yaml_config(None, path)

    def test_stage2_requires_finetune_checkpoint(self):
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
        with self.assertRaisesRegex(ValueError, 'training.stage=stage2 requires training.finetune'):
            get_args(['--config_file', path])

    def test_stage1_requires_matching_freeze_policy(self):
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
        with self.assertRaisesRegex(ValueError, 'training.stage=stage1 requires freeze_image_backbone=True'):
            get_args(['--config_file', path])

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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
