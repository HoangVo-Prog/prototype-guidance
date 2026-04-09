import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from utils.options import get_args
    from model.build import build_model
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    get_args = None
    build_model = None
    IMPORT_ERROR = exc


@unittest.skipUnless(get_args is not None and build_model is not None, f'Prototype wiring tests require runtime imports: {IMPORT_ERROR}')
class PrototypeWiringSurfaceTests(unittest.TestCase):
    def _write_config(self, payload):
        handle = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
        with handle:
            handle.write(json.dumps(payload))
        self.addCleanup(lambda: os.path.exists(handle.name) and os.remove(handle.name))
        return handle.name

    def test_legacy_training_mode_and_stage_keys_are_accepted(self):
        path = self._write_config(
            {
                'model': {
                    'training_mode': 'pas',
                    'use_prototype_branch': True,
                    'use_prototype_bank': True,
                    'use_image_conditioned_pooling': True,
                },
                'training': {
                    'stage': 'stage1',
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertEqual(args.training_mode, 'pas')
        self.assertEqual(args.training_stage, 'stage1')

    def test_branch_is_inferred_from_bank_and_pooling_flags(self):
        path = self._write_config(
            {
                'model': {
                    'use_prototype_bank': True,
                    'use_image_conditioned_pooling': True,
                },
            }
        )
        args = get_args(['--config_file', path])
        self.assertTrue(args.use_prototype_branch)
        self.assertTrue(args.use_prototype_bank)
        self.assertTrue(args.use_image_conditioned_pooling)

    def test_build_routes_to_pas_model_when_prototype_flags_are_on(self):
        args = SimpleNamespace(
            host_type='clip',
            use_prototype_branch=None,
            use_prototype_bank=True,
            use_image_conditioned_pooling=True,
        )
        sentinel = object()
        with mock.patch('model.build._pas_model.build_model', return_value=sentinel):
            model = build_model(args, num_classes=2)
        self.assertIs(model, sentinel)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

