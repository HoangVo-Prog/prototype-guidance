import os
import sys
import shutil
import types
import unittest
from unittest import mock

config_stub = types.ModuleType('utils.config')
config_stub.build_runtime_config = lambda args: {'logging': {'use_wandb': getattr(args, 'use_wandb', False)}}
sys.modules['utils.config'] = config_stub

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.experiment import ExperimentTracker


class ExperimentTrackerTests(unittest.TestCase):
    def test_wandb_init_disables_tensorboard_sync_to_avoid_duplicate_train_namespaces(self):
        args = types.SimpleNamespace(
            use_wandb=True,
            wandb_project='PAS',
            wandb_entity=None,
            wandb_run_name='unit-test-run',
            wandb_group=None,
            wandb_mode='offline',
            wandb_tags=None,
            wandb_notes=None,
            wandb_log_code=False,
        )

        fake_wandb = mock.Mock()
        fake_wandb.init.return_value = object()

        tmpdir = os.path.join(REPO_ROOT, 'tests_tmp_experiment_tracker')
        os.makedirs(tmpdir, exist_ok=True)
        try:
            with mock.patch('utils.experiment.wandb', fake_wandb):
                tracker = ExperimentTracker(args, tmpdir, distributed_rank=0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertTrue(tracker.enabled)
        _, kwargs = fake_wandb.init.call_args
        self.assertIn('sync_tensorboard', kwargs)
        self.assertFalse(kwargs['sync_tensorboard'])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
