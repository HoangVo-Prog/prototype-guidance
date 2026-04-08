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

    def test_wandb_uploads_run_config_files_and_only_defines_train_val_axes(self):
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
            config_file=None,
        )

        fake_run = mock.Mock()
        fake_wandb = mock.Mock()
        fake_wandb.init.return_value = fake_run
        fake_artifact = mock.Mock()
        fake_wandb.Artifact.return_value = fake_artifact

        tmpdir = os.path.join(REPO_ROOT, 'tests_tmp_experiment_tracker')
        os.makedirs(tmpdir, exist_ok=True)
        try:
            for filename in ('configs.yaml', 'resolved_config.yaml'):
                with open(os.path.join(tmpdir, filename), 'w', encoding='utf-8') as handle:
                    handle.write('test: true\n')
            with mock.patch('utils.experiment.wandb', fake_wandb):
                tracker = ExperimentTracker(args, tmpdir, distributed_rank=0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertTrue(tracker.enabled)
        defined_metric_names = [call.args[0] for call in fake_wandb.define_metric.call_args_list]
        self.assertIn('train/step', defined_metric_names)
        self.assertIn('train/*', defined_metric_names)
        self.assertNotIn('debug/*', defined_metric_names)
        self.assertIn('val/epoch', defined_metric_names)
        self.assertIn('val/*', defined_metric_names)
        self.assertNotIn('heldout_val/epoch', defined_metric_names)
        self.assertNotIn('heldout_val/*', defined_metric_names)
        self.assertNotIn('curves/epoch', defined_metric_names)
        self.assertNotIn('curves/*', defined_metric_names)
        fake_wandb.Artifact.assert_called_once()
        self.assertEqual(fake_artifact.add_file.call_count, 2)
        fake_run.log_artifact.assert_called_once_with(fake_artifact)

    def test_log_filters_out_debug_metric_namespaces(self):
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
            config_file=None,
        )

        fake_run = mock.Mock()
        fake_wandb = mock.Mock()
        fake_wandb.init.return_value = fake_run

        tmpdir = os.path.join(REPO_ROOT, 'tests_tmp_experiment_tracker')
        os.makedirs(tmpdir, exist_ok=True)
        try:
            with mock.patch('utils.experiment.wandb', fake_wandb):
                tracker = ExperimentTracker(args, tmpdir, distributed_rank=0)
                tracker.log({
                    'train/loss_total': 1.0,
                    'debug/prototype_usage_entropy': 0.5,
                    'debugs/legacy_metric': 0.1,
                }, step=10)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        _, kwargs = fake_wandb.log.call_args
        self.assertEqual(kwargs['step'], 10)
        logged_metrics = fake_wandb.log.call_args.args[0]
        self.assertIn('train/loss_total', logged_metrics)
        self.assertNotIn('debug/prototype_usage_entropy', logged_metrics)
        self.assertNotIn('debugs/legacy_metric', logged_metrics)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
