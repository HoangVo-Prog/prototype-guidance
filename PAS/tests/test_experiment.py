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

    def test_wandb_uploads_run_config_files_and_only_defines_train_debug_val_axes(self):
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
        fake_wandb.plot = mock.Mock()

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
        self.assertIn('debug/*', defined_metric_names)
        self.assertIn('val/epoch', defined_metric_names)
        self.assertIn('val/*', defined_metric_names)
        self.assertNotIn('heldout_val/epoch', defined_metric_names)
        self.assertNotIn('heldout_val/*', defined_metric_names)
        self.assertNotIn('curves/epoch', defined_metric_names)
        self.assertNotIn('curves/*', defined_metric_names)
        fake_wandb.Artifact.assert_called_once()
        self.assertEqual(fake_artifact.add_file.call_count, 2)
        fake_run.log_artifact.assert_called_once_with(fake_artifact)

    def test_log_comparison_charts_emits_combined_train_and_val_plots(self):
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
        fake_wandb.plot = mock.Mock()
        fake_wandb.plot.line_series.side_effect = lambda **kwargs: kwargs

        tmpdir = os.path.join(REPO_ROOT, 'tests_tmp_experiment_tracker')
        os.makedirs(tmpdir, exist_ok=True)
        try:
            with mock.patch('utils.experiment.wandb', fake_wandb):
                tracker = ExperimentTracker(args, tmpdir, distributed_rank=0)
                tracker.log_comparison_charts(
                    3,
                    train_metrics={'loss_total': 1.2, 'loss_diag': 0.4},
                    val_metrics={'loss_total': 0.9, 'R1': 44.0},
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        logged_payload = fake_wandb.log.call_args_list[-1].args[0]
        self.assertIn('plots/loss_total', logged_payload)
        self.assertIn('plots/loss_diag', logged_payload)
        self.assertIn('plots/R1', logged_payload)
        loss_total_chart = logged_payload['plots/loss_total']
        self.assertEqual(loss_total_chart['keys'], ['train', 'val'])
        self.assertEqual(loss_total_chart['xs'], [3.0])
        self.assertEqual(loss_total_chart['ys'], [[1.2], [0.9]])
        r1_chart = logged_payload['plots/R1']
        self.assertEqual(r1_chart['keys'], ['val'])
        self.assertEqual(r1_chart['ys'], [[44.0]])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
