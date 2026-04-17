import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment-dependent
    torch = None
    nn = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if torch is not None:
    from utils.metrics import Evaluator
    from utils.modular_checkpoint import MetricAuthorityPolicy, ModularCheckpointManager


@unittest.skipUnless(torch is not None, 'Torch is required for metric/checkpoint authority tests.')
class MetricCheckpointAuthorityTests(unittest.TestCase):
    if nn is not None:
        class _TinyHostModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_model = nn.Module()
                self.base_model.visual = nn.Linear(1, 1)
                self.host_head = nn.Linear(1, 1)

    def _manager(self, checkpointing_config):
        args = SimpleNamespace(
            config_data={'checkpointing': checkpointing_config},
            host_type='clip',
            run_name='unit-test',
            training_stage='joint',
        )
        logger = __import__('logging').getLogger('test.authority')
        logger.setLevel(__import__('logging').INFO)
        return ModularCheckpointManager(args=args, save_dir=checkpointing_config['save']['dir'], logger=logger)

    def _checkpointing_config(self, save_dir):
        return {
            'groups': {
                'host': {'enabled': True},
                'prototype_bank': {'enabled': False},
                'prototype_projector': {'enabled': False},
                'fusion': {'enabled': False},
            },
            'save': {
                'dir': save_dir,
                'save_latest': True,
                'save_best': True,
                'keep_last_n': 1,
                'artifacts': {
                    'host': {
                        'enabled': True,
                        'filename_latest': 'host_latest.pth',
                        'filename_best': 'host_best.pth',
                    },
                    'prototype_bank': {'enabled': False, 'filename_latest': 'x', 'filename_best': 'x'},
                    'prototype_projector': {'enabled': False, 'filename_latest': 'x', 'filename_best': 'x'},
                    'fusion': {'enabled': False, 'filename_latest': 'x', 'filename_best': 'x'},
                },
            },
            'load': {
                'enabled': False,
                'strict': True,
                'sources': {
                    'host': {'enabled': False, 'path': None},
                    'prototype_bank': {'enabled': False, 'path': None},
                    'prototype_projector': {'enabled': False, 'path': None},
                    'fusion': {'enabled': False, 'path': None},
                },
            },
            'authority_validation': {
                'enabled': True,
                'strict': True,
                'warn_only': False,
                'allow_fallback_row_name_classification': True,
            },
        }

    def test_default_case_display_equals_source(self):
        metrics_rows = [
            {'task': 'host-t2i', 'R1': 55.0, 'R5': 80.0, 'R10': 90.0, 'mAP': 50.0, 'mINP': 40.0, 'rSum': 225.0},
            {'task': 'pas-t2i', 'R1': 60.0, 'R5': 82.0, 'R10': 91.0, 'mAP': 54.0, 'mINP': 42.0, 'rSum': 233.0},
        ]
        row_metadata = {
            'host-t2i': {'authority_role': 'host'},
            'pas-t2i': {'authority_role': 'fused'},
        }
        authority = Evaluator.build_authority_context(
            metrics_rows=metrics_rows,
            row_metadata=row_metadata,
            selected_display_row='pas-t2i',
            selected_source_row='pas-t2i',
        )
        self.assertEqual(authority['display_row'], 'pas-t2i')
        self.assertEqual(authority['source_row'], 'pas-t2i')
        self.assertFalse(authority['mismatch'])

    def test_subset_selection_case_preserves_display_source_divergence(self):
        metrics_rows = [
            {'task': 'pas-t2i', 'R1': 61.0, 'R5': 82.0, 'R10': 91.0, 'mAP': 54.0, 'mINP': 42.0, 'rSum': 234.0},
            {'task': 'proto_020-t2i', 'R1': 64.0, 'R5': 84.0, 'R10': 92.0, 'mAP': 56.0, 'mINP': 43.0, 'rSum': 240.0},
        ]
        row_metadata = {
            'pas-t2i': {'authority_role': 'fused'},
            'proto_020-t2i': {'authority_role': 'fused'},
        }
        authority = Evaluator.build_authority_context(
            metrics_rows=metrics_rows,
            row_metadata=row_metadata,
            selected_display_row='pas-t2i',
            selected_source_row='proto_020-t2i',
        )
        self.assertTrue(authority['mismatch'])
        self.assertEqual(authority['display_row'], 'pas-t2i')
        self.assertEqual(authority['source_row'], 'proto_020-t2i')
        self.assertEqual(authority['candidates']['fused'], 'proto_020-t2i')

    def test_checkpoint_save_for_host_authority_with_valid_host_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._manager(self._checkpointing_config(tmpdir))
            model = self._TinyHostModel()
            authority_context = {
                'display_row': 'host-t2i',
                'source_row': 'host-t2i',
                'row_roles': {'host-t2i': 'host'},
                'row_metrics': {'host-t2i': {'R1': 77.7}},
                'candidates': {'host': 'host-t2i', 'prototype': None, 'fused': None},
            }
            manager.save_latest(
                model=model,
                epoch=1,
                global_step=10,
                metric_value=0.0,
                metric_row='host-t2i',
                metric_display_row='host-t2i',
                metric_source_row='host-t2i',
                authority_context=authority_context,
            )
            latest_path = os.path.join(tmpdir, 'host_latest.pth')
            self.assertTrue(os.path.exists(latest_path))
            payload = torch.load(latest_path, map_location='cpu')
            self.assertEqual(payload['metric']['row'], 'host-t2i')
            self.assertEqual(payload['metric']['display_row'], 'host-t2i')
            self.assertEqual(payload['metric']['source_row'], 'host-t2i')
            self.assertTrue(payload['metric']['authority_valid'])

    def test_checkpoint_authority_prefers_source_not_display_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._manager(self._checkpointing_config(tmpdir))
            model = self._TinyHostModel()
            authority_context = {
                'display_row': 'pas-t2i',
                'source_row': 'host-t2i',
                'row_roles': {'pas-t2i': 'fused', 'host-t2i': 'host'},
                'row_metrics': {
                    'pas-t2i': {'R1': 99.0},
                    'host-t2i': {'R1': 66.0},
                },
                'candidates': {'host': 'host-t2i', 'prototype': None, 'fused': 'pas-t2i'},
            }
            manager.save_latest(
                model=model,
                epoch=1,
                global_step=10,
                metric_value=99.0,
                metric_row='pas-t2i',
                metric_display_row='pas-t2i',
                metric_source_row='host-t2i',
                authority_context=authority_context,
            )
            latest_path = os.path.join(tmpdir, 'host_latest.pth')
            payload = torch.load(latest_path, map_location='cpu')
            self.assertEqual(payload['metric']['row'], 'host-t2i')
            self.assertEqual(payload['metric']['display_row'], 'pas-t2i')
            self.assertEqual(payload['metric']['source_row'], 'host-t2i')

    def test_checkpoint_save_rejection_when_row_is_disallowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._manager(self._checkpointing_config(tmpdir))
            model = self._TinyHostModel()
            authority_context = {
                'display_row': 'prototype-t2i',
                'source_row': 'prototype-t2i',
                'row_roles': {'prototype-t2i': 'prototype'},
                'row_metrics': {'prototype-t2i': {'R1': 88.8}},
                'candidates': {'host': None, 'prototype': 'prototype-t2i', 'fused': None},
            }
            with self.assertRaisesRegex(ValueError, 'Authority validation failed'):
                manager.save_latest(
                    model=model,
                    epoch=1,
                    global_step=10,
                    metric_value=88.8,
                    metric_row='prototype-t2i',
                    metric_display_row='prototype-t2i',
                    metric_source_row='prototype-t2i',
                    authority_context=authority_context,
                )
            self.assertFalse(os.path.exists(os.path.join(tmpdir, 'host_latest.pth')))

    def test_validation_utility_is_deterministic(self):
        policy = MetricAuthorityPolicy(enabled=True, strict=True, warn_only=False, allow_fallback_row_name_classification=True)
        row_roles = {'host-t2i': 'host'}
        lhs = policy.validate_row_for_group(group_name='host', row_name='host-t2i', row_roles=row_roles)
        rhs = policy.validate_row_for_group(group_name='host', row_name='host-t2i', row_roles=row_roles)
        self.assertEqual(lhs, rhs)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
