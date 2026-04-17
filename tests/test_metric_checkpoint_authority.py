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
    from model.interface_contract import HOST_EXPORT_INTERFACE_VERSION, HOST_SCORE_SCHEMA_VERSION, PROTOTYPE_SCORE_SCHEMA_VERSION
    from model.runtime_modes import (
        RUNTIME_MODE_CALIBRATION_ONLY,
        RUNTIME_MODE_FUSED_EXTERNAL,
        RUNTIME_MODE_HOST_ONLY,
        RUNTIME_MODE_JOINT_TRAINING,
        RUNTIME_MODE_PROTOTYPE_ONLY,
    )
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

        class _TinyComponentModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.host_core = nn.Module()
                self.host_core.base_model = nn.Module()
                self.host_core.base_model.visual = nn.Linear(1, 1)
                self.host_core.host_head = nn.Linear(1, 1)

                self.prototype_plugin = nn.Module()
                self.prototype_plugin.prototype_head = nn.Module()
                self.prototype_plugin.prototype_head.prototype_bank = nn.Parameter(torch.randn(2, 2))
                self.prototype_plugin.prototype_head.image_projector = nn.Linear(2, 2)
                self.prototype_plugin.prototype_head.text_projector = nn.Linear(2, 2)

                self.composer = nn.Module()
                self.composer.fusion_module = nn.Linear(1, 1)

        class _TinyCompatibleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.runtime_mode = RUNTIME_MODE_FUSED_EXTERNAL
                self.host_export_interface_version = HOST_EXPORT_INTERFACE_VERSION
                self.host_component_schema_version = 'host_core_v1'

                self.host_core = nn.Module()
                self.host_core.base_model = nn.Module()
                self.host_core.base_model.visual = nn.Linear(1, 1)
                self.host_core.host_head = nn.Linear(1, 1)

                self.prototype_plugin = nn.Module()
                self.prototype_plugin.schema_version = 'prototype_plugin_v1'
                self.prototype_plugin.accepted_host_interface_versions = {HOST_EXPORT_INTERFACE_VERSION}
                self.prototype_plugin.prototype_head = nn.Module()
                self.prototype_plugin.prototype_head.prototype_bank = nn.Parameter(torch.randn(2, 2))
                self.prototype_plugin.prototype_head.image_projector = nn.Linear(2, 2)
                self.prototype_plugin.prototype_head.text_projector = nn.Linear(2, 2)

                self.composer = nn.Module()
                self.composer.schema_version = 'composer_v1'
                self.composer.expected_host_score_schema_version = HOST_SCORE_SCHEMA_VERSION
                self.composer.expected_prototype_score_schema_version = PROTOTYPE_SCORE_SCHEMA_VERSION
                self.composer.fusion_module = nn.Linear(1, 1)

            def get_group_checkpoint_compatibility(self, group_name: str):
                all_modes = [
                    RUNTIME_MODE_HOST_ONLY,
                    RUNTIME_MODE_PROTOTYPE_ONLY,
                    RUNTIME_MODE_FUSED_EXTERNAL,
                    RUNTIME_MODE_JOINT_TRAINING,
                    RUNTIME_MODE_CALIBRATION_ONLY,
                ]
                common = {
                    'runtime_mode': self.runtime_mode,
                    'host_export_interface_version': self.host_export_interface_version,
                }
                if group_name == 'host':
                    return {
                        **common,
                        'component_name': 'HostCore',
                        'component_schema_version': self.host_component_schema_version,
                        'compatible_runtime_modes': list(all_modes),
                    }
                if group_name in {'prototype_bank', 'prototype_projector'}:
                    return {
                        **common,
                        'component_name': 'PrototypePlugin',
                        'component_schema_version': self.prototype_plugin.schema_version,
                        'accepted_host_interface_versions': sorted(self.prototype_plugin.accepted_host_interface_versions),
                        'compatible_runtime_modes': [
                            RUNTIME_MODE_PROTOTYPE_ONLY,
                            RUNTIME_MODE_FUSED_EXTERNAL,
                            RUNTIME_MODE_JOINT_TRAINING,
                            RUNTIME_MODE_CALIBRATION_ONLY,
                        ],
                    }
                if group_name == 'fusion':
                    return {
                        **common,
                        'component_name': 'Composer',
                        'component_schema_version': self.composer.schema_version,
                        'expected_host_score_schema_version': self.composer.expected_host_score_schema_version,
                        'expected_prototype_score_schema_version': self.composer.expected_prototype_score_schema_version,
                        'compatible_runtime_modes': [
                            RUNTIME_MODE_FUSED_EXTERNAL,
                            RUNTIME_MODE_JOINT_TRAINING,
                            RUNTIME_MODE_CALIBRATION_ONLY,
                        ],
                    }
                return dict(common)

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

    def _checkpointing_config(self, save_dir, enabled_groups=None):
        enabled_groups = set(enabled_groups or {'host'})
        def _enabled(group_name):
            return bool(group_name in enabled_groups)
        return {
            'groups': {
                'host': {'enabled': _enabled('host')},
                'prototype_bank': {'enabled': _enabled('prototype_bank')},
                'prototype_projector': {'enabled': _enabled('prototype_projector')},
                'fusion': {'enabled': _enabled('fusion')},
            },
            'save': {
                'dir': save_dir,
                'save_latest': True,
                'save_best': True,
                'keep_last_n': 1,
                'artifacts': {
                    'host': {
                        'enabled': _enabled('host'),
                        'filename_latest': 'host_latest.pth',
                        'filename_best': 'host_best.pth',
                    },
                    'prototype_bank': {'enabled': _enabled('prototype_bank'), 'filename_latest': 'prototype_bank_latest.pth', 'filename_best': 'prototype_bank_best.pth'},
                    'prototype_projector': {'enabled': _enabled('prototype_projector'), 'filename_latest': 'prototype_projector_latest.pth', 'filename_best': 'prototype_projector_best.pth'},
                    'fusion': {'enabled': _enabled('fusion'), 'filename_latest': 'fusion_latest.pth', 'filename_best': 'fusion_best.pth'},
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

    @staticmethod
    def _load_sources_for(*, host=None, prototype_bank=None, prototype_projector=None, fusion=None):
        return {
            'host': {'enabled': host is not None, 'path': host},
            'prototype_bank': {'enabled': prototype_bank is not None, 'path': prototype_bank},
            'prototype_projector': {'enabled': prototype_projector is not None, 'path': prototype_projector},
            'fusion': {'enabled': fusion is not None, 'path': fusion},
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

    def test_checkpoint_save_for_prototype_authority_with_valid_prototype_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._manager(self._checkpointing_config(tmpdir, enabled_groups={'prototype_bank'}))
            model = self._TinyComponentModel()
            authority_context = {
                'display_row': 'prototype-t2i',
                'source_row': 'prototype-t2i',
                'row_roles': {'prototype-t2i': 'prototype'},
                'row_metrics': {'prototype-t2i': {'R1': 71.1}},
                'candidates': {'host': None, 'prototype': 'prototype-t2i', 'fused': None},
            }
            manager.save_latest(
                model=model,
                epoch=1,
                global_step=10,
                metric_value=0.0,
                metric_row='prototype-t2i',
                metric_display_row='prototype-t2i',
                metric_source_row='prototype-t2i',
                authority_context=authority_context,
            )
            latest_path = os.path.join(tmpdir, 'prototype_bank_latest.pth')
            self.assertTrue(os.path.exists(latest_path))
            payload = torch.load(latest_path, map_location='cpu')
            self.assertEqual(payload['metric']['row'], 'prototype-t2i')
            self.assertTrue(payload['metric']['authority_valid'])

    def test_checkpoint_save_for_fused_authority_with_valid_fused_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._manager(self._checkpointing_config(tmpdir, enabled_groups={'fusion'}))
            model = self._TinyComponentModel()
            authority_context = {
                'display_row': 'pas-t2i',
                'source_row': 'pas-t2i',
                'row_roles': {'pas-t2i': 'fused'},
                'row_metrics': {'pas-t2i': {'R1': 81.5}},
                'candidates': {'host': None, 'prototype': None, 'fused': 'pas-t2i'},
            }
            manager.save_latest(
                model=model,
                epoch=1,
                global_step=10,
                metric_value=0.0,
                metric_row='pas-t2i',
                metric_display_row='pas-t2i',
                metric_source_row='pas-t2i',
                authority_context=authority_context,
            )
            latest_path = os.path.join(tmpdir, 'fusion_latest.pth')
            self.assertTrue(os.path.exists(latest_path))
            payload = torch.load(latest_path, map_location='cpu')
            self.assertEqual(payload['metric']['row'], 'pas-t2i')
            self.assertTrue(payload['metric']['authority_valid'])

    def test_checkpoint_load_valid_host_and_prototype_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_model = self._TinyCompatibleModel()
            save_cfg = self._checkpointing_config(tmpdir, enabled_groups={'host', 'prototype_bank'})
            save_manager = self._manager(save_cfg)
            authority_context = {
                'display_row': 'pas-t2i',
                'source_row': 'pas-t2i',
                'row_roles': {'host-t2i': 'host', 'prototype-t2i': 'prototype', 'pas-t2i': 'fused'},
                'row_metrics': {
                    'host-t2i': {'R1': 65.0},
                    'prototype-t2i': {'R1': 64.0},
                    'pas-t2i': {'R1': 66.0},
                },
                'candidates': {'host': 'host-t2i', 'prototype': 'prototype-t2i', 'fused': 'pas-t2i'},
            }
            save_manager.save_latest(
                model=source_model,
                epoch=1,
                global_step=10,
                metric_value=66.0,
                metric_row='pas-t2i',
                metric_display_row='pas-t2i',
                metric_source_row='pas-t2i',
                authority_context=authority_context,
            )
            host_path = os.path.join(tmpdir, 'host_latest.pth')
            proto_path = os.path.join(tmpdir, 'prototype_bank_latest.pth')
            self.assertTrue(os.path.exists(host_path))
            self.assertTrue(os.path.exists(proto_path))

            load_cfg = self._checkpointing_config(tmpdir, enabled_groups={'host', 'prototype_bank'})
            load_cfg['load']['enabled'] = True
            load_cfg['load']['strict'] = True
            load_cfg['load']['sources'] = self._load_sources_for(host=host_path, prototype_bank=proto_path)
            load_manager = self._manager(load_cfg)
            target_model = self._TinyCompatibleModel()
            load_manager.load_configured_groups(target_model)

    def test_checkpoint_load_rejects_wrong_component_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._TinyCompatibleModel()
            save_cfg = self._checkpointing_config(tmpdir, enabled_groups={'host'})
            save_manager = self._manager(save_cfg)
            authority_context = {
                'display_row': 'host-t2i',
                'source_row': 'host-t2i',
                'row_roles': {'host-t2i': 'host'},
                'row_metrics': {'host-t2i': {'R1': 70.0}},
                'candidates': {'host': 'host-t2i', 'prototype': None, 'fused': None},
            }
            save_manager.save_latest(
                model=model,
                epoch=1,
                global_step=10,
                metric_value=70.0,
                metric_row='host-t2i',
                metric_display_row='host-t2i',
                metric_source_row='host-t2i',
                authority_context=authority_context,
            )
            host_path = os.path.join(tmpdir, 'host_latest.pth')
            payload = torch.load(host_path, map_location='cpu')
            payload['metadata']['component_name'] = 'PrototypePlugin'
            payload.setdefault('metadata', {}).setdefault('compatibility', {})['component_name'] = 'PrototypePlugin'
            torch.save(payload, host_path)

            load_cfg = self._checkpointing_config(tmpdir, enabled_groups={'host'})
            load_cfg['load']['enabled'] = True
            load_cfg['load']['strict'] = True
            load_cfg['load']['sources'] = self._load_sources_for(host=host_path)
            load_manager = self._manager(load_cfg)
            with self.assertRaisesRegex(RuntimeError, 'component mismatch'):
                load_manager.load_configured_groups(model)

    def test_checkpoint_load_rejects_incompatible_composer_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._TinyCompatibleModel()
            save_cfg = self._checkpointing_config(tmpdir, enabled_groups={'fusion'})
            save_manager = self._manager(save_cfg)
            authority_context = {
                'display_row': 'pas-t2i',
                'source_row': 'pas-t2i',
                'row_roles': {'pas-t2i': 'fused'},
                'row_metrics': {'pas-t2i': {'R1': 79.0}},
                'candidates': {'host': None, 'prototype': None, 'fused': 'pas-t2i'},
            }
            save_manager.save_latest(
                model=model,
                epoch=1,
                global_step=10,
                metric_value=79.0,
                metric_row='pas-t2i',
                metric_display_row='pas-t2i',
                metric_source_row='pas-t2i',
                authority_context=authority_context,
            )
            fusion_path = os.path.join(tmpdir, 'fusion_latest.pth')
            payload = torch.load(fusion_path, map_location='cpu')
            payload['metadata']['expected_prototype_score_schema_version'] = 'prototype_score_v999'
            payload.setdefault('metadata', {}).setdefault('compatibility', {})['expected_prototype_score_schema_version'] = 'prototype_score_v999'
            torch.save(payload, fusion_path)

            load_cfg = self._checkpointing_config(tmpdir, enabled_groups={'fusion'})
            load_cfg['load']['enabled'] = True
            load_cfg['load']['strict'] = True
            load_cfg['load']['sources'] = self._load_sources_for(fusion=fusion_path)
            load_manager = self._manager(load_cfg)
            with self.assertRaisesRegex(RuntimeError, 'prototype score schema mismatch'):
                load_manager.load_configured_groups(model)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
