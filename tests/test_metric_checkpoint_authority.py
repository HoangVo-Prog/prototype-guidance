import unittest

from utils.modular_checkpoint import MetricAuthorityPolicy, resolve_checkpointing_config
from utils.module_group_registry import CHECKPOINT_GROUPS


class MetricCheckpointAuthorityTests(unittest.TestCase):
    def test_checkpoint_groups_remove_fusion(self):
        self.assertNotIn('fusion', CHECKPOINT_GROUPS)

    def test_default_checkpointing_config_has_no_fusion_group(self):
        cfg = resolve_checkpointing_config({})
        self.assertNotIn('fusion', cfg['groups'])
        self.assertNotIn('fusion', cfg['save']['artifacts'])
        self.assertNotIn('fusion', cfg['load']['sources'])

    def test_checkpoint_authority_is_host_only(self):
        policy = MetricAuthorityPolicy(enabled=True, strict=True)
        self.assertEqual(policy.expected_bucket_for_group('host'), 'host')
        self.assertEqual(policy.expected_bucket_for_group('prototype_bank'), 'host')
        self.assertEqual(policy.expected_bucket_for_group('prototype_projector'), 'host')

    def test_prototype_semantic_loss_codepath_still_exists(self):
        with open('model/prototype/losses.py', 'r', encoding='utf-8') as handle:
            source = handle.read()
        self.assertIn('loss_semantic_pbt', source)
        self.assertIn('use_loss_semantic_pbt', source)


if __name__ == '__main__':
    unittest.main()
