import os
import sys
import types
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.launch import build_nohup_log_path, build_run_name, get_effective_wandb_run_name, strip_nohup_flag


class LaunchHelperTests(unittest.TestCase):
    def test_strip_nohup_flag_removes_bool_variants(self):
        argv = ['train.py', '--config_file', 'cfg.yaml', '--nohup', 'true', '--epochs', '5']
        self.assertEqual(strip_nohup_flag(argv), ['train.py', '--config_file', 'cfg.yaml', '--epochs', '5'])

    def test_strip_nohup_flag_removes_equals_variant(self):
        argv = ['train.py', '--nohup=true', '--name', 'demo']
        self.assertEqual(strip_nohup_flag(argv), ['train.py', '--name', 'demo'])

    def test_effective_wandb_run_name_prefers_explicit_name(self):
        args = types.SimpleNamespace(wandb_run_name='wandb-demo', run_name='local-demo', name='demo', model_variant='pas_v1')
        self.assertEqual(get_effective_wandb_run_name(args), 'wandb-demo')

    def test_build_nohup_log_path_uses_wandb_name(self):
        args = types.SimpleNamespace(wandb_run_name='wandb-demo', run_name='local-demo', name='demo', model_variant='pas_v1')
        log_path = build_nohup_log_path(args)
        self.assertEqual(log_path.name, 'wandb-demo.log')
        self.assertEqual(log_path.parent.name, 'logs')

    def test_build_run_name_honors_env_override(self):
        args = types.SimpleNamespace(name='demo', model_variant='pas_v1')
        previous = os.environ.get('PAS_RUN_NAME_OVERRIDE')
        os.environ['PAS_RUN_NAME_OVERRIDE'] = 'fixed-run-name'
        try:
            self.assertEqual(build_run_name(args), 'fixed-run-name')
        finally:
            if previous is None:
                os.environ.pop('PAS_RUN_NAME_OVERRIDE', None)
            else:
                os.environ['PAS_RUN_NAME_OVERRIDE'] = previous


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
