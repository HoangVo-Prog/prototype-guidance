import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.metric_logging import build_train_metrics


class MetricLoggingTests(unittest.TestCase):
    def test_build_train_metrics_does_not_duplicate_proxy_branch_keys_for_wandb(self):
        outputs = {
            'loss_total': 3.0,
            'loss_proxy': 2.0,
            'loss_proxy_image': 1.25,
            'loss_proxy_text': 0.75,
            'loss_align': 0.5,
            'loss_diag': 0.25,
            'loss_support': 0.125,
            'loss_diversity': 0.1,
            'loss_balance': 0.05,
            'loss_proxy_weighted': 2.0,
            'loss_align_weighted': 0.5,
            'loss_diag_weighted': 0.25,
            'loss_support_weighted': 0.0125,
            'loss_diversity_weighted': 0.1,
            'loss_balance_weighted': 0.05,
            'debug': {},
        }
        metrics = build_train_metrics(epoch=2, step=17, outputs=outputs, lr=1e-3, include_debug_metrics=False)
        self.assertEqual(metrics['train/loss_proxy_image'], 1.25)
        self.assertEqual(metrics['train/loss_proxy_text'], 0.75)
        self.assertEqual(metrics['train/loss_support'], 0.125)
        self.assertEqual(metrics['train/loss_support_weighted'], 0.0125)
        self.assertNotIn('train/loss_proxy_image_branch', metrics)
        self.assertNotIn('train/loss_proxy_text_branch', metrics)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
