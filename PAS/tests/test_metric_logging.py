import math
import os
import sys
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if torch is not None:
    from utils.metric_logging import RoutingCoverageTracker, build_train_metrics, build_validation_metrics


@unittest.skipUnless(torch is not None, 'Torch is required for metric logging tests.')
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
            'debug': {
                'routing_top1_active_count_window_100': 7.0,
                'prototype_usage_entropy_window_500': 1.75,
            },
        }
        metrics = build_train_metrics(epoch=2, step=17, outputs=outputs, lr=1e-3, include_debug_metrics=True)
        self.assertEqual(metrics['train/loss_proxy_image'], 1.25)
        self.assertEqual(metrics['train/loss_proxy_text'], 0.75)
        self.assertEqual(metrics['train/loss_support'], 0.125)
        self.assertEqual(metrics['train/loss_support_weighted'], 0.0125)
        self.assertEqual(metrics['debug/routing_top1_active_count_window_100'], 7.0)
        self.assertEqual(metrics['debug/prototype_usage_entropy_window_500'], 1.75)
        self.assertNotIn('train/loss_proxy_image_branch', metrics)
        self.assertNotIn('train/loss_proxy_text_branch', metrics)

    def test_build_train_metrics_skips_duplicate_alias_debug_metrics_for_wandb(self):
        outputs = {
            'loss_total': 1.0,
            'loss_proxy': 0.0,
            'loss_proxy_image': 0.0,
            'loss_proxy_text': 0.0,
            'loss_proxy_text_exact': 0.0,
            'loss_ret_exact': 0.0,
            'loss_align': 0.0,
            'loss_diag': 0.0,
            'loss_support': 0.0,
            'loss_diversity': 0.0,
            'loss_balance': 0.0,
            'loss_proxy_weighted': 0.0,
            'loss_ret_exact_weighted': 0.0,
            'loss_align_weighted': 0.0,
            'loss_diag_weighted': 0.0,
            'loss_support_weighted': 0.0,
            'loss_diversity_weighted': 0.0,
            'loss_balance_weighted': 0.0,
            'debug': {
                't_pool_norm': 2.0,
                'surrogate_t_pool_norm': 2.0,
                'image_embed_norm': 3.0,
                'image_embed_norm_raw': 3.0,
                'text_embed_norm': 4.0,
                'surrogate_text_embed_norm_raw': 4.0,
                'exact_text_embed_norm': 5.0,
                'exact_text_embed_norm_raw': 5.0,
            },
        }
        metrics = build_train_metrics(epoch=1, step=5, outputs=outputs, lr=1e-3, include_debug_metrics=True)
        self.assertIn('debug/surrogate_t_pool_norm', metrics)
        self.assertIn('debug/image_embed_norm_raw', metrics)
        self.assertIn('debug/surrogate_text_embed_norm_raw', metrics)
        self.assertIn('debug/exact_text_embed_norm_raw', metrics)
        self.assertNotIn('debug/t_pool_norm', metrics)
        self.assertNotIn('debug/image_embed_norm', metrics)
        self.assertNotIn('debug/text_embed_norm', metrics)
        self.assertNotIn('debug/exact_text_embed_norm', metrics)

    def test_build_validation_metrics_merges_loss_and_retrieval_under_val_namespace(self):
        evaluator = type('EvaluatorStub', (), {
            'latest_metrics': {
                'val/pas/R1': 42.0,
                'val/debug/eval_exact_margin_mean': 0.15,
            }
        })()
        metrics = build_validation_metrics(
            epoch=3,
            evaluator=evaluator,
            loss_metrics={'loss_total': 0.75, 'loss_diag': 0.2},
        )
        self.assertEqual(metrics['val/epoch'], 3.0)
        self.assertEqual(metrics['val/loss_total'], 0.75)
        self.assertEqual(metrics['val/loss_diag'], 0.2)
        self.assertEqual(metrics['val/pas/R1'], 42.0)
        self.assertEqual(metrics['val/debug/eval_exact_margin_mean'], 0.15)

    def test_routing_coverage_tracker_window_metrics_capture_rotation(self):
        tracker = RoutingCoverageTracker(window_sizes=(3,), activity_epsilons=(1e-3, 1e-2))
        for alpha in (
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 1.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
        ):
            tracker.update(alpha)

        metrics = tracker.get_debug_metrics()
        self.assertEqual(metrics['routing_top1_active_count_window_3'], 3.0)
        self.assertEqual(metrics['routing_top1_dead_count_window_3'], 0.0)
        self.assertAlmostEqual(metrics['routing_top1_usage_max_window_3'], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics['routing_top1_usage_entropy_window_3'], math.log(3.0), places=6)
        self.assertEqual(metrics['prototype_active_count_eps_1e-3_window_3'], 3.0)
        self.assertEqual(metrics['prototype_active_count_eps_1e-2_window_3'], 3.0)
        self.assertAlmostEqual(metrics['prototype_usage_max_window_3'], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics['prototype_usage_entropy_window_3'], math.log(3.0), places=6)

    def test_routing_coverage_tracker_window_and_epoch_metrics_capture_hub_collapse(self):
        tracker = RoutingCoverageTracker(window_sizes=(2,), activity_epsilons=(1e-3, 1e-2))
        tracker.update(torch.tensor([[0.7, 0.2, 0.1, 0.0, 0.0]]))
        tracker.update(torch.tensor([[0.6, 0.3, 0.0, 0.1, 0.0]]))
        tracker.update(torch.tensor([[0.8, 0.1, 0.0, 0.1, 0.0]]))

        debug_metrics = tracker.get_debug_metrics()
        self.assertEqual(debug_metrics['routing_top1_active_count_window_2'], 1.0)
        self.assertEqual(debug_metrics['routing_top1_dead_count_window_2'], 4.0)
        self.assertAlmostEqual(debug_metrics['routing_top1_usage_max_window_2'], 1.0, places=6)

        epoch_metrics = tracker.flush_epoch_metrics(epoch=4)
        self.assertEqual(epoch_metrics['train_epoch/epoch'], 4.0)
        self.assertEqual(epoch_metrics['train_epoch/routing_top1_active_count'], 1.0)
        self.assertEqual(epoch_metrics['train_epoch/routing_top1_dead_count'], 4.0)
        self.assertAlmostEqual(epoch_metrics['train_epoch/routing_top1_usage_max'], 1.0, places=6)
        self.assertEqual(epoch_metrics['train_epoch/prototype_active_count_eps_1e-3'], 4.0)
        self.assertEqual(epoch_metrics['train_epoch/prototype_active_count_eps_1e-2'], 4.0)
        self.assertEqual(epoch_metrics['train_epoch/routing_top3_active_count'], 4.0)

        tracker.reset_epoch()
        self.assertEqual(tracker.flush_epoch_metrics(epoch=5), {'train_epoch/epoch': 5.0})


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
