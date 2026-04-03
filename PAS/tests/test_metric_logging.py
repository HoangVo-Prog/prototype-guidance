import math
import os
import sys
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.metric_logging import RoutingCoverageTracker, build_train_metrics


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
