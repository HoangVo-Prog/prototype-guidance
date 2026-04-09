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
    from utils.metric_logging import RoutingCoverageTracker, build_train_metrics, build_train_metrics_from_scalars, build_validation_metrics


@unittest.skipUnless(torch is not None, 'Torch is required for metric logging tests.')
class MetricLoggingTests(unittest.TestCase):
    def test_build_train_metrics_logs_surrogate_retrieval_keys(self):
        outputs = {
            'loss_total': 3.0,
            'loss_proxy': 2.0,
            'loss_proxy_image': 1.25,
            'loss_proxy_text': 0.75,
            'loss_proxy_text_exact': 0.1,
            'loss_ret': 0.5,
            'loss_align': 0.0,
            'loss_diag': 0.25,
            'loss_support': 0.125,
            'loss_diversity': 0.1,
            'loss_balance': 0.05,
            'loss_proxy_weighted': 2.0,
            'loss_ret_weighted': 0.5,
            'loss_align_weighted': 0.0,
            'loss_diag_weighted': 0.25,
            'loss_support_weighted': 0.0125,
            'loss_diversity_weighted': 0.1,
            'loss_balance_weighted': 0.05,
            'debug': {
                'surrogate_pairwise_logit_mean': 0.2,
                'surrogate_retrieval_grad_norm': 1.5,
            },
        }
        metrics = build_train_metrics(epoch=2, step=17, outputs=outputs, lr=1e-3, include_debug_metrics=True)
        self.assertEqual(metrics['train/loss_ret'], 0.5)
        self.assertEqual(metrics['train/loss_ret_weighted'], 0.5)
        self.assertEqual(metrics['debug/surrogate_pairwise_logit_mean'], 0.2)
        self.assertEqual(metrics['debug/surrogate_retrieval_grad_norm'], 1.5)

    def test_build_train_metrics_from_scalars_maps_losses_and_debug(self):
        metrics = build_train_metrics_from_scalars(
            epoch=2,
            step=17,
            scalar_metrics={
                'loss_total': 1.25,
                'loss_ret': 0.4,
                'routing_entropy': 2.0,
                'surrogate_retrieval_grad_norm': 1.3,
            },
            lr=5e-4,
        )
        self.assertEqual(metrics['train/epoch'], 2.0)
        self.assertEqual(metrics['train/step'], 17.0)
        self.assertEqual(metrics['train/lr'], 5e-4)
        self.assertEqual(metrics['train/loss_total'], 1.25)
        self.assertEqual(metrics['train/loss_ret'], 0.4)
        self.assertEqual(metrics['debug/routing_entropy'], 2.0)
        self.assertEqual(metrics['debug/surrogate_retrieval_grad_norm'], 1.3)

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
            loss_metrics={'loss_total': 0.75, 'loss_diag': 0.2, 'loss_ret': 0.4},
        )
        self.assertEqual(metrics['val/epoch'], 3.0)
        self.assertEqual(metrics['val/loss_total'], 0.75)
        self.assertEqual(metrics['val/loss_diag'], 0.2)
        self.assertEqual(metrics['val/loss_ret'], 0.4)
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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
