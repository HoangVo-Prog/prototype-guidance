import unittest

from utils.metric_logging import build_train_metrics_from_scalars, build_validation_metrics


class MetricLoggingHBRDiagnosticsTests(unittest.TestCase):
    def test_train_hbr_metrics_map_to_stable_wandb_namespace(self):
        scalar_metrics = {
            'num_hard_pairs': 12.0,
            'num_active_hbr_pairs': 9.0,
            'active_hbr_pair_ratio': 0.75,
            'hbr_host_tail_margin_mean': 0.22,
            'hbr_proto_signal_mean_in_tail': 0.41,
            'hbr_omega_entropy_in_tail': 0.11,
            # Alias form without `hbr_` prefix must map identically.
            'active_pairs_per_anchor': 2.5,
        }
        metrics = build_train_metrics_from_scalars(
            epoch=2,
            step=15,
            scalar_metrics=scalar_metrics,
            lr=1e-4,
        )
        self.assertIn('train/hbr/num_hard_pairs', metrics)
        self.assertIn('train/hbr/num_active_hbr_pairs', metrics)
        self.assertIn('train/hbr/active_hbr_pair_ratio', metrics)
        self.assertIn('train/hbr/host_tail_margin_mean', metrics)
        self.assertIn('train/hbr/proto_signal_mean_in_tail', metrics)
        self.assertIn('train/hbr/omega_entropy_in_tail', metrics)
        self.assertIn('train/hbr/active_pairs_per_anchor', metrics)

    def test_validation_hbr_scalars_use_val_hbr_namespace(self):
        metrics = build_validation_metrics(
            epoch=3,
            loss_metrics={
                'hbr_host_tail_margin_mean': 0.17,
                'hbr_host_tail_margin_p10': 0.05,
                'hbr_hardest_tail_margin_mean': 0.03,
                'hbr_hardest_tail_margin_p10': -0.02,
                'hbr_nonzero_omega_fraction': 0.61,
                'hbr_omega_mean_in_tail': 0.44,
                'hbr_omega_max_in_tail': 0.99,
                'hbr_omega_entropy_in_tail': 0.16,
                'hbr_proto_signal_mean_in_tail': 0.38,
                'hbr_proto_signal_std_in_tail': 0.07,
                'hbr_proto_signal_vs_host_margin_corr_in_tail': 0.23,
                'hbr_top_weight_in_host_tail_fraction': 1.0,
                'hbr_top_weight_overlap_with_bottom_host_margin': 0.84,
                'hbr_active_pairs_per_anchor': 1.75,
                'hbr_hbr_loss_active_mean': 0.28,
            },
        )
        expected_keys = (
            'val/hbr/host_tail_margin_mean',
            'val/hbr/host_tail_margin_p10',
            'val/hbr/hardest_tail_margin_mean',
            'val/hbr/hardest_tail_margin_p10',
            'val/hbr/nonzero_omega_fraction',
            'val/hbr/omega_mean_in_tail',
            'val/hbr/omega_max_in_tail',
            'val/hbr/omega_entropy_in_tail',
            'val/hbr/proto_signal_mean_in_tail',
            'val/hbr/proto_signal_std_in_tail',
            'val/hbr/proto_signal_vs_host_margin_corr_in_tail',
            'val/hbr/top_weight_in_host_tail_fraction',
            'val/hbr/top_weight_overlap_with_bottom_host_margin',
            'val/hbr/active_pairs_per_anchor',
            'val/hbr/hbr_loss_active_mean',
        )
        for key in expected_keys:
            self.assertIn(key, metrics)


if __name__ == '__main__':
    unittest.main()
