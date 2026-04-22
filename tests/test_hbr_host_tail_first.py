import unittest

import torch

from model.prototype.losses import PrototypeLosses


class HostTailFirstHBRTests(unittest.TestCase):
    def _build_losses(self, **overrides) -> PrototypeLosses:
        kwargs = dict(
            temperature_init=0.07,
            learnable_temperature=False,
            normalize_embeddings=True,
            num_classes=4,
            embedding_dim=3,
            proxy_temperature=0.07,
            use_loss_diag=False,
            lambda_diag=0.0,
            diag_temperature=0.07,
            use_loss_semantic_pbt=False,
            lambda_semantic_pbt=0.0,
            use_loss_semantic_hardneg_margin=False,
            lambda_semantic_hardneg_margin=0.0,
            use_loss_semantic_hosthard_weighted=False,
            lambda_semantic_hosthard_weighted=0.0,
            use_loss_hbr=True,
            lambda_hbr=1.0,
            hbr_topk_hard_negatives=2,
            hbr_base_margin=0.5,
            hbr_tail_selection_mode='bottomk',
            hbr_tail_bottomk=2,
            hbr_inner_tail_weight_mode='uniform',
            hbr_proto_inner_tau=0.1,
            hbr_proto_inner_center=0.0,
            hbr_adaptive_margin_enabled=False,
            hbr_adaptive_margin_lambda=0.0,
            hbr_use_prototype_pair_signal=True,
            hbr_stopgrad_proto_signal=True,
            hbr_control_mode='proto_weight',
            prototype_method_role='retrieval_branch',
            prototype_semantic_enabled=False,
            semantic_structure_enabled=False,
            semantic_feature_space='prototype_projected',
            semantic_pbt_enabled=False,
            semantic_soft_target_enabled=True,
            semantic_target_temperature=0.01,
            semantic_pred_temperature=0.07,
            semantic_min_cluster_count_for_pbt=1.0,
            semantic_empty_cluster_policy='skip',
            use_diversity_loss=False,
            diversity_loss_weight=0.0,
            use_balance_loss=False,
            balance_loss_weight=0.0,
        )
        kwargs.update(overrides)
        return PrototypeLosses(**kwargs)

    @staticmethod
    def _host_scores() -> torch.Tensor:
        return torch.tensor(
            [
                [2.0, 1.8, 0.3, 0.1],
                [0.2, 2.2, 1.9, 0.4],
                [0.3, 1.5, 2.1, 1.7],
                [0.1, 0.5, 1.6, 2.0],
            ],
            dtype=torch.float32,
        )

    def test_tail_selection_uses_host_bottomk_pairs(self):
        losses = self._build_losses(hbr_inner_tail_weight_mode='uniform')
        host_scores = self._host_scores()
        exact = torch.randn(4, 3)

        info = losses._host_boundary_repair_loss(
            host_pairwise_logits=host_scores,
            host_pairwise_logits_global=None,
            host_pairwise_logits_local=None,
            exact_text_embeddings=exact,
            routing_weights=None,
            basis_bank=None,
            control_mode='proto_weight',
        )
        export = info['pairwise_export']
        observed_pairs = set(
            zip(
                export['anchor_index'].tolist(),
                export['negative_index'].tolist(),
            )
        )

        margin = host_scores.diagonal().unsqueeze(1) - host_scores
        offdiag = ~torch.eye(host_scores.size(0), dtype=torch.bool)
        margins_for_tail = margin.masked_fill(~offdiag, float('inf'))
        expected = set()
        expected_idx = torch.topk(margins_for_tail, k=2, dim=1, largest=False).indices
        for anchor in range(host_scores.size(0)):
            for negative in expected_idx[anchor].tolist():
                expected.add((anchor, negative))
        self.assertSetEqual(observed_pairs, expected)

    def test_uniform_mode_gives_equal_in_tail_weights(self):
        losses = self._build_losses(hbr_inner_tail_weight_mode='uniform')
        host_scores = self._host_scores()
        exact = torch.randn(4, 3)

        info = losses._host_boundary_repair_loss(
            host_pairwise_logits=host_scores,
            host_pairwise_logits_global=None,
            host_pairwise_logits_local=None,
            exact_text_embeddings=exact,
            routing_weights=None,
            basis_bank=None,
            control_mode='proto_weight',
        )
        export = info['pairwise_export']
        anchors = export['anchor_index'].tolist()
        omega = export['omega']
        for anchor in sorted(set(anchors)):
            mask = torch.tensor([idx == anchor for idx in anchors], dtype=torch.bool)
            row_weights = omega[mask]
            self.assertTrue(torch.allclose(row_weights, torch.ones_like(row_weights), atol=1e-6))

    def test_softmax_mode_normalizes_weights_per_anchor(self):
        losses = self._build_losses(hbr_inner_tail_weight_mode='softmax_proto')
        host_scores = self._host_scores()
        exact = torch.randn(4, 3)
        routing_weights = torch.softmax(torch.randn(4, 3), dim=-1)
        basis_bank = torch.randn(4, 3, 3)

        info = losses._host_boundary_repair_loss(
            host_pairwise_logits=host_scores,
            host_pairwise_logits_global=None,
            host_pairwise_logits_local=None,
            exact_text_embeddings=exact,
            routing_weights=routing_weights,
            basis_bank=basis_bank,
            control_mode='proto_weight',
        )
        export = info['pairwise_export']
        anchors = export['anchor_index'].tolist()
        omega = export['omega']
        for anchor in sorted(set(anchors)):
            mask = torch.tensor([idx == anchor for idx in anchors], dtype=torch.bool)
            row_sum = omega[mask].sum()
            self.assertAlmostEqual(float(row_sum.item()), 1.0, places=5)

    def test_disabled_adaptive_margin_matches_constant_delta_formula(self):
        losses = self._build_losses(
            hbr_base_margin=0.4,
            hbr_inner_tail_weight_mode='softmax_proto',
            hbr_adaptive_margin_enabled=False,
            hbr_adaptive_margin_lambda=0.0,
        )
        host_scores = self._host_scores()
        exact = torch.randn(4, 3)
        routing_weights = torch.softmax(torch.randn(4, 2), dim=-1)
        basis_bank = torch.randn(4, 2, 3)

        info = losses._host_boundary_repair_loss(
            host_pairwise_logits=host_scores,
            host_pairwise_logits_global=None,
            host_pairwise_logits_local=None,
            exact_text_embeddings=exact,
            routing_weights=routing_weights,
            basis_bank=basis_bank,
            control_mode='proto_weight',
        )
        export = info['pairwise_export']
        manual_loss = (
            export['omega'] * torch.relu(torch.full_like(export['margin_host'], 0.4) - export['margin_host'])
        ).sum() / host_scores.size(0)
        torch.testing.assert_close(info['loss'], manual_loss)

    def test_prototype_signal_does_not_change_outer_tail_selection(self):
        losses = self._build_losses(hbr_inner_tail_weight_mode='softmax_proto')
        host_scores = self._host_scores()
        exact = torch.randn(4, 3)

        routing_a = torch.softmax(torch.randn(4, 3), dim=-1)
        basis_a = torch.randn(4, 3, 3)
        info_a = losses._host_boundary_repair_loss(
            host_pairwise_logits=host_scores,
            host_pairwise_logits_global=None,
            host_pairwise_logits_local=None,
            exact_text_embeddings=exact,
            routing_weights=routing_a,
            basis_bank=basis_a,
            control_mode='proto_weight',
        )

        routing_b = torch.softmax(torch.randn(4, 3) * 4.0, dim=-1)
        basis_b = torch.randn(4, 3, 3) * 5.0
        info_b = losses._host_boundary_repair_loss(
            host_pairwise_logits=host_scores,
            host_pairwise_logits_global=None,
            host_pairwise_logits_local=None,
            exact_text_embeddings=exact,
            routing_weights=routing_b,
            basis_bank=basis_b,
            control_mode='proto_weight',
        )

        pairs_a = set(zip(info_a['pairwise_export']['anchor_index'].tolist(), info_a['pairwise_export']['negative_index'].tolist()))
        pairs_b = set(zip(info_b['pairwise_export']['anchor_index'].tolist(), info_b['pairwise_export']['negative_index'].tolist()))
        self.assertSetEqual(pairs_a, pairs_b)


if __name__ == '__main__':
    unittest.main()

