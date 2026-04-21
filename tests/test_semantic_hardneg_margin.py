import unittest

import torch

from model.prototype.losses import PrototypeLosses
from utils.metric_logging import TRAIN_LOSS_KEYS


class SemanticHardnegMarginTests(unittest.TestCase):
    def _build_losses(self, **overrides) -> PrototypeLosses:
        kwargs = dict(
            temperature_init=0.07,
            learnable_temperature=False,
            normalize_embeddings=True,
            num_classes=2,
            embedding_dim=4,
            proxy_temperature=0.07,
            use_loss_diag=False,
            lambda_diag=1.0,
            diag_temperature=0.07,
            use_loss_semantic_pbt=False,
            lambda_semantic_pbt=0.0,
            use_loss_semantic_hardneg_margin=True,
            lambda_semantic_hardneg_margin=1.7,
            semantic_hardneg_margin=0.05,
            semantic_hardneg_eps=1e-8,
            prototype_method_role='semantic_structure',
            prototype_semantic_enabled=True,
            semantic_structure_enabled=True,
            semantic_feature_space='prototype_projected',
            semantic_pbt_enabled=True,
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

    def test_hardneg_margin_matches_manual_computation(self):
        torch.manual_seed(7)
        losses = self._build_losses()

        image = torch.randn(4, 4)
        surrogate_text = torch.randn(4, 4)
        exact_text = torch.randn(4, 4)
        base_prototypes = torch.randn(3, 4)
        host_scores = torch.tensor(
            [
                [2.0, 1.0, 0.2, 0.3],
                [0.1, 2.1, 1.7, 0.4],
                [0.5, 1.3, 1.9, 1.4],
                [0.9, 0.2, 1.6, 2.2],
            ],
            dtype=torch.float32,
        )

        outputs = losses(
            image,
            surrogate_text,
            exact_text,
            pids=torch.arange(4, dtype=torch.long),
            host_pairwise_logits=host_scores,
            semantic_image_student_embeddings=image,
            semantic_text_student_embeddings=surrogate_text,
            semantic_text_teacher_embeddings=exact_text,
            semantic_base_prototypes=base_prototypes,
        )
        semantic_info = losses._semantic_pbt_loss(
            image_student=image,
            text_student=surrogate_text,
            text_teacher=exact_text,
            base_prototypes=base_prototypes,
        )
        manual = losses._semantic_hardneg_margin_loss(
            semantic_info=semantic_info,
            host_pairwise_logits=host_scores,
        )

        torch.testing.assert_close(outputs['loss_semantic_hardneg_margin'], manual['loss'])
        torch.testing.assert_close(outputs['loss_semantic_hardneg_margin_image'], manual['loss_image'])
        torch.testing.assert_close(outputs['loss_semantic_hardneg_margin_text'], manual['loss_text'])
        torch.testing.assert_close(
            outputs['loss_semantic_hardneg_margin_weighted'],
            outputs['lambda_semantic_hardneg_margin'] * outputs['loss_semantic_hardneg_margin'],
        )

    def test_batch_size_one_returns_zero(self):
        losses = self._build_losses()
        x = torch.randn(1, 4)
        base_prototypes = torch.randn(2, 4)
        host_scores = torch.tensor([[1.0]], dtype=torch.float32)

        outputs = losses(
            x,
            x,
            x,
            pids=torch.arange(1, dtype=torch.long),
            host_pairwise_logits=host_scores,
            semantic_image_student_embeddings=x,
            semantic_text_student_embeddings=x,
            semantic_text_teacher_embeddings=x,
            semantic_base_prototypes=base_prototypes,
        )
        self.assertEqual(float(outputs['loss_semantic_hardneg_margin'].item()), 0.0)
        self.assertEqual(float(outputs['loss_semantic_hardneg_margin_image'].item()), 0.0)
        self.assertEqual(float(outputs['loss_semantic_hardneg_margin_text'].item()), 0.0)

    def test_hardneg_margin_uses_dedicated_ramp_scale_when_provided(self):
        torch.manual_seed(11)
        losses = self._build_losses()

        x = torch.randn(4, 4)
        base_prototypes = torch.randn(3, 4)
        host_scores = torch.randn(4, 4)

        baseline = losses(
            x,
            x,
            x,
            pids=torch.arange(4, dtype=torch.long),
            host_pairwise_logits=host_scores,
            semantic_image_student_embeddings=x,
            semantic_text_student_embeddings=x,
            semantic_text_teacher_embeddings=x,
            semantic_base_prototypes=base_prototypes,
        )
        half_scaled = losses(
            x,
            x,
            x,
            pids=torch.arange(4, dtype=torch.long),
            host_pairwise_logits=host_scores,
            semantic_image_student_embeddings=x,
            semantic_text_student_embeddings=x,
            semantic_text_teacher_embeddings=x,
            semantic_base_prototypes=base_prototypes,
            semantic_hardneg_margin_loss_scale=0.5,
        )

        torch.testing.assert_close(
            half_scaled['loss_semantic_hardneg_margin_weighted'],
            baseline['loss_semantic_hardneg_margin_weighted'] * 0.5,
        )
        self.assertEqual(float(half_scaled['loss_semantic_hardneg_margin_scale'].item()), 0.5)

    def test_hardneg_margin_scale_defaults_to_semantic_pbt_scale(self):
        losses = self._build_losses()
        x = torch.randn(3, 4)
        base_prototypes = torch.randn(2, 4)
        host_scores = torch.randn(3, 3)

        outputs = losses(
            x,
            x,
            x,
            pids=torch.arange(3, dtype=torch.long),
            host_pairwise_logits=host_scores,
            semantic_image_student_embeddings=x,
            semantic_text_student_embeddings=x,
            semantic_text_teacher_embeddings=x,
            semantic_base_prototypes=base_prototypes,
            semantic_pbt_loss_scale=0.25,
        )
        self.assertEqual(float(outputs['loss_semantic_hardneg_margin_scale'].item()), 0.25)

    def test_missing_semantic_structures_raise(self):
        losses = self._build_losses()
        x = torch.randn(3, 4)
        host_scores = torch.randn(3, 3)
        with self.assertRaisesRegex(ValueError, 'requires semantic student probabilities/targets'):
            losses(
                x,
                x,
                x,
                pids=torch.arange(3, dtype=torch.long),
                host_pairwise_logits=host_scores,
                semantic_image_student_embeddings=None,
                semantic_text_student_embeddings=None,
                semantic_text_teacher_embeddings=None,
                semantic_base_prototypes=None,
            )

    def test_new_loss_keys_are_in_tracked_loss_surface(self):
        for key in (
            'loss_semantic_hardneg_margin',
            'loss_semantic_hardneg_margin_image',
            'loss_semantic_hardneg_margin_text',
        ):
            self.assertIn(key, TRAIN_LOSS_KEYS)


if __name__ == '__main__':
    unittest.main()
