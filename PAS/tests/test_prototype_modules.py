import os
import sys
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if torch is not None:
    from model.prototype import PrototypeConditionedTextHead, PrototypeLosses


@unittest.skipUnless(torch is not None, 'PyTorch is required for prototype module tests.')
class PrototypeModuleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.batch_size = 3
        self.seq_len = 6
        self.feature_dim = 8
        self.num_prototypes = 5
        self.num_classes = 3
        self.token_ids = torch.tensor(
            [
                [49406, 11, 12, 49407, 0, 0],
                [49406, 21, 49407, 0, 0, 0],
                [49406, 31, 32, 33, 49407, 0],
            ],
            dtype=torch.long,
        )
        self.attention_mask = self.token_ids.ne(0)
        self.special_positions = {
            'cls': torch.zeros(self.batch_size, dtype=torch.long),
            'eos': torch.tensor([3, 2, 4], dtype=torch.long),
        }

    def _build_head(self, **overrides):
        base = dict(
            input_dim=self.feature_dim,
            num_prototypes=self.num_prototypes,
            prototype_dim=self.feature_dim,
            projector_output_dim=4,
            projector_hidden_dim=12,
            projector_dropout=0.0,
            projector_type='mlp2',
            normalize_projector_outputs=True,
            prototype_init='normalized_random',
            prototype_init_path=None,
            prototype_init_hybrid_ratio=0.5,
            prototype_init_max_iters=50,
            prototype_init_tol=1e-4,
            prototype_init_seed=17,
            routing_type='cosine',
            routing_temperature=0.07,
            token_scoring_type='cosine',
            token_temperature=0.07,
            normalize_for_token_scoring=True,
            token_policy='content_only',
            special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
            error_on_empty_kept_tokens=True,
            contextualization_enabled=True,
            contextualization_type='self_attention',
            contextualization_residual=True,
            normalize_for_self_interaction=True,
            normalize_for_routing=True,
            num_classes=self.num_classes,
            proxy_temperature=0.2,
            lambda_proxy=1.0,
            use_loss_proxy_text_exact=True,
            use_loss_ret=True,
            lambda_ret=1.0,
            use_loss_align=False,
            lambda_align=0.0,
            use_loss_diag=True,
            lambda_diag=1.0,
            use_loss_support=False,
            support_loss_weight=0.0,
            support_min=2.0,
            use_diversity_loss=True,
            diversity_loss_weight=0.05,
            use_balance_loss=True,
            balance_loss_weight=0.1,
            contrastive_temperature_init=0.07,
            learnable_contrastive_temperature=False,
            dead_prototype_threshold=0.01,
        )
        base.update(overrides)
        return PrototypeConditionedTextHead(**base)

    def test_surrogate_retrieval_loss_prefers_stronger_diagonal_logits(self):
        losses = PrototypeLosses(
            temperature_init=0.07,
            normalize_embeddings=True,
            num_classes=self.num_classes,
            embedding_dim=4,
            proxy_temperature=0.2,
            use_loss_proxy_image=False,
            use_loss_proxy_text=False,
            use_loss_proxy_text_exact=False,
            use_loss_align=False,
            use_loss_diag=False,
            use_loss_ret=True,
            lambda_ret=1.0,
            use_diversity_loss=False,
            use_balance_loss=False,
            balance_loss_weight=0.0,
        )
        strong = torch.tensor(
            [
                [5.0, 0.5, 0.1],
                [0.4, 4.5, 0.2],
                [0.3, 0.6, 4.0],
            ],
            dtype=torch.float32,
        )
        weak = torch.tensor(
            [
                [0.5, 1.4, 0.8],
                [1.1, 0.4, 1.2],
                [1.0, 0.9, 0.3],
            ],
            dtype=torch.float32,
        )
        self.assertLess(losses.surrogate_retrieval_loss(strong)['loss'].item(), losses.surrogate_retrieval_loss(weak)['loss'].item())

    def test_loss_module_reports_weighted_surrogate_retrieval(self):
        losses = PrototypeLosses(
            temperature_init=0.07,
            normalize_embeddings=True,
            num_classes=self.num_classes,
            embedding_dim=4,
            proxy_temperature=0.2,
            use_loss_proxy_image=False,
            use_loss_proxy_text=False,
            use_loss_proxy_text_exact=False,
            use_loss_align=False,
            use_loss_diag=True,
            lambda_diag=1.0,
            use_loss_ret=True,
            lambda_ret=2.0,
            use_diversity_loss=False,
            use_balance_loss=False,
            balance_loss_weight=0.0,
        )
        outputs = losses(
            torch.randn(self.batch_size, 4),
            torch.randn(self.batch_size, 4),
            torch.randn(self.batch_size, 4),
            pids=torch.tensor([0, 1, 2]),
            surrogate_pairwise_logits=torch.tensor(
                [
                    [3.0, 0.2, 0.1],
                    [0.3, 2.5, 0.4],
                    [0.1, 0.4, 2.8],
                ],
                dtype=torch.float32,
            ),
        )
        self.assertAlmostEqual(outputs['loss_ret_weighted'].item(), 2.0 * outputs['loss_ret'].item(), places=6)

    def test_head_forward_exposes_surrogate_pairwise_logits(self):
        head = self._build_head()
        outputs = head(
            image_embeddings=torch.randn(self.batch_size, self.feature_dim, requires_grad=True),
            text_token_states=torch.randn(self.batch_size, self.seq_len, self.feature_dim, requires_grad=True),
            token_ids=self.token_ids,
            pids=torch.tensor([0, 1, 2], dtype=torch.long),
            attention_mask=self.attention_mask,
            special_token_positions=self.special_positions,
            return_debug=False,
        )
        self.assertEqual(tuple(outputs['surrogate_pairwise_logits'].shape), (self.batch_size, self.batch_size))
        self.assertIn('loss_ret', outputs['losses'])
        self.assertTrue(torch.isfinite(outputs['losses']['loss_ret']))

    def test_head_surrogate_pairwise_logits_receive_gradients(self):
        head = self._build_head(
            use_loss_proxy_image=False,
            use_loss_proxy_text=False,
            use_loss_proxy_text_exact=False,
            use_loss_align=False,
            use_loss_diag=False,
            use_loss_ret=True,
            use_diversity_loss=False,
            use_balance_loss=False,
            balance_loss_weight=0.0,
        )
        outputs = head(
            image_embeddings=torch.randn(self.batch_size, self.feature_dim, requires_grad=True),
            text_token_states=torch.randn(self.batch_size, self.seq_len, self.feature_dim, requires_grad=True),
            token_ids=self.token_ids,
            pids=torch.tensor([0, 1, 2], dtype=torch.long),
            attention_mask=self.attention_mask,
            special_token_positions=self.special_positions,
            return_debug=False,
        )
        outputs['surrogate_pairwise_logits'].retain_grad()
        outputs['losses']['loss_total'].backward()
        self.assertIsNotNone(outputs['surrogate_pairwise_logits'].grad)

    def test_compute_approximate_pairwise_similarity_runs(self):
        head = self._build_head()
        image_projected = torch.randn(2, 4)
        routing_weights = torch.softmax(torch.randn(2, self.num_prototypes), dim=-1)
        basis_bank = torch.randn(3, self.num_prototypes, self.feature_dim)
        similarity = head.compute_approximate_pairwise_similarity(image_projected, routing_weights, basis_bank, image_chunk_size=1, text_chunk_size=2)
        self.assertEqual(tuple(similarity.shape), (3, 2))
        self.assertTrue(torch.isfinite(similarity).all())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
