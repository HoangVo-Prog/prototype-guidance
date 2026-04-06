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

IMPORT_ERROR = None
if torch is not None:
    try:
        from model.host_heads import CLIPHostAdapter, ITSELFHostHead, build_host_head
        from model.interfaces import EncoderOutput
    except Exception as exc:  # pragma: no cover - environment-dependent
        CLIPHostAdapter = None
        ITSELFHostHead = None
        build_host_head = None
        EncoderOutput = None
        IMPORT_ERROR = exc


@unittest.skipUnless(
    torch is not None and CLIPHostAdapter is not None and ITSELFHostHead is not None and build_host_head is not None and EncoderOutput is not None,
    f'Host-head tests require torch and repo runtime imports: {IMPORT_ERROR}',
)
class HostHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.batch_size = 4
        self.embed_dim = 8
        self.grab_dim = 16
        self.seq_len = 6
        self.num_classes = 3
        self.image_tokens = torch.randn(self.batch_size, 5, self.embed_dim)
        self.text_tokens = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        self.image_attention = [torch.eye(5).unsqueeze(0).repeat(self.batch_size, 1, 1) for _ in range(2)]
        self.text_attention = [torch.eye(self.seq_len).unsqueeze(0).repeat(self.batch_size, 1, 1) for _ in range(2)]
        self.token_ids = torch.tensor(
            [
                [49406, 11, 12, 49407, 0, 0],
                [49406, 21, 22, 49407, 0, 0],
                [49406, 31, 32, 49407, 0, 0],
                [49406, 41, 42, 49407, 0, 0],
            ],
            dtype=torch.long,
        )
        self.pids = torch.tensor([0, 1, 0, 2], dtype=torch.long)

    def _build_args(self, **overrides):
        base = dict(
            host_type='itself',
            projection_dim=4,
            projector_hidden_dim=8,
            projector_dropout=0.0,
            projector_type='mlp2',
            normalize_projector_outputs=True,
            use_custom_projector=True,
            special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
            error_on_empty_kept_tokens=True,
            temperature=0.07,
            use_host_loss=True,
            itself_loss_names='tal+cid',
            itself_only_global=False,
            itself_select_ratio=0.4,
            itself_grab_embed_dim=self.grab_dim,
            itself_score_weight_global=0.68,
            itself_tau=0.015,
            itself_margin=0.1,
            itself_topk_type='mean',
            itself_layer_index=-1,
            itself_modify_k=False,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def _build_image_output(self):
        return EncoderOutput(
            tokens=self.image_tokens,
            pooled=self.image_tokens[:, 0, :],
            projected_tokens=self.image_tokens,
            projected_pooled=self.image_tokens[:, 0, :],
            pre_projection_tokens=self.image_tokens,
            pre_projection_pooled=self.image_tokens[:, 0, :],
            attention_weights=self.image_attention,
            token_mask=None,
            special_token_positions={},
            pooling_mode='cls',
            metadata={'encoder': 'image'},
        )

    def _build_text_output(self):
        eos_positions = self.token_ids.argmax(dim=-1)
        batch_indices = torch.arange(self.batch_size)
        return EncoderOutput(
            tokens=self.text_tokens,
            pooled=self.text_tokens[batch_indices, eos_positions],
            projected_tokens=self.text_tokens,
            projected_pooled=self.text_tokens[batch_indices, eos_positions],
            pre_projection_tokens=self.text_tokens,
            pre_projection_pooled=self.text_tokens[batch_indices, eos_positions],
            attention_weights=self.text_attention,
            token_mask=(self.token_ids != 0),
            special_token_positions={'eos': eos_positions, 'cls': torch.zeros(self.batch_size, dtype=torch.long)},
            pooling_mode='eos_only',
            metadata={'encoder': 'text'},
        )

    def test_build_host_head_selects_clip_adapter(self):
        head = build_host_head(self._build_args(host_type='clip'), input_dim=self.embed_dim, num_classes=self.num_classes)
        self.assertIsInstance(head, CLIPHostAdapter)

    def test_build_host_head_selects_itself_head(self):
        head = build_host_head(self._build_args(host_type='itself'), input_dim=self.embed_dim, num_classes=self.num_classes)
        self.assertIsInstance(head, ITSELFHostHead)

    def test_itself_encode_branches_expose_global_and_grab_embeddings(self):
        head = ITSELFHostHead(self._build_args(), input_dim=self.embed_dim, num_classes=self.num_classes)
        image_features = head.encode_image_branch(self._build_image_output(), return_debug=True)
        text_features = head.encode_text_branch(self._build_text_output(), self.token_ids, return_debug=True)
        self.assertEqual(tuple(image_features['global_image_embedding'].shape), (self.batch_size, self.embed_dim))
        self.assertEqual(tuple(text_features['global_text_embedding'].shape), (self.batch_size, self.embed_dim))
        self.assertEqual(tuple(image_features['grab_image_embedding'].shape), (self.batch_size, self.grab_dim))
        self.assertEqual(tuple(text_features['grab_text_embedding'].shape), (self.batch_size, self.grab_dim))
        self.assertIn('itself_only_global', image_features['debug'])
        self.assertIn('itself_only_global', text_features['debug'])

    def test_itself_only_global_disables_grab_branch(self):
        head = ITSELFHostHead(self._build_args(itself_only_global=True), input_dim=self.embed_dim, num_classes=self.num_classes)
        image_features = head.encode_image_branch(self._build_image_output(), return_debug=True)
        text_features = head.encode_text_branch(self._build_text_output(), self.token_ids, return_debug=True)
        self.assertIsNone(image_features['grab_image_embedding'])
        self.assertIsNone(text_features['grab_text_embedding'])
        similarity = head.compute_similarity_matrix(image_features, text_features)
        self.assertEqual(tuple(similarity.shape), (self.batch_size, self.batch_size))
        self.assertTrue(torch.isfinite(similarity).all())

    def test_itself_forward_returns_host_losses_and_similarity(self):
        head = ITSELFHostHead(self._build_args(), input_dim=self.embed_dim, num_classes=self.num_classes)
        outputs = head(
            self._build_image_output(),
            self._build_text_output(),
            self.token_ids,
            pids=self.pids,
            return_debug=True,
        )
        self.assertIn('losses', outputs)
        self.assertIn('metrics', outputs)
        self.assertIn('surrogate_pairwise_logits', outputs)
        self.assertTrue(torch.isfinite(outputs['losses']['loss_total']))
        self.assertTrue(torch.isfinite(outputs['losses']['loss_cid']))
        self.assertEqual(tuple(outputs['surrogate_pairwise_logits'].shape), (self.batch_size, self.batch_size))
        self.assertIn('itself_loss_tal', outputs['metrics'])
        self.assertIn('itself_loss_cid', outputs['metrics'])

    def test_itself_similarity_uses_weighted_global_and_grab_scores(self):
        head = ITSELFHostHead(self._build_args(itself_score_weight_global=0.25), input_dim=self.embed_dim, num_classes=self.num_classes)
        image_features = head.encode_image_branch(self._build_image_output(), return_debug=False)
        text_features = head.encode_text_branch(self._build_text_output(), self.token_ids, return_debug=False)
        combined = head.compute_similarity_matrix(image_features, text_features)
        global_only = head.compute_similarity_matrix(
            {
                **image_features,
                'grab_image_embedding': None,
            },
            {
                **text_features,
                'grab_text_embedding': None,
            },
        )
        self.assertEqual(tuple(combined.shape), (self.batch_size, self.batch_size))
        self.assertEqual(tuple(global_only.shape), (self.batch_size, self.batch_size))
        self.assertFalse(torch.allclose(combined, global_only))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
