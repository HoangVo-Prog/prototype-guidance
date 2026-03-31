import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.build import PASModel
from utils.metrics import Evaluator


class DummyCLIPBackbone(nn.Module):
    def __init__(self, embed_dim=8, image_shape=(3, 4, 4), vocab_size=50010, text_length=77):
        super().__init__()
        self.image_input_dim = image_shape[0] * image_shape[1] * image_shape[2]
        self.visual = nn.Linear(self.image_input_dim, embed_dim)
        self.transformer = nn.Linear(embed_dim, embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(text_length, embed_dim))
        self.ln_final = nn.LayerNorm(embed_dim)
        self.text_projection = nn.Parameter(torch.eye(embed_dim))

    def encode_image_intermediates(self, image, return_all=False, average_attn_weights=True):
        flat = image.view(image.size(0), -1).float()
        cls_token = self.visual(flat)
        aux_token = torch.tanh(cls_token)
        tokens = torch.stack([cls_token, aux_token], dim=1)
        return {
            'projected_tokens': tokens,
            'pre_projection_tokens': tokens,
            'attention_weights': None,
        }

    def encode_text_intermediates(self, text, return_all=False, average_attn_weights=True):
        embedded = self.token_embedding(text.long())
        positional = self.positional_embedding[:text.size(1)].unsqueeze(0)
        hidden = self.transformer(embedded + positional)
        hidden = self.ln_final(hidden)
        projected = hidden @ self.text_projection
        return {
            'projected_tokens': projected,
            'pre_projection_tokens': hidden,
            'attention_weights': None,
        }


def build_args():
    return SimpleNamespace(
        pretrain_choice='ViT-B/16',
        img_size=(4, 4),
        stride_size=1,
        model_name='PAS',
        model_variant='pas_smoke',
        image_backbone='dummy_visual',
        text_backbone='dummy_text',
        embedding_dim=8,
        projection_dim=4,
        projector_hidden_dim=8,
        projector_dropout=0.0,
        projector_type='mlp2',
        normalize_projector_outputs=True,
        temperature=0.07,
        learn_logit_scale=True,
        text_length=77,
        vocab_size=50010,
        use_prototype_bank=True,
        use_image_conditioned_pooling=True,
        prototype_contextualization_enabled=True,
        return_debug_outputs=True,
        prototype_num_prototypes=4,
        prototype_dim=8,
        prototype_init='normalized_random',
        prototype_init_path=None,
        prototype_routing_type='cosine',
        prototype_temperature=0.07,
        prototype_contextualization_type='self_attention',
        prototype_contextualization_residual=True,
        normalize_for_self_interaction=True,
        normalize_for_routing=True,
        use_balancing_loss=False,
        prototype_balance_loss_weight=0.0,
        prototype_dead_threshold=0.005,
        use_diversity_loss=True,
        diversity_loss_weight=0.01,
        token_policy='content_only',
        token_scoring_type='cosine',
        normalize_for_token_scoring=True,
        token_pooling_temperature=0.07,
        special_token_ids={'bos_token_id': 49406, 'eos_token_id': 49407, 'pad_token_id': 0},
        error_on_empty_kept_tokens=True,
        freeze_image_backbone=True,
        freeze_text_backbone=True,
        prototype_eval_image_chunk_size=2,
        prototype_eval_text_chunk_size=2,
    )


def main():
    torch.manual_seed(19)
    args = build_args()
    batch = {
        'images': torch.randn(4, 3, 4, 4),
        'caption_ids': torch.tensor(
            [
                [49406, 11, 12, 49407, 0, 0],
                [49406, 21, 22, 49407, 0, 0],
                [49406, 31, 32, 49407, 0, 0],
                [49406, 41, 42, 49407, 0, 0],
            ],
            dtype=torch.long,
        ),
        'pids': torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }
    text_loader = [
        (torch.tensor([0, 1], dtype=torch.long), batch['caption_ids'][:2]),
        (torch.tensor([0, 1], dtype=torch.long), batch['caption_ids'][2:]),
    ]
    img_loader = [
        (torch.tensor([0, 1], dtype=torch.long), batch['images'][:2]),
        (torch.tensor([0, 1], dtype=torch.long), batch['images'][2:]),
    ]

    with mock.patch('model.build.build_CLIP_from_openai_pretrained', side_effect=lambda *a, **k: (DummyCLIPBackbone(), {'embed_dim': 8, 'vision_layers': 1, 'transformer_width': 8})):
        model = PASModel(args, num_classes=2)
        outputs = model(batch, return_debug=True)
        if not torch.isfinite(outputs['loss_total']):
            raise RuntimeError('Smoke forward produced a non-finite loss_total.')
        torch.testing.assert_close(outputs['debug']['alpha'].sum(dim=-1), torch.ones(outputs['debug']['alpha'].size(0)))
        torch.testing.assert_close(outputs['debug']['beta'].sum(dim=-1), torch.ones(outputs['debug']['beta'].size(0)))
        if not torch.equal(outputs['debug']['beta'][~outputs['debug']['token_keep_mask']], torch.zeros_like(outputs['debug']['beta'][~outputs['debug']['token_keep_mask']])):
            raise RuntimeError('Masked tokens received non-zero beta mass in the smoke test.')

        optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.05)
        losses = []
        model.train()
        for _ in range(8):
            optimizer.zero_grad()
            step_outputs = model(batch)
            step_loss = step_outputs['loss_total']
            if not torch.isfinite(step_loss):
                raise RuntimeError('Tiny-overfit loop produced a non-finite loss_total.')
            step_loss.backward()
            if model.prototype_head.prototype_bank.prototypes.grad is None:
                raise RuntimeError('Prototype bank did not receive gradients in the smoke test.')
            optimizer.step()
            losses.append(step_loss.detach().item())

        evaluator = Evaluator(img_loader, text_loader, args)
        top1 = evaluator.eval(model.eval())
        print(f'smoke_forward_loss={outputs["loss_total"].item():.6f}')
        print(f'tiny_overfit_start={losses[0]:.6f}')
        print(f'tiny_overfit_end={losses[-1]:.6f}')
        print(f'tiny_overfit_best={min(losses[1:]):.6f}')
        print(f'eval_top1={float(top1):.6f}')

        if losses[-1] >= losses[0]:
            raise RuntimeError('Tiny-overfit loop did not reduce the prototype loss.')


if __name__ == '__main__':
    main()
