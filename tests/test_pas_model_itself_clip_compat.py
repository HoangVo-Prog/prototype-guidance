import os
import sys
import unittest

import torch
import torch.nn as nn


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.pas_model import PASModel
from model.hosts.itself import attach_itself_clip_text_intermediates
from model.interfaces import EncoderOutput


class _DummyItselfClip:
    def encode_image(self, image):
        batch_size = image.size(0)
        projected_tokens = torch.randn(batch_size, 193, 512)
        attention = torch.randn(batch_size, 193, 193)
        return projected_tokens, attention

    def encode_text(self, text):
        batch_size = text.size(0)
        projected_tokens = torch.randn(batch_size, 77, 512)
        attention = torch.randn(batch_size, 77, 77)
        return projected_tokens, attention


class _DummyTransformer:
    def __call__(self, inputs, average_attn_weights=True):
        del average_attn_weights
        x = inputs[0]
        return [x + 1.0, torch.zeros(x.size(1), x.size(0), x.size(0), device=x.device, dtype=x.dtype)]

    def forward(self, inputs, return_all=False, average_attn_weights=True):
        del average_attn_weights
        x = inputs[0]
        if return_all:
            return ([x + 1.0, torch.zeros(x.size(1), x.size(0), x.size(0), device=x.device, dtype=x.dtype)], [])
        return [x + 1.0, torch.zeros(x.size(1), x.size(0), x.size(0), device=x.device, dtype=x.dtype)]


class _DummyItselfClipWithInternals:
    def __init__(self, seq_len=6, width=4):
        self.token_embedding = nn.Embedding(32, width)
        self.positional_embedding = nn.Parameter(torch.zeros(seq_len, width))
        self.transformer = _DummyTransformer()
        self.ln_final = nn.Identity()
        self.text_projection = nn.Parameter(2.0 * torch.eye(width))
        self.dtype = torch.float32

    def encode_image(self, image):
        batch_size = image.size(0)
        projected_tokens = torch.randn(batch_size, 193, 4)
        attention = torch.randn(batch_size, 193, 193)
        return projected_tokens, attention


class PASModelItselfClipCompatTests(unittest.TestCase):
    def _build_model_shell(self):
        model = PASModel.__new__(PASModel)
        model.base_model = _DummyItselfClip()
        model.host_type = 'clip'
        return model

    def test_image_fallback_uses_encode_image_when_intermediates_are_missing(self):
        model = self._build_model_shell()
        outputs = model._encode_image_intermediates(
            image=torch.randn(2, 3, 384, 128),
            return_all=False,
            average_attn_weights=True,
        )
        self.assertIn('projected_tokens', outputs)
        self.assertIn('attention_weights', outputs)
        self.assertEqual(tuple(outputs['projected_tokens'].shape), (2, 193, 512))

    def test_text_fallback_uses_encode_text_when_intermediates_are_missing(self):
        model = self._build_model_shell()
        outputs = model._encode_text_intermediates(
            text=torch.randint(low=0, high=10, size=(2, 77)),
            return_all=False,
            average_attn_weights=True,
        )
        self.assertIn('projected_tokens', outputs)
        self.assertIn('attention_weights', outputs)
        self.assertEqual(tuple(outputs['projected_tokens'].shape), (2, 77, 512))

    def test_itself_adapter_exposes_pre_projection_text_states(self):
        model = PASModel.__new__(PASModel)
        model.base_model = _DummyItselfClipWithInternals()
        model.host_type = 'itself'
        attach_itself_clip_text_intermediates(model.base_model)
        token_ids = torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 0, 0]], dtype=torch.long)

        outputs = model._encode_text_intermediates(
            text=token_ids,
            return_all=False,
            average_attn_weights=True,
        )

        self.assertIsNotNone(outputs['pre_projection_tokens'])
        self.assertEqual(tuple(outputs['pre_projection_tokens'].shape), (2, 6, 4))
        self.assertFalse(torch.equal(outputs['projected_tokens'], outputs['pre_projection_tokens']))

    def test_resolve_text_states_falls_back_to_projected_tokens_for_non_itself(self):
        model = self._build_model_shell()
        projected = torch.randn(2, 77, 512)
        text_output = EncoderOutput(
            tokens=projected,
            pooled=projected[:, 0, :],
            projected_tokens=projected,
            projected_pooled=projected[:, 0, :],
            pre_projection_tokens=None,
            pre_projection_pooled=None,
        )
        resolved = model._resolve_text_states(text_output)
        self.assertTrue(torch.equal(resolved, projected))

    def test_itself_resolve_text_states_raises_when_pre_projection_unavailable(self):
        model = PASModel.__new__(PASModel)
        model.host_type = 'itself'
        projected = torch.randn(2, 77, 512)
        text_output = EncoderOutput(
            tokens=projected,
            pooled=projected[:, 0, :],
            projected_tokens=projected,
            projected_pooled=projected[:, 0, :],
            pre_projection_tokens=None,
            pre_projection_pooled=None,
        )
        with self.assertRaisesRegex(RuntimeError, 'ITSELF host requires text pre-projection token states'):
            _ = model._resolve_text_states(text_output)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
