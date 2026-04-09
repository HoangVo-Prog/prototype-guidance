import os
import sys
import unittest

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.pas_model import PASModel
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


class PASModelItselfClipCompatTests(unittest.TestCase):
    def _build_model_shell(self):
        model = PASModel.__new__(PASModel)
        model.base_model = _DummyItselfClip()
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

    def test_resolve_text_states_falls_back_to_projected_tokens(self):
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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
