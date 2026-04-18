import types
import unittest

import torch

from model.runtime_modes import normalize_runtime_mode, resolve_runtime_mode_from_args
from utils.config import validate_config_data
from utils.metrics import Evaluator


class _TinyLoader:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        for row in self._rows:
            yield row


class _HostOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def encode_text_for_retrieval(self, caption):
        return {'text_projected': caption.float()}

    def encode_image_for_retrieval(self, image):
        return {'image_projected': image.float()}

    def compute_retrieval_similarity(self, image_features, text_features):
        return text_features['text_projected'] @ image_features['image_projected'].t()


class StructuralSplitRuntimeTests(unittest.TestCase):
    def test_obsolete_runtime_modes_are_rejected(self):
        for removed in ('prototype_only', 'fused_external', 'calibration_only'):
            with self.assertRaises(ValueError):
                normalize_runtime_mode(removed)

    def test_auto_runtime_mode_is_host_only_for_inference(self):
        args = types.SimpleNamespace(runtime_mode='auto', use_prototype_branch=True)
        self.assertEqual(resolve_runtime_mode_from_args(args, for_training=False), 'host_only')

    def test_host_only_retrieval_is_only_eval_row(self):
        args = types.SimpleNamespace(
            retrieval_metrics=['R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum'],
            amp=False,
            log_debug_metrics=False,
        )
        txt = _TinyLoader([
            (torch.tensor([1]), torch.tensor([[1.0, 0.0]])),
            (torch.tensor([2]), torch.tensor([[0.0, 1.0]])),
        ])
        img = _TinyLoader([
            (torch.tensor([1]), torch.tensor([[1.0, 0.0]])),
            (torch.tensor([2]), torch.tensor([[0.0, 1.0]])),
        ])
        evaluator = Evaluator(img, txt, args)
        r1 = evaluator.eval(_HostOnlyModel())

        self.assertGreaterEqual(r1, 99.0)
        self.assertEqual(evaluator.latest_metrics['val/top1_row'], 'host-t2i')
        self.assertEqual(evaluator.latest_metrics['val/top1_source_row'], 'host-t2i')
        self.assertEqual(evaluator.latest_authority['candidates'], {'host': 'host-t2i'})

    def test_removed_fusion_config_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, 'fusion'):
            validate_config_data({'fusion': {'enabled': True}})

    def test_removed_retrieval_scorer_flag_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, 'retrieval_scorer'):
            validate_config_data({'evaluation': {'retrieval_scorer': 'approximate'}})


if __name__ == '__main__':
    unittest.main()
