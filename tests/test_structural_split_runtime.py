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


class _ITSELFAblationModel(torch.nn.Module):
    def __init__(self, alpha=0.32):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.alpha = float(alpha)

    def encode_text_for_retrieval(self, caption):
        return {
            'text_projected': caption.float(),
            'global_text_embedding': caption.float(),
            'grab_text_embedding': torch.flip(caption.float(), dims=[1]),
        }

    def encode_image_for_retrieval(self, image):
        return {
            'image_projected': image.float(),
            'global_image_embedding': image.float(),
            'grab_image_embedding': image.float(),
        }

    def compute_retrieval_similarity(self, image_features, text_features):
        global_sim = text_features['global_text_embedding'] @ image_features['global_image_embedding'].t()
        grab_sim = text_features['grab_text_embedding'] @ image_features['grab_image_embedding'].t()
        return (self.alpha * global_sim) + ((1.0 - self.alpha) * grab_sim)


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

    def test_semantic_hardneg_margin_requires_joint_runtime_when_enabled(self):
        with self.assertRaisesRegex(ValueError, 'use_loss_semantic_hardneg_margin'):
            validate_config_data(
                {
                    'model': {
                        'use_prototype_branch': True,
                        'use_prototype_bank': True,
                        'use_image_conditioned_pooling': True,
                        'runtime_mode': 'host_only',
                    },
                    'objectives': {
                        'objectives': {
                            'use_loss_semantic_hardneg_margin': True,
                        },
                    },
                }
            )

    def test_semantic_hosthard_weighted_requires_joint_runtime_when_enabled(self):
        with self.assertRaisesRegex(ValueError, 'use_loss_semantic_hosthard_weighted'):
            validate_config_data(
                {
                    'model': {
                        'use_prototype_branch': True,
                        'use_prototype_bank': True,
                        'use_image_conditioned_pooling': True,
                        'runtime_mode': 'host_only',
                    },
                    'objectives': {
                        'objectives': {
                            'use_loss_semantic_hosthard_weighted': True,
                        },
                    },
                }
            )

    def test_itself_ablation_alphas_out_of_range_fail(self):
        with self.assertRaisesRegex(ValueError, 'itself_lambda_ablation_alphas'):
            validate_config_data({'evaluation': {'itself_lambda_ablation_alphas': [1.2]}})

    def test_early_stopping_config_is_accepted(self):
        validate_config_data(
            {
                'training': {
                    'early_stopping_enabled': True,
                    'early_stopping_metric': 'R1',
                    'early_stopping_mode': 'max',
                    'early_stopping_patience': 3,
                    'early_stopping_min_delta': 0.0,
                    'early_stopping_start_epoch': 1,
                    'early_stopping_monitored_bucket': 'host',
                    'early_stopping_monitored_task_pattern': '*grab*',
                    'early_stopping_stop_on_nan': False,
                },
            }
        )

    def test_early_stopping_invalid_mode_fails(self):
        with self.assertRaisesRegex(ValueError, 'early_stopping_mode'):
            validate_config_data({'training': {'early_stopping_mode': 'largest'}})

    def test_itself_inference_ablation_adds_global_grab_rows(self):
        args = types.SimpleNamespace(
            retrieval_metrics=['R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum'],
            amp=False,
            log_debug_metrics=False,
            host_type='itself',
            training=False,
            itself_lambda_ablation_enabled=True,
            itself_lambda_ablation_alphas=[0.1, 0.9],
            itself_lambda_ablation_include_default=True,
            itself_score_weight_global=0.32,
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
        _ = evaluator.eval(_ITSELFAblationModel(alpha=0.32))

        rows = evaluator.latest_authority['row_roles']
        self.assertIn('host-t2i', rows)
        self.assertIn('global-t2i', rows)
        self.assertIn('grab-t2i', rows)
        self.assertIn('global+grab(0.1)-t2i', rows)
        self.assertIn('global+grab(0.9)-t2i', rows)
        self.assertIn('global+grab(0.32)-t2i', rows)
        self.assertEqual(evaluator.latest_metrics['val/top1_row'], 'host-t2i')


if __name__ == '__main__':
    unittest.main()
