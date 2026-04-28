import types
import unittest
from collections import OrderedDict

import torch

from solver.build import build_optimizer


class _TinyOptimizerProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.host_head = torch.nn.Module()
        self.host_head.visual_embedding_layer = torch.nn.Linear(4, 4)
        self.host_head.textual_embedding_layer = torch.nn.Linear(4, 4)
        self.host_head.classifier_global = torch.nn.Linear(4, 4)
        self.host_head.mlp_global = torch.nn.Linear(4, 4)
        self.other_block = torch.nn.Linear(4, 4)

    def named_optimizer_groups(self):
        groups = OrderedDict(
            prototype_bank=[],
            prototype_projectors=[],
            prototype_routing=[],
            prototype_pooling=[],
            prototype_contextualization=[],
            host_projectors=[],
            class_proxies=[],
            image_backbone=[],
            text_backbone=[],
            other=[],
        )
        for name, parameter in self.named_parameters():
            if name.startswith('host_head.'):
                groups['host_projectors'].append((name, parameter))
            else:
                groups['other'].append((name, parameter))
        return groups


def _make_args(host_type: str):
    return types.SimpleNamespace(
        host_type=host_type,
        use_prototype_branch=True,
        optimizer='Adam',
        lr=1e-5,
        weight_decay=4e-5,
        lr_factor=5.0,
        bias_lr_factor=2.0,
        weight_decay_bias=0.0,
        alpha=0.9,
        beta=0.999,
        momentum=0.9,
        optimizer_eps=1e-8,
        lr_host_projectors=1e-5,
    )


class SolverOptimizerITSELFTests(unittest.TestCase):
    def _lr_by_parameter_name(self, optimizer, model):
        id_to_name = {id(parameter): name for name, parameter in model.named_parameters()}
        mapping = {}
        for group in optimizer.param_groups:
            lr = float(group['lr'])
            for parameter in group.get('params', []):
                mapping[id_to_name[id(parameter)]] = lr
        return mapping

    def test_itself_host_projectors_apply_legacy_grab_lr_policy(self):
        model = _TinyOptimizerProbeModel()
        optimizer = build_optimizer(_make_args(host_type='itself'), model)
        lr_by_name = self._lr_by_parameter_name(optimizer, model)

        self.assertAlmostEqual(lr_by_name['host_head.visual_embedding_layer.weight'], 1e-3, places=12)
        self.assertAlmostEqual(lr_by_name['host_head.visual_embedding_layer.bias'], 1e-3, places=12)
        self.assertAlmostEqual(lr_by_name['host_head.textual_embedding_layer.weight'], 1e-3, places=12)
        self.assertAlmostEqual(lr_by_name['host_head.textual_embedding_layer.bias'], 1e-3, places=12)

        self.assertAlmostEqual(lr_by_name['host_head.classifier_global.weight'], 5e-5, places=12)
        self.assertAlmostEqual(lr_by_name['host_head.classifier_global.bias'], 5e-5, places=12)
        self.assertAlmostEqual(lr_by_name['host_head.mlp_global.weight'], 1e-5, places=12)

    def test_non_itself_host_projectors_keep_configured_group_lr(self):
        model = _TinyOptimizerProbeModel()
        optimizer = build_optimizer(_make_args(host_type='clip'), model)
        lr_by_name = self._lr_by_parameter_name(optimizer, model)

        self.assertAlmostEqual(lr_by_name['host_head.visual_embedding_layer.weight'], 1e-5, places=12)
        self.assertAlmostEqual(lr_by_name['host_head.textual_embedding_layer.weight'], 1e-5, places=12)
        self.assertAlmostEqual(lr_by_name['host_head.classifier_global.weight'], 1e-5, places=12)
        self.assertAlmostEqual(lr_by_name['host_head.mlp_global.weight'], 1e-5, places=12)


if __name__ == '__main__':
    unittest.main()
