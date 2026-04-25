import types

import torch

from model import pas_model


class _TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))


class _DummyPASModel(torch.nn.Module):
    def __init__(self, args, num_classes, train_loader=None):
        super().__init__()
        del num_classes, train_loader
        self.backbone_precision = str(getattr(args, 'backbone_precision', 'fp32')).lower()
        self.prototype_precision = str(getattr(args, 'prototype_precision', 'fp32')).lower()
        self.base_model = _TinyModule()
        self.host_head = _TinyModule()
        self.prototype_head = _TinyModule()


def _mock_convert_weights(module):
    module.half()


def test_backbone_precision_controls_host_backbone_and_host_retrieval(monkeypatch):
    monkeypatch.setattr(pas_model, 'PASModel', _DummyPASModel)
    monkeypatch.setattr(pas_model, 'convert_weights', _mock_convert_weights)

    args = types.SimpleNamespace(
        host_type='itself',
        itself_use_original_impl=True,
        backbone_precision='fp32',
        prototype_precision='fp16',
    )
    model = pas_model.build_model(args=args, num_classes=1, train_loader=None)

    assert model.base_model.weight.dtype == torch.float32
    assert model.host_head.weight.dtype == torch.float32
    assert model.prototype_head.weight.dtype == torch.float16


def test_backbone_fp16_applies_to_host_backbone_and_host_retrieval(monkeypatch):
    monkeypatch.setattr(pas_model, 'PASModel', _DummyPASModel)
    monkeypatch.setattr(pas_model, 'convert_weights', _mock_convert_weights)

    args = types.SimpleNamespace(
        host_type='itself',
        itself_use_original_impl=True,
        backbone_precision='fp16',
        prototype_precision='fp32',
    )
    model = pas_model.build_model(args=args, num_classes=1, train_loader=None)

    assert model.base_model.weight.dtype == torch.float16
    assert model.host_head.weight.dtype == torch.float16
    assert model.prototype_head.weight.dtype == torch.float32

