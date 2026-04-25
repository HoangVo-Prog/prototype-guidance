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
        del args, num_classes, train_loader
        self.backbone_precision = 'fp16'
        self.prototype_precision = 'fp16'
        self.base_model = _TinyModule()
        self.host_head = _TinyModule()
        self.prototype_head = _TinyModule()


def _mock_convert_weights(module):
    module.half()


def test_pas_model_builder_forces_fp16(monkeypatch):
    monkeypatch.setattr(pas_model, 'PASModel', _DummyPASModel)
    monkeypatch.setattr(pas_model, 'convert_weights', _mock_convert_weights)

    args = types.SimpleNamespace(host_type='itself', itself_use_original_impl=True)
    model = pas_model.build_model(args=args, num_classes=1, train_loader=None)

    assert model.base_model.weight.dtype == torch.float16
    assert model.host_head.weight.dtype == torch.float16
    assert model.prototype_head.weight.dtype == torch.float16

