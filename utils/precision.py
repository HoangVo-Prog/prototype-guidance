from contextlib import nullcontext

import torch


def canonicalize_backbone_precision(value=None) -> str:
    del value
    return 'fp16'


def canonicalize_prototype_precision(value=None) -> str:
    del value
    return 'fp16'


def precision_to_torch_dtype(precision=None) -> torch.dtype:
    del precision
    return torch.float16


def canonicalize_amp_dtype(value=None) -> str:
    del value
    return 'fp16'


def is_cuda_device(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == 'cuda'
    return str(device).startswith('cuda')


def is_amp_enabled(args=None, device=None) -> bool:
    del args
    return is_cuda_device(device)


def build_autocast_context(args, device):
    del args, device
    # Parameters are already forced to fp16; keep execution path stable by disabling autocast.
    return nullcontext()


def build_grad_scaler(args, device):
    del args, device
    # FP16 parameters are the canonical runtime; GradScaler unscale on FP16 params is invalid.
    return torch.cuda.amp.GradScaler(enabled=False)
