from contextlib import nullcontext
from typing import Optional

import torch


MODULE_PRECISION_ALIASES = {
    'fp16': 'fp16',
    'float16': 'fp16',
    'half': 'fp16',
    'fp32': 'fp32',
    'float32': 'fp32',
    'full': 'fp32',
}

BACKBONE_PRECISION_ALIASES = MODULE_PRECISION_ALIASES
PROTOTYPE_PRECISION_ALIASES = MODULE_PRECISION_ALIASES

AMP_DTYPE_ALIASES = {
    'fp16': 'fp16',
    'float16': 'fp16',
    'half': 'fp16',
    'bf16': 'bf16',
    'bfloat16': 'bf16',
}


def _canonicalize_precision(value: Optional[str], aliases, *, surface_name: str) -> str:
    canonical = aliases.get(str(value or 'fp32').lower())
    if canonical is None:
        raise ValueError(f'Unsupported {surface_name}: {value}')
    return canonical


def canonicalize_backbone_precision(value: Optional[str]) -> str:
    return _canonicalize_precision(value, BACKBONE_PRECISION_ALIASES, surface_name='backbone precision')


def canonicalize_prototype_precision(value: Optional[str]) -> str:
    return _canonicalize_precision(value, PROTOTYPE_PRECISION_ALIASES, surface_name='prototype precision')


def precision_to_torch_dtype(precision: str) -> torch.dtype:
    canonical = _canonicalize_precision(precision, MODULE_PRECISION_ALIASES, surface_name='module precision')
    if canonical == 'fp16':
        return torch.float16
    return torch.float32


def canonicalize_amp_dtype(value: Optional[str]) -> str:
    canonical = AMP_DTYPE_ALIASES.get(str(value or 'fp16').lower())
    if canonical is None:
        raise ValueError(f'Unsupported AMP dtype: {value}')
    return canonical


def amp_dtype_to_torch(amp_dtype: str) -> torch.dtype:
    canonical = canonicalize_amp_dtype(amp_dtype)
    if canonical == 'bf16':
        return torch.bfloat16
    return torch.float16


def is_cuda_device(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == 'cuda'
    return str(device).startswith('cuda')


def is_amp_enabled(args, device) -> bool:
    return bool(getattr(args, 'amp', False)) and is_cuda_device(device)


def build_autocast_context(args, device):
    if not is_amp_enabled(args, device):
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=amp_dtype_to_torch(getattr(args, 'amp_dtype', 'fp16')))


def build_grad_scaler(args, device):
    enabled = is_amp_enabled(args, device) and canonicalize_amp_dtype(getattr(args, 'amp_dtype', 'fp16')) == 'fp16'
    return torch.cuda.amp.GradScaler(enabled=enabled)
