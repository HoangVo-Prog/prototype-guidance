import hashlib
import os
import random
from typing import Any, Iterable, Optional

import numpy as np
import torch


def _cfg_value(cfg, key: str, default: Any = None) -> Any:
    return getattr(cfg, key, default)


def resolve_repro_seed(cfg, default_seed):
    repro_seed = _cfg_value(cfg, 'repro_seed', None)
    if repro_seed in (None, ''):
        return int(default_seed)
    return int(repro_seed)


def set_reproducibility(cfg, seed: int, logger=None):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

    if bool(_cfg_value(cfg, 'repro_disable_tf32', True)):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cudnn_benchmark = bool(_cfg_value(cfg, 'repro_cudnn_benchmark', False))
    cudnn_deterministic = bool(_cfg_value(cfg, 'repro_cudnn_deterministic', True))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic

    if bool(_cfg_value(cfg, 'repro_deterministic_algorithms', False)):
        warn_only = bool(_cfg_value(cfg, 'repro_deterministic_warn_only', True))
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

    cublas_cfg = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if cublas_cfg is None:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        cublas_cfg = ':16:8'
    if logger is not None:
        logger.info(
            'REPRO_DEBUG flags seed=%d cudnn.benchmark=%s cudnn.deterministic=%s tf32_disabled=%s cublas_workspace=%s',
            int(seed),
            cudnn_benchmark,
            cudnn_deterministic,
            bool(_cfg_value(cfg, 'repro_disable_tf32', True)),
            cublas_cfg,
        )


def seed_worker(worker_id: int):
    del worker_id
    worker_seed = int(torch.initial_seed()) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int, device: str = 'cpu'):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _sha256_for_bytes(parts: Iterable[bytes], *, name: str = '') -> str:
    h = hashlib.sha256()
    if name:
        h.update(str(name).encode('utf-8'))
    for part in parts:
        h.update(part)
    return h.hexdigest()[:16]


def tensor_hash(x, *, name: str = '') -> str:
    if not isinstance(x, torch.Tensor):
        return array_hash(x, name=name)
    y = x.detach().cpu().contiguous()
    meta = f'{str(y.dtype)}|{tuple(y.shape)}'.encode('utf-8')
    return _sha256_for_bytes((meta, y.numpy().tobytes()), name=name)


def array_hash(x, *, name: str = '') -> str:
    if isinstance(x, np.ndarray):
        y = np.ascontiguousarray(x)
        meta = f'{str(y.dtype)}|{tuple(y.shape)}'.encode('utf-8')
        return _sha256_for_bytes((meta, y.tobytes()), name=name)
    if isinstance(x, torch.Tensor):
        return tensor_hash(x, name=name)
    arr = np.asarray(list(x) if isinstance(x, (list, tuple)) else [x])
    arr = np.ascontiguousarray(arr)
    meta = f'{str(arr.dtype)}|{tuple(arr.shape)}'.encode('utf-8')
    return _sha256_for_bytes((meta, arr.tobytes()), name=name)

