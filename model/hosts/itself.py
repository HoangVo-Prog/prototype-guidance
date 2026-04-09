"""Thin adapter to reuse the original ITSELF implementation directly."""

from __future__ import annotations

import importlib
import sys
import threading
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Tuple


_ADAPTER_ROOT = Path(__file__).resolve().parents[2] / 'adapter' / 'WACV2026-Oral-ITSELF'
_ADAPTER_NAMESPACE = '_itself_original_source'
_IMPORT_LOCK = threading.Lock()
_CACHED_COMPONENTS = None


@dataclass(frozen=True)
class OriginalITSELFComponents:
    model_build: ModuleType
    solver_build: ModuleType
    processor: ModuleType
    metrics: ModuleType


def should_use_original_itself_runtime(args) -> bool:
    return (
        str(getattr(args, 'host_type', 'clip')).lower() == 'itself'
        and not bool(getattr(args, 'use_prototype_branch', False))
    )


def prepare_itself_legacy_args(args):
    """Populate legacy ITSELF argument names expected by the original source."""

    alias_pairs = (
        ('loss_names', 'itself_loss_names'),
        ('only_global', 'itself_only_global'),
        ('select_ratio', 'itself_select_ratio'),
        ('grab_embed_dim', 'itself_grab_embed_dim'),
        ('score_weight_global', 'itself_score_weight_global'),
        ('tau', 'itself_tau'),
        ('margin', 'itself_margin'),
        ('return_all', 'itself_return_all'),
        ('topk_type', 'itself_topk_type'),
        ('layer_index', 'itself_layer_index'),
        ('average_attn_weights', 'itself_average_attn_weights'),
        ('modify_k', 'itself_modify_k'),
    )
    for legacy_name, canonical_name in alias_pairs:
        if hasattr(args, canonical_name):
            setattr(args, legacy_name, getattr(args, canonical_name))
    return args


def _ensure_namespace_package(package_name: str, package_path: Path) -> ModuleType:
    module = sys.modules.get(package_name)
    normalized_path = str(package_path)
    if module is None:
        module = types.ModuleType(package_name)
        module.__path__ = [normalized_path]
        module.__package__ = package_name
        sys.modules[package_name] = module
    else:
        path_list = list(getattr(module, '__path__', []))
        if normalized_path not in path_list:
            path_list.append(normalized_path)
            module.__path__ = path_list

    if '.' in package_name:
        parent_name, child_name = package_name.rsplit('.', 1)
        parent_module = sys.modules.get(parent_name)
        if parent_module is not None and not hasattr(parent_module, child_name):
            setattr(parent_module, child_name, module)
    return module


def _ensure_adapter_namespace() -> None:
    if not _ADAPTER_ROOT.exists():
        raise FileNotFoundError(f'Original ITSELF adapter path not found: {_ADAPTER_ROOT}')

    _ensure_namespace_package(_ADAPTER_NAMESPACE, _ADAPTER_ROOT)
    _ensure_namespace_package(f'{_ADAPTER_NAMESPACE}.model', _ADAPTER_ROOT / 'model')
    _ensure_namespace_package(f'{_ADAPTER_NAMESPACE}.solver', _ADAPTER_ROOT / 'solver')
    _ensure_namespace_package(f'{_ADAPTER_NAMESPACE}.processor', _ADAPTER_ROOT / 'processor')
    _ensure_namespace_package(f'{_ADAPTER_NAMESPACE}.utils', _ADAPTER_ROOT / 'utils')


@contextmanager
def _temporary_module_alias(module_name: str, alias_target: ModuleType):
    had_original = module_name in sys.modules
    original_module = sys.modules.get(module_name)
    original_children = {
        name: module
        for name, module in list(sys.modules.items())
        if name.startswith(f'{module_name}.')
    }
    sys.modules[module_name] = alias_target
    try:
        yield
    finally:
        for name in list(sys.modules.keys()):
            if name.startswith(f'{module_name}.') and name not in original_children:
                sys.modules.pop(name, None)
        for name, module in original_children.items():
            sys.modules[name] = module
        if had_original:
            sys.modules[module_name] = original_module
        else:
            sys.modules.pop(module_name, None)


def _import_with_optional_alias(module_path: str, alias_name: str = '', alias_target: ModuleType | None = None) -> ModuleType:
    if alias_name and alias_target is not None:
        with _temporary_module_alias(alias_name, alias_target):
            return importlib.import_module(module_path)
    return importlib.import_module(module_path)


def get_original_itself_components() -> OriginalITSELFComponents:
    global _CACHED_COMPONENTS
    with _IMPORT_LOCK:
        if _CACHED_COMPONENTS is not None:
            return _CACHED_COMPONENTS

        _ensure_adapter_namespace()

        model_pkg = sys.modules[f'{_ADAPTER_NAMESPACE}.model']
        model_build = _import_with_optional_alias(
            f'{_ADAPTER_NAMESPACE}.model.build',
            alias_name='model',
            alias_target=model_pkg,
        )
        solver_build = importlib.import_module(f'{_ADAPTER_NAMESPACE}.solver.build')
        metrics = importlib.import_module(f'{_ADAPTER_NAMESPACE}.utils.metrics')
        processor = importlib.import_module(f'{_ADAPTER_NAMESPACE}.processor.processor')

        _CACHED_COMPONENTS = OriginalITSELFComponents(
            model_build=model_build,
            solver_build=solver_build,
            processor=processor,
            metrics=metrics,
        )
        return _CACHED_COMPONENTS


def get_original_itself_module_paths() -> Dict[str, str]:
    components = get_original_itself_components()
    return {
        'model_build': str(Path(components.model_build.__file__).resolve()),
        'solver_build': str(Path(components.solver_build.__file__).resolve()),
        'processor': str(Path(components.processor.__file__).resolve()),
        'metrics': str(Path(components.metrics.__file__).resolve()),
    }


def build_itself_host(args, num_classes, **kwargs):
    del kwargs
    prepare_itself_legacy_args(args)
    components = get_original_itself_components()
    return components.model_build.build_model(args, num_classes)


def build_original_itself_optimizer(args, model):
    prepare_itself_legacy_args(args)
    components = get_original_itself_components()
    return components.solver_build.build_optimizer(args, model)


def build_original_itself_lr_scheduler(args, optimizer):
    prepare_itself_legacy_args(args)
    components = get_original_itself_components()
    return components.solver_build.build_lr_scheduler(args, optimizer)


def get_original_itself_training_components(args) -> Tuple[Callable, type]:
    prepare_itself_legacy_args(args)
    components = get_original_itself_components()
    return components.processor.do_train, components.metrics.Evaluator


def get_original_itself_inference_fn(args) -> Callable:
    prepare_itself_legacy_args(args)
    components = get_original_itself_components()
    return components.processor.do_inference
