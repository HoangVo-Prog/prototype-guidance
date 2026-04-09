"""Thin adapter to reuse the original ITSELF implementation directly."""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import types
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Tuple

import torch.nn.functional as F
from prettytable import PrettyTable


_ADAPTER_ROOT = Path(__file__).resolve().parents[2] / 'adapter' / 'WACV2026-Oral-ITSELF'
_ADAPTER_NAMESPACE = '_itself_original_source'
_IMPORT_LOCK = threading.Lock()
_CACHED_COMPONENTS = None
_STATIC_MIX_EVALUATOR_CACHE = None


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

    spec = getattr(module, '__spec__', None)
    if spec is None or not isinstance(spec, ModuleSpec):
        spec = ModuleSpec(name=package_name, loader=None, is_package=True)
    spec.submodule_search_locations = list(getattr(module, '__path__', [normalized_path]))
    module.__spec__ = spec

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


def _load_module_from_file(module_name: str, file_path: Path) -> ModuleType:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load module spec for {module_name!r} from {str(file_path)!r}.')

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    if '.' in module_name:
        parent_name, child_name = module_name.rsplit('.', 1)
        parent_module = sys.modules.get(parent_name)
        if parent_module is not None and not hasattr(parent_module, child_name):
            setattr(parent_module, child_name, module)
    return module


def _build_static_mix_evaluator_class(metrics_module: ModuleType):
    global _STATIC_MIX_EVALUATOR_CACHE
    if _STATIC_MIX_EVALUATOR_CACHE is not None:
        return _STATIC_MIX_EVALUATOR_CACHE

    class ITSELFStaticMixEvaluator(metrics_module.Evaluator):
        """Evaluator variant with static global/grab mixing from config."""

        def eval(self, model, i2t_metric=False):
            if bool(getattr(self.args, 'only_global', False)):
                return super().eval(model, i2t_metric=i2t_metric)

            alpha = getattr(self.args, 'score_weight_global', None)
            if alpha is None:
                alpha = getattr(self.args, 'itself_score_weight_global', None)
            if alpha is None:
                return super().eval(model, i2t_metric=i2t_metric)

            alpha = float(alpha)
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError(f'score_weight_global must be in [0, 1], got {alpha}.')

            qfeats, gfeats, qids, gids = self._compute_embedding(model)
            qfeats = F.normalize(qfeats, p=2, dim=1)
            gfeats = F.normalize(gfeats, p=2, dim=1)
            sims_global = qfeats @ gfeats.t()

            vq_feats, vg_feats, _, _ = self._compute_embedding_grab(model)
            vq_feats = F.normalize(vq_feats, p=2, dim=1)
            vg_feats = F.normalize(vg_feats, p=2, dim=1)
            sims_grab = vq_feats @ vg_feats.t()

            sims = alpha * sims_global + (1.0 - alpha) * sims_grab
            row = metrics_module.get_metrics(sims, qids, gids, f'global+grab({alpha:.2f})-t2i', False)
            top1 = float(row[1])

            table = PrettyTable(['task', 'R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum'])
            table.add_row(row)
            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = metrics_module.rank(
                    similarity=sims.t(),
                    q_pids=gids,
                    g_pids=qids,
                    max_rank=10,
                    get_mAP=True,
                )
                i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

            table.custom_format['R1'] = lambda _, value: f'{value:.2f}'
            table.custom_format['R5'] = lambda _, value: f'{value:.2f}'
            table.custom_format['R10'] = lambda _, value: f'{value:.2f}'
            table.custom_format['mAP'] = lambda _, value: f'{value:.2f}'
            table.custom_format['mINP'] = lambda _, value: f'{value:.2f}'
            table.custom_format['RSum'] = lambda _, value: f'{value:.2f}'
            self.logger.info('\n' + str(table))
            self.logger.info('\n' + f'static global-grab alpha = {alpha:.4f}')
            self.logger.info('\n' + f'best R1 = {top1}')
            return top1

    _STATIC_MIX_EVALUATOR_CACHE = ITSELFStaticMixEvaluator
    return _STATIC_MIX_EVALUATOR_CACHE


def get_original_itself_components() -> OriginalITSELFComponents:
    global _CACHED_COMPONENTS
    with _IMPORT_LOCK:
        if _CACHED_COMPONENTS is not None:
            return _CACHED_COMPONENTS

        _ensure_adapter_namespace()

        model_pkg = sys.modules[f'{_ADAPTER_NAMESPACE}.model']
        utils_pkg = sys.modules[f'{_ADAPTER_NAMESPACE}.utils']

        _load_module_from_file(f'{_ADAPTER_NAMESPACE}.model.simple_tokenizer', _ADAPTER_ROOT / 'model' / 'simple_tokenizer.py')
        _load_module_from_file(f'{_ADAPTER_NAMESPACE}.model.clip_model', _ADAPTER_ROOT / 'model' / 'clip_model.py')
        _load_module_from_file(f'{_ADAPTER_NAMESPACE}.model.grab', _ADAPTER_ROOT / 'model' / 'grab.py')
        _load_module_from_file(f'{_ADAPTER_NAMESPACE}.model.objectives', _ADAPTER_ROOT / 'model' / 'objectives.py')
        with _temporary_module_alias('model', model_pkg):
            model_build = _load_module_from_file(f'{_ADAPTER_NAMESPACE}.model.build', _ADAPTER_ROOT / 'model' / 'build.py')

        _load_module_from_file(f'{_ADAPTER_NAMESPACE}.solver.lr_scheduler', _ADAPTER_ROOT / 'solver' / 'lr_scheduler.py')
        solver_build = _load_module_from_file(f'{_ADAPTER_NAMESPACE}.solver.build', _ADAPTER_ROOT / 'solver' / 'build.py')

        _load_module_from_file(f'{_ADAPTER_NAMESPACE}.utils.comm', _ADAPTER_ROOT / 'utils' / 'comm.py')
        _load_module_from_file(f'{_ADAPTER_NAMESPACE}.utils.meter', _ADAPTER_ROOT / 'utils' / 'meter.py')
        metrics = _load_module_from_file(f'{_ADAPTER_NAMESPACE}.utils.metrics', _ADAPTER_ROOT / 'utils' / 'metrics.py')
        with _temporary_module_alias('utils', utils_pkg):
            processor = _load_module_from_file(f'{_ADAPTER_NAMESPACE}.processor.processor', _ADAPTER_ROOT / 'processor' / 'processor.py')

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
    evaluator_class = _build_static_mix_evaluator_class(components.metrics)
    return components.processor.do_train, evaluator_class


def get_original_itself_inference_fn(args) -> Callable:
    prepare_itself_legacy_args(args)
    components = get_original_itself_components()
    evaluator_class = _build_static_mix_evaluator_class(components.metrics)

    def _wrapped_do_inference(model, test_img_loader, test_txt_loader, runtime_args):
        logger = logging.getLogger('ITSELF.test')
        logger.info('Enter inferencing')
        evaluator = evaluator_class(test_img_loader, test_txt_loader, runtime_args)
        _ = evaluator.eval(model.eval())

    return _wrapped_do_inference
