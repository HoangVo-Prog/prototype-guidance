"""Thin adapter to reuse the original ITSELF implementation directly."""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import types
import time
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from prettytable import PrettyTable


_ADAPTER_ROOT = Path(__file__).resolve().parents[2] / 'adapter' / 'WACV2026-Oral-ITSELF'
_ADAPTER_NAMESPACE = '_itself_original_source'
_IMPORT_LOCK = threading.Lock()
_CACHED_COMPONENTS = None
_STATIC_MIX_EVALUATOR_CACHE = None
_DEFAULT_ITSELF_ABLATION_ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.68, 0.32)


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

        @staticmethod
        def _format_alpha(alpha: float) -> str:
            return f'{alpha:.2f}'.rstrip('0').rstrip('.')

        @staticmethod
        def _to_row_dict(row):
            # Original ITSELF get_metrics returns:
            # [task, R1, R5, R10, mAP, mINP, rSum]
            return {
                'task': str(row[0]),
                'R1': float(row[1]),
                'R5': float(row[2]),
                'R10': float(row[3]),
                'mAP': float(row[4]),
                'mINP': float(row[5]),
                'rSum': float(row[6]),
                'bucket': 'host',
            }

        def _ablation_enabled(self) -> bool:
            return bool(
                str(getattr(self.args, 'host_type', 'itself')).lower() == 'itself'
                and bool(getattr(self.args, 'itself_lambda_ablation_enabled', False))
            )

        def _resolve_alphas(self):
            configured = getattr(self.args, 'itself_lambda_ablation_alphas', None)
            if configured in (None, []):
                alphas = list(_DEFAULT_ITSELF_ABLATION_ALPHAS)
            else:
                alphas = [float(alpha) for alpha in configured]
            if bool(getattr(self.args, 'itself_lambda_ablation_include_default', True)):
                default_alpha = getattr(self.args, 'score_weight_global', None)
                if default_alpha is None:
                    default_alpha = getattr(self.args, 'itself_score_weight_global', None)
                if default_alpha is not None:
                    alphas.append(float(default_alpha))
            deduped = []
            seen = set()
            for alpha in alphas:
                if alpha < 0.0 or alpha > 1.0:
                    continue
                rounded = round(float(alpha), 6)
                if rounded in seen:
                    continue
                seen.add(rounded)
                deduped.append(float(alpha))
            return deduped

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

            start_time = time.time()
            self.logger.info(
                'Static ITSELF evaluation started (alpha_global=%.4f, only_global=%s).',
                alpha,
                bool(getattr(self.args, 'only_global', False)),
            )
            qfeats, gfeats, qids, gids = self._compute_embedding(model)
            qfeats = F.normalize(qfeats, p=2, dim=1)
            gfeats = F.normalize(gfeats, p=2, dim=1)
            sims_global = qfeats @ gfeats.t()

            vq_feats, vg_feats, _, _ = self._compute_embedding_grab(model)
            vq_feats = F.normalize(vq_feats, p=2, dim=1)
            vg_feats = F.normalize(vg_feats, p=2, dim=1)
            sims_grab = vq_feats @ vg_feats.t()

            table = PrettyTable(['task', 'R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum'])
            if self._ablation_enabled():
                rows = [
                    metrics_module.get_metrics(sims_global, qids, gids, 'global-t2i', False),
                    metrics_module.get_metrics(sims_grab, qids, gids, 'grab-t2i', False),
                ]
                for alpha_sweep in self._resolve_alphas():
                    sims_mix = alpha_sweep * sims_global + (1.0 - alpha_sweep) * sims_grab
                    rows.append(
                        metrics_module.get_metrics(
                            sims_mix,
                            qids,
                            gids,
                            f'global+grab({self._format_alpha(alpha_sweep)})-t2i',
                            False,
                        )
                    )
                top1 = 0.0
                top1_row = None
                for row in rows:
                    table.add_row(row)
                    row_r1 = float(row[1])
                    if (top1_row is None) or (row_r1 > top1):
                        top1 = row_r1
                        top1_row = row
            else:
                sims = alpha * sims_global + (1.0 - alpha) * sims_grab
                row = metrics_module.get_metrics(sims, qids, gids, f'global+grab({alpha:.2f})-t2i', False)
                top1 = float(row[1])
                table.add_row(row)
                top1_row = row
            if i2t_metric:
                i2t_similarity = sims if 'sims' in locals() else sims_global
                i2t_cmc, i2t_mAP, i2t_mINP, _ = metrics_module.rank(
                    similarity=i2t_similarity.t(),
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
            if self._ablation_enabled():
                self.logger.info('\n' + f'itself lambda ablation enabled with {len(self._resolve_alphas())} mix settings.')
            else:
                self.logger.info('\n' + f'static global-grab alpha = {alpha:.4f}')
            self.logger.info('\n' + f'best R1 = {top1}')
            self.logger.info('Static ITSELF evaluation finished in %.1fs.', time.time() - start_time)

            structured_rows = [self._to_row_dict(row) for row in rows] if self._ablation_enabled() else [self._to_row_dict(top1_row)]
            top1_task = str(top1_row[0]) if top1_row is not None else 'host-t2i'
            self.latest_eval_rows = [dict(item) for item in structured_rows]
            self.latest_authority = {
                'display_row': top1_task,
                'source_row': top1_task,
                'mismatch': False,
                'selected_source_role': 'host',
                'candidates': {'host': top1_task},
                'row_roles': {str(item['task']): 'host' for item in structured_rows},
                'row_metrics': {
                    str(item['task']): {
                        'R1': float(item['R1']),
                        'R5': float(item['R5']),
                        'R10': float(item['R10']),
                        'mAP': float(item['mAP']),
                        'mINP': float(item['mINP']),
                        'rSum': float(item['rSum']),
                    }
                    for item in structured_rows
                },
            }
            self.latest_metrics = {
                'val/retrieval/R1': float(top1_row[1]) if top1_row is not None else float(top1),
                'val/retrieval/R5': float(top1_row[2]) if top1_row is not None else 0.0,
                'val/retrieval/R10': float(top1_row[3]) if top1_row is not None else 0.0,
                'val/retrieval/mAP': float(top1_row[4]) if top1_row is not None else 0.0,
                'val/retrieval/mINP': float(top1_row[5]) if top1_row is not None else 0.0,
                'val/retrieval/rSum': float(top1_row[6]) if top1_row is not None else 0.0,
                'val/top1': float(top1),
                'val/top1_row': top1_task,
                'val/top1_source_row': top1_task,
                'val/top1_display_row': top1_task,
            }
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


def attach_itself_clip_text_intermediates(base_model) -> None:
    """Attach a thin `encode_text_intermediates` adapter for original ITSELF CLIP."""

    existing = getattr(base_model, 'encode_text_intermediates', None)
    if callable(existing):
        return

    required = ('token_embedding', 'positional_embedding', 'transformer', 'ln_final', 'text_projection')
    missing = [name for name in required if not hasattr(base_model, name)]
    if missing:
        return

    def _encode_text_intermediates(text: torch.Tensor, return_all: bool = False, average_attn_weights: bool = True):
        text = text.long()
        model_dtype: Optional[torch.dtype] = getattr(base_model, 'dtype', None)

        x = base_model.token_embedding(text)
        if model_dtype is not None:
            x = x.type(model_dtype)
        x = x + base_model.positional_embedding[: text.size(1)].type(x.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        if return_all:
            outputs = base_model.transformer.forward([x], return_all=True, average_attn_weights=average_attn_weights)
            x = outputs[0][0]
            attention_weights = outputs[1]
        else:
            outputs = base_model.transformer([x], average_attn_weights=average_attn_weights)
            x = outputs[0]
            attention_weights = outputs[1]

        x = x.permute(1, 0, 2)  # LND -> NLD
        pre_projection_tokens = base_model.ln_final(x).type(x.dtype)
        projected_tokens = pre_projection_tokens @ base_model.text_projection
        return {
            'projected_tokens': projected_tokens,
            'pre_projection_tokens': pre_projection_tokens,
            'attention_weights': attention_weights,
        }

    base_model.encode_text_intermediates = _encode_text_intermediates


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
