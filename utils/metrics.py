import logging
import re
from typing import Dict, List, Optional, Tuple

from prettytable import PrettyTable
import torch
import torch.nn.functional as F

from utils.precision import build_autocast_context, is_cuda_device
from utils.metric_logging import build_validation_debug_metrics, build_validation_retrieval_metrics


SUPPORTED_RETRIEVAL_METRICS = ('R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum')
SUPPORTED_RETRIEVAL_SCORERS = ('exact', 'approximate')


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        _, indices = torch.topk(similarity, k=max_rank, dim=1, largest=True, sorted=True)
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)
    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.0) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


def get_metrics(similarity, qids, gids, name):
    t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()

    def _cmc_at(rank_index: int) -> float:
        if t2i_cmc.size == 0:
            return 0.0
        clamped_index = min(rank_index, t2i_cmc.shape[0] - 1)
        return float(t2i_cmc[clamped_index])

    r1 = _cmc_at(0)
    r5 = _cmc_at(4)
    r10 = _cmc_at(9)
    return {
        'task': name,
        'R1': r1,
        'R5': r5,
        'R10': r10,
        'mAP': float(t2i_mAP),
        'mINP': float(t2i_mINP),
        'rSum': float(r1 + r5 + r10),
    }


class Evaluator:
    def __init__(self, img_loader, txt_loader, args):
        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.logger = logging.getLogger('pas.eval')
        self.args = args
        self.latest_metrics = {}
        self.latest_authority = {}
        requested_metrics = tuple(getattr(args, 'retrieval_metrics', SUPPORTED_RETRIEVAL_METRICS) or SUPPORTED_RETRIEVAL_METRICS)
        unknown_metrics = sorted(set(requested_metrics) - set(SUPPORTED_RETRIEVAL_METRICS))
        if unknown_metrics:
            raise ValueError(
                f'Unsupported evaluation.retrieval_metrics values: {unknown_metrics}. '
                f'Allowed values: {list(SUPPORTED_RETRIEVAL_METRICS)}'
            )
        self.requested_metrics = requested_metrics
        self.retrieval_scorer = str(getattr(args, 'retrieval_scorer', 'exact')).lower()
        if self.retrieval_scorer not in SUPPORTED_RETRIEVAL_SCORERS:
            raise ValueError(
                f'Unsupported evaluation.retrieval_scorer={self.retrieval_scorer!r}. '
                f'Allowed values: {list(SUPPORTED_RETRIEVAL_SCORERS)}'
            )
        if (not bool(getattr(args, 'use_prototype_branch', False)) or not bool(getattr(args, 'use_prototype_bank', True))) and self.retrieval_scorer == 'approximate':
            self.logger.warning(
                'evaluation.retrieval_scorer=approximate requires an active prototype bank; falling back to exact retrieval scoring.'
            )
            self.retrieval_scorer = 'exact'
        (
            self.default_fusion_lambda_host,
            self.default_fusion_lambda_prototype,
        ) = self._resolve_default_fusion_weights()
        self.eval_fusion_subsets = self._resolve_fusion_eval_subsets()
        self.selection_from_eval_subsets = self._should_select_from_eval_subsets()
        self.eval_subset_selection_row_names = self._resolve_eval_subset_selection_row_names()

    @staticmethod
    def _pair_close(lhs: float, rhs: float, tol: float = 1e-6) -> bool:
        return abs(float(lhs) - float(rhs)) <= tol

    @classmethod
    def _pair_key(cls, lambda_host: float, lambda_prototype: float) -> Tuple[int, int]:
        return (int(round(float(lambda_host) * 1_000_000)), int(round(float(lambda_prototype) * 1_000_000)))

    @classmethod
    def _validate_unit_subset_pair(cls, lambda_host: float, lambda_prototype: float, field_name: str) -> None:
        if lambda_host < 0.0 or lambda_host > 1.0:
            raise ValueError(f'{field_name}.lambda_host must be within [0, 1], got {lambda_host}.')
        if lambda_prototype < 0.0 or lambda_prototype > 1.0:
            raise ValueError(f'{field_name}.lambda_prototype must be within [0, 1], got {lambda_prototype}.')
        pair_sum = lambda_host + lambda_prototype
        if not cls._pair_close(pair_sum, 1.0):
            raise ValueError(f'{field_name}.lambda_host + {field_name}.lambda_prototype must equal 1.0, got {pair_sum}.')

    def _resolve_default_fusion_weights(self) -> Tuple[float, float]:
        lambda_host = getattr(self.args, 'fusion_lambda_host', None)
        lambda_prototype = getattr(self.args, 'fusion_lambda_prototype', None)
        if lambda_host is None and lambda_prototype is None:
            legacy_coefficient = getattr(self.args, 'fusion_coefficient', None)
            if legacy_coefficient is not None:
                return 1.0, float(legacy_coefficient)
            if bool(getattr(self.args, 'use_prototype_branch', False)):
                return 1.0, 1.0
            return 1.0, 0.0
        if lambda_host is None:
            return 1.0 - float(lambda_prototype), float(lambda_prototype)
        if lambda_prototype is None:
            return float(lambda_host), 1.0 - float(lambda_host)
        return float(lambda_host), float(lambda_prototype)

    def _resolve_fusion_eval_subsets(self) -> List[Dict[str, object]]:
        raw_subsets = getattr(self.args, 'fusion_eval_subsets', None)
        if raw_subsets is None:
            return []
        if not isinstance(raw_subsets, list):
            raise ValueError('fusion_eval_subsets must be a list of subset mappings.')
        normalized = []
        for subset_index, subset in enumerate(raw_subsets):
            field_name = f'fusion_eval_subsets[{subset_index}]'
            if not isinstance(subset, dict):
                raise ValueError(f'{field_name} must be a mapping.')
            if 'lambda_host' not in subset or 'lambda_prototype' not in subset:
                raise ValueError(f'{field_name} must include lambda_host and lambda_prototype.')
            lambda_host = float(subset['lambda_host'])
            lambda_prototype = float(subset['lambda_prototype'])
            self._validate_unit_subset_pair(lambda_host, lambda_prototype, field_name=field_name)
            subset_name = subset.get('name')
            if subset_name is not None and not isinstance(subset_name, str):
                raise ValueError(f'{field_name}.name must be a string when provided.')
            normalized.append(
                {
                    'name': subset_name.strip() if isinstance(subset_name, str) else None,
                    'lambda_host': lambda_host,
                    'lambda_prototype': lambda_prototype,
                }
            )
        return normalized

    @staticmethod
    def _metric_slug(value: str) -> str:
        normalized = re.sub(r'[^a-zA-Z0-9]+', '_', str(value).strip().lower()).strip('_')
        return normalized or 'row'

    @classmethod
    def _is_host_only_selection_row(cls, task_name: object) -> bool:
        label = str(task_name or '').strip().lower()
        if not label:
            return False
        # Exclude host-only labels (e.g., "host-t2i", "host_only-t2i", "host-t2i out").
        if re.match(r'^host(?:[-_\s]*only)?(?:[-_\s]*t2i)?(?:\b|[-_\s].*)$', label):
            return True
        # Exclude explicit fusion labels that reduce to host-only behavior.
        compact = re.sub(r'\s+', '', label)
        if re.match(
            r'^host\([0-9]*\.?[0-9]+\)\+prototype\((?:0|0\.0+)\)-t2i(?:[-_a-z0-9]*)$',
            compact,
        ):
            return True
        return False

    @classmethod
    def _is_pas_selection_row(cls, task_name: object) -> bool:
        label = str(task_name or '').strip().lower()
        if not label:
            return False
        return bool(re.match(r'^pas[-_\s]*t2i(?:\b|[-_\s].*)?$', label))

    def _should_select_from_eval_subsets(self) -> bool:
        config_data = getattr(self.args, 'config_data', None)
        if not isinstance(config_data, dict):
            return False
        fusion_cfg = config_data.get('fusion')
        if not isinstance(fusion_cfg, dict):
            return False
        has_explicit_lambda = ('lambda_host' in fusion_cfg) or ('lambda_prototype' in fusion_cfg)
        return (not has_explicit_lambda) and bool(self.eval_fusion_subsets)

    def _resolve_eval_subset_selection_row_names(self) -> set:
        names = set()
        for spec in self.eval_fusion_subsets:
            lambda_host = float(spec['lambda_host'])
            lambda_prototype = float(spec['lambda_prototype'])
            row_name = spec.get('name') or self._format_fusion_row_name(lambda_host, lambda_prototype)
            row_name = str(row_name).strip() or self._format_fusion_row_name(lambda_host, lambda_prototype)
            if not row_name.endswith('-t2i'):
                row_name = f'{row_name}-t2i'
            names.add(row_name)
        return names

    @classmethod
    def _authority_role_from_weights(cls, lambda_host: float, lambda_prototype: float) -> str:
        if cls._pair_close(lambda_host, 1.0) and cls._pair_close(lambda_prototype, 0.0):
            return 'host'
        if cls._pair_close(lambda_host, 0.0) and cls._pair_close(lambda_prototype, 1.0):
            return 'prototype'
        return 'fused'

    @staticmethod
    def _canonical_row_name(task_name: object) -> str:
        return str(task_name or '').strip()

    @classmethod
    def _authority_role_from_row_name(cls, row_name: object) -> Optional[str]:
        label = cls._canonical_row_name(row_name).lower()
        if not label:
            return None
        if cls._is_host_only_selection_row(label):
            return 'host'
        if re.match(r'^prototype(?:[-_\s]*t2i)?(?:\b|[-_\s].*)?$', label):
            return 'prototype'
        if cls._is_pas_selection_row(label):
            return 'fused'
        if 'prototype' in label and 'host' in label:
            return 'fused'
        return None

    @classmethod
    def _build_authority_candidates(
        cls,
        metrics_rows: List[Dict[str, object]],
        row_metadata: Dict[str, Dict[str, object]],
    ) -> Dict[str, Optional[str]]:
        candidates: Dict[str, Optional[str]] = {
            'host': None,
            'prototype': None,
            'fused': None,
        }
        best_scores: Dict[str, float] = {
            'host': float('-inf'),
            'prototype': float('-inf'),
            'fused': float('-inf'),
        }
        for row in metrics_rows:
            row_name = cls._canonical_row_name(row.get('task'))
            role = str(row_metadata.get(row_name, {}).get('authority_role', '')).strip().lower()
            if role not in candidates:
                role = cls._authority_role_from_row_name(row_name)
            if role not in candidates:
                continue
            score = float(row.get('R1', float('-inf')))
            if score > best_scores[role]:
                best_scores[role] = score
                candidates[role] = row_name
        return candidates

    @classmethod
    def build_authority_context(
        cls,
        metrics_rows: List[Dict[str, object]],
        row_metadata: Dict[str, Dict[str, object]],
        selected_display_row: Optional[str],
        selected_source_row: Optional[str],
    ) -> Dict[str, object]:
        row_metrics: Dict[str, Dict[str, float]] = {}
        normalized_row_metadata: Dict[str, Dict[str, object]] = {}
        for row in metrics_rows:
            row_name = cls._canonical_row_name(row.get('task'))
            row_metrics[row_name] = {
                metric_name: float(row.get(metric_name, 0.0))
                for metric_name in SUPPORTED_RETRIEVAL_METRICS
                if metric_name in row
            }
            metadata = dict(row_metadata.get(row_name, {}))
            authority_role = str(metadata.get('authority_role', '')).strip().lower()
            if authority_role not in {'host', 'prototype', 'fused'}:
                inferred_role = cls._authority_role_from_row_name(row_name)
                if inferred_role is not None:
                    authority_role = inferred_role
            if authority_role:
                metadata['authority_role'] = authority_role
            normalized_row_metadata[row_name] = metadata

        candidates = cls._build_authority_candidates(metrics_rows, normalized_row_metadata)
        source_row = cls._canonical_row_name(selected_source_row)
        display_row = cls._canonical_row_name(selected_display_row)
        source_role = str(normalized_row_metadata.get(source_row, {}).get('authority_role', '')).strip().lower()
        if source_role not in {'host', 'prototype', 'fused'}:
            inferred_source_role = cls._authority_role_from_row_name(source_row)
            source_role = inferred_source_role if inferred_source_role is not None else ''

        return {
            'display_row': display_row or None,
            'source_row': source_row or None,
            'mismatch': bool(display_row and source_row and display_row != source_row),
            'selected_source_role': source_role or None,
            'candidates': candidates,
            'row_roles': {
                row_name: str(metadata.get('authority_role', '')).strip().lower()
                for row_name, metadata in normalized_row_metadata.items()
                if str(metadata.get('authority_role', '')).strip().lower()
            },
            'row_metadata': normalized_row_metadata,
            'row_metrics': row_metrics,
        }

    @classmethod
    def _format_fusion_row_name(cls, lambda_host: float, lambda_prototype: float) -> str:
        if cls._pair_close(lambda_host, 1.0) and cls._pair_close(lambda_prototype, 0.0):
            return 'host-t2i'
        if cls._pair_close(lambda_host, 0.0) and cls._pair_close(lambda_prototype, 1.0):
            return 'prototype-t2i'
        if cls._pair_close(lambda_host + lambda_prototype, 1.0):
            return f'host+prototype({lambda_prototype:.2f})-t2i'
        return f'host({lambda_host:.2f})+prototype({lambda_prototype:.2f})-t2i'

    def _concat_feature_batches(self, batches):
        first = batches[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(batches, 0)
        if isinstance(first, dict):
            return {key: self._concat_feature_batches([batch[key] for batch in batches]) for key in first.keys()}
        if isinstance(first, (list, tuple)):
            return type(first)(self._concat_feature_batches(list(items)) for items in zip(*batches))
        return first

    def _feature_batch_size(self, features):
        if isinstance(features, torch.Tensor):
            if features.ndim == 0:
                raise ValueError('Feature tensors must have a batch dimension.')
            return int(features.size(0))
        if isinstance(features, dict):
            if not features:
                raise ValueError('Feature dictionaries must not be empty.')
            for value in features.values():
                try:
                    return self._feature_batch_size(value)
                except (TypeError, ValueError):
                    continue
            raise ValueError('Feature dictionaries must contain at least one batched tensor value.')
        if isinstance(features, (list, tuple)) and features:
            return self._feature_batch_size(features[0])
        raise TypeError(f'Unsupported feature container type: {type(features)}')

    def _feature_to_cpu(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: self._feature_to_cpu(sub_value) for key, sub_value in value.items()}
        if isinstance(value, list):
            return [self._feature_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._feature_to_cpu(item) for item in value)
        return value

    def _feature_to_device(self, value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: self._feature_to_device(sub_value, device) for key, sub_value in value.items()}
        if isinstance(value, list):
            return [self._feature_to_device(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(self._feature_to_device(item, device) for item in value)
        return value

    def _positive_gallery_structure(self, text_ids: torch.Tensor, image_ids: torch.Tensor):
        positive_mask = text_ids.view(-1, 1).eq(image_ids.view(1, -1))
        positive_counts = positive_mask.sum(dim=1)
        if not positive_mask.any(dim=1).all():
            missing = (~positive_mask.any(dim=1)).nonzero(as_tuple=False).view(-1).tolist()
            preview = missing[:10]
            suffix = '' if len(missing) <= 10 else '...'
            raise ValueError(
                'Each text query must have at least one positive gallery image. '
                f'Missing positives for query indices {preview}{suffix}.'
            )
        first_positive = positive_mask.to(dtype=torch.int64).argmax(dim=1)
        return positive_mask, positive_counts, first_positive

    def _compute_eval_debug_metrics(self, model, similarity, text_ids, image_ids, image_features, text_features):
        if not bool(getattr(self.args, 'log_debug_metrics', True)):
            return {}
        metrics = {}
        core_model = model.module if hasattr(model, 'module') else model
        similarity = similarity.detach().float().cpu()
        positive_mask, positive_counts, first_positive = self._positive_gallery_structure(text_ids, image_ids)
        metrics['eval_positive_gallery_count_min'] = float(positive_counts.min().item())
        metrics['eval_positive_gallery_count_mean'] = float(positive_counts.float().mean().item())

        logit_scale_value = None
        if hasattr(core_model, 'prototype_head') and hasattr(core_model.prototype_head, 'losses'):
            logit_scale = core_model.prototype_head.losses.get_logit_scale().detach().float().cpu()
            retrieval_temperature = core_model.prototype_head.losses.get_retrieval_temperature().detach().float().cpu()
            logit_scale_value = float(logit_scale.item())
            metrics['eval_logit_scale'] = logit_scale_value
            metrics['eval_retrieval_temperature'] = float(retrieval_temperature.item())

        cosine_similarity = similarity
        if logit_scale_value is not None and logit_scale_value > 0.0:
            cosine_similarity = similarity / logit_scale_value

        positive_scores = cosine_similarity.gather(1, first_positive.view(-1, 1)).squeeze(1)
        negative_mask = ~positive_mask
        if negative_mask.any(dim=1).all():
            hardest_negative = cosine_similarity.masked_fill(~negative_mask, float('-inf')).max(dim=1).values
        else:
            hardest_negative = torch.zeros_like(positive_scores)
        metrics['eval_positive_exact_cosine_mean'] = float(positive_scores.mean().item())
        metrics['eval_hardest_negative_exact_cosine_mean'] = float(hardest_negative.mean().item())
        metrics['eval_exact_margin_mean'] = float((positive_scores - hardest_negative).mean().item())

        image_projected = None
        if isinstance(image_features, dict):
            image_projected = image_features.get('prototype_image_projected', image_features.get('image_projected'))
        if isinstance(image_projected, torch.Tensor):
            image_norms = image_projected.detach().float().norm(dim=-1).cpu()
            metrics['eval_image_projected_norm_mean'] = float(image_norms.mean().item())
            metrics['eval_image_projected_norm_std'] = float(image_norms.std(unbiased=False).item())

        if self.retrieval_scorer == 'exact' and isinstance(image_projected, torch.Tensor):
            required_text_keys = ('text_token_states', 'token_ids')
            if all(isinstance(text_features.get(key), torch.Tensor) for key in required_text_keys):
                with torch.no_grad():
                    positive_indices_device = first_positive.to(device=image_projected.device, dtype=torch.long)
                    positive_summaries = image_features['summary'].index_select(0, positive_indices_device)
                    positive_image_projected = image_projected.index_select(0, positive_indices_device)
                    exact_outputs = core_model.prototype_head.pool_text_with_summary(
                        positive_summaries,
                        text_features['text_token_states'],
                        text_features['token_ids'],
                        attention_mask=text_features.get('attention_mask'),
                        special_token_positions=text_features.get('special_token_positions'),
                        return_debug=False,
                    )
                exact_raw_norms = exact_outputs['text_projected_raw'].detach().float().norm(dim=-1).cpu()
                exact_unit_norms = exact_outputs['text_projected'].detach().float().norm(dim=-1).cpu()
                paired_cosine = (
                    F.normalize(positive_image_projected.detach().float(), dim=-1)
                    * F.normalize(exact_outputs['text_projected'].detach().float(), dim=-1)
                ).sum(dim=-1).cpu()
                metrics['eval_positive_exact_text_embed_norm_mean'] = float(exact_raw_norms.mean().item())
                metrics['eval_positive_exact_text_embed_norm_std'] = float(exact_raw_norms.std(unbiased=False).item())
                metrics['eval_positive_exact_text_embed_unit_norm_mean'] = float(exact_unit_norms.mean().item())
                metrics['eval_positive_exact_pair_cosine_mean'] = float(paired_cosine.mean().item())

        return build_validation_debug_metrics(metrics)

    def _check_similarity_matrix(self, similarity: torch.Tensor, expected_shape: Tuple[int, int], field_name: str) -> torch.Tensor:
        if not isinstance(similarity, torch.Tensor):
            raise TypeError(f'{field_name} must be a tensor.')
        if tuple(similarity.shape) != expected_shape:
            raise ValueError(
                f'{field_name} has shape {tuple(similarity.shape)} but expected {expected_shape} '
                'from concatenated text/image ordering.'
            )
        if not torch.isfinite(similarity).all():
            raise FloatingPointError(f'{field_name} contains NaN or Inf values after evaluation.')
        return similarity.float().cpu()

    def _fuse_from_components(
        self,
        model,
        host_similarity: torch.Tensor,
        prototype_similarity: Optional[torch.Tensor],
        lambda_host: float,
        lambda_prototype: float,
    ) -> torch.Tensor:
        core_model = model.module if hasattr(model, 'module') else model
        if hasattr(core_model, 'fuse_retrieval_similarity'):
            return core_model.fuse_retrieval_similarity(
                host_similarity=host_similarity,
                prototype_similarity=prototype_similarity,
                lambda_host=lambda_host,
                lambda_prototype=lambda_prototype,
            ).float()

        if prototype_similarity is None:
            if abs(lambda_prototype) > 1e-12:
                raise RuntimeError(
                    'prototype_similarity is unavailable for this model, but lambda_prototype is non-zero '
                    f'({lambda_prototype}).'
                )
            return (lambda_host * host_similarity).float()

        if host_similarity.shape != prototype_similarity.shape:
            raise ValueError('Host and prototype similarities must have identical shapes for fusion sweep.')
        return ((lambda_host * host_similarity) + (lambda_prototype * prototype_similarity)).float()

    def _build_similarity_rows(
        self,
        model,
        host_similarity: Optional[torch.Tensor],
        prototype_similarity: Optional[torch.Tensor],
        default_similarity: torch.Tensor,
    ) -> Tuple[List[Tuple[str, torch.Tensor]], Dict[str, Dict[str, object]]]:
        rows: List[Tuple[str, torch.Tensor]] = [('pas-t2i', default_similarity.float().cpu())]
        row_metadata: Dict[str, Dict[str, object]] = {
            'pas-t2i': {
                'lambda_host': float(self.default_fusion_lambda_host),
                'lambda_prototype': float(self.default_fusion_lambda_prototype),
                'authority_role': self._authority_role_from_weights(
                    self.default_fusion_lambda_host,
                    self.default_fusion_lambda_prototype,
                ),
                'source': 'default_similarity',
            }
        }
        if not isinstance(host_similarity, torch.Tensor):
            return rows, row_metadata

        host_similarity = host_similarity.float().cpu()
        prototype_similarity = prototype_similarity.float().cpu() if isinstance(prototype_similarity, torch.Tensor) else None

        candidate_specs: List[Dict[str, object]] = [
            {'name': 'host-t2i', 'lambda_host': 1.0, 'lambda_prototype': 0.0},
        ]
        if prototype_similarity is not None:
            candidate_specs.append({'name': 'prototype-t2i', 'lambda_host': 0.0, 'lambda_prototype': 1.0})
        candidate_specs.extend(self.eval_fusion_subsets)
        candidate_specs.append(
            {
                'name': None,
                'lambda_host': self.default_fusion_lambda_host,
                'lambda_prototype': self.default_fusion_lambda_prototype,
            }
        )

        emitted_names = {'pas-t2i'}
        emitted_pairs = set()
        for spec in candidate_specs:
            lambda_host = float(spec['lambda_host'])
            lambda_prototype = float(spec['lambda_prototype'])
            pair_key = self._pair_key(lambda_host, lambda_prototype)
            if pair_key in emitted_pairs:
                continue
            if prototype_similarity is None and abs(lambda_prototype) > 1e-12:
                self.logger.warning(
                    'Skipping fusion subset (lambda_host=%.4f, lambda_prototype=%.4f): prototype similarity is unavailable.',
                    lambda_host,
                    lambda_prototype,
                )
                continue

            similarity = self._fuse_from_components(
                model=model,
                host_similarity=host_similarity,
                prototype_similarity=prototype_similarity,
                lambda_host=lambda_host,
                lambda_prototype=lambda_prototype,
            ).cpu()
            row_name = spec.get('name') or self._format_fusion_row_name(lambda_host, lambda_prototype)
            row_name = str(row_name).strip() or self._format_fusion_row_name(lambda_host, lambda_prototype)
            if not row_name.endswith('-t2i'):
                row_name = f'{row_name}-t2i'
            if row_name in emitted_names:
                row_name = self._format_fusion_row_name(lambda_host, lambda_prototype)
            if row_name in emitted_names:
                row_name = f'{row_name}-{len(emitted_names)}'
            rows.append((row_name, similarity))
            row_metadata[row_name] = {
                'lambda_host': float(lambda_host),
                'lambda_prototype': float(lambda_prototype),
                'authority_role': self._authority_role_from_weights(lambda_host, lambda_prototype),
                'source': 'candidate_spec',
                'name_specified': bool(spec.get('name')),
            }
            emitted_pairs.add(pair_key)
            emitted_names.add(row_name)
        return rows, row_metadata

    def _compute_similarity(self, model):
        model = model.eval()
        device = next(model.parameters()).device
        if bool(getattr(self.args, 'amp', False)) and not is_cuda_device(device):
            raise ValueError('training.amp=true requires a CUDA device.')

        text_ids, image_ids = [], []
        text_batches, image_batches = [], []

        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                with build_autocast_context(self.args, device):
                    if self.retrieval_scorer == 'approximate':
                        text_features = model.encode_text_basis_for_retrieval(caption)
                    else:
                        text_features = model.encode_text_for_retrieval(caption)
            text_ids.append(pid.view(-1))
            text_batches.append(self._feature_to_cpu(text_features))

        for pid, image in self.img_loader:
            image = image.to(device)
            with torch.no_grad():
                with build_autocast_context(self.args, device):
                    image_features = model.encode_image_for_retrieval(image)
            image_ids.append(pid.view(-1))
            image_batches.append(self._feature_to_cpu(image_features))

        text_ids = torch.cat(text_ids, 0).cpu()
        image_ids = torch.cat(image_ids, 0).cpu()
        text_features = self._concat_feature_batches(text_batches)
        image_features = self._concat_feature_batches(image_batches)

        if self._feature_batch_size(text_features) != int(text_ids.numel()):
            raise ValueError('Text feature concatenation produced a batch size that does not match text_ids ordering.')
        if self._feature_batch_size(image_features) != int(image_ids.numel()):
            raise ValueError('Image feature concatenation produced a batch size that does not match image_ids ordering.')

        text_features = self._feature_to_device(text_features, device)
        image_features = self._feature_to_device(image_features, device)

        core_model = model.module if hasattr(model, 'module') else model
        host_similarity = None
        prototype_similarity = None
        with torch.no_grad():
            with build_autocast_context(self.args, device):
                if self.retrieval_scorer == 'approximate':
                    if hasattr(core_model, 'compute_approximate_retrieval_similarity_components'):
                        host_similarity, prototype_similarity = core_model.compute_approximate_retrieval_similarity_components(
                            image_features,
                            text_features,
                        )
                        similarity = self._fuse_from_components(
                            model=model,
                            host_similarity=host_similarity,
                            prototype_similarity=prototype_similarity,
                            lambda_host=self.default_fusion_lambda_host,
                            lambda_prototype=self.default_fusion_lambda_prototype,
                        )
                    else:
                        similarity = model.compute_approximate_retrieval_similarity(image_features, text_features)
                else:
                    if hasattr(core_model, 'compute_retrieval_similarity_components'):
                        host_similarity, prototype_similarity = core_model.compute_retrieval_similarity_components(
                            image_features,
                            text_features,
                        )
                        similarity = self._fuse_from_components(
                            model=model,
                            host_similarity=host_similarity,
                            prototype_similarity=prototype_similarity,
                            lambda_host=self.default_fusion_lambda_host,
                            lambda_prototype=self.default_fusion_lambda_prototype,
                        )
                    else:
                        similarity = model.compute_retrieval_similarity(image_features, text_features)

        expected_shape = (int(text_ids.numel()), int(image_ids.numel()))
        similarity = self._check_similarity_matrix(similarity, expected_shape, field_name='Retrieval similarity')
        if isinstance(host_similarity, torch.Tensor):
            host_similarity = self._check_similarity_matrix(host_similarity, expected_shape, field_name='Host retrieval similarity')
        if isinstance(prototype_similarity, torch.Tensor):
            prototype_similarity = self._check_similarity_matrix(prototype_similarity, expected_shape, field_name='Prototype retrieval similarity')

        similarity_rows, row_metadata = self._build_similarity_rows(
            model=model,
            host_similarity=host_similarity,
            prototype_similarity=prototype_similarity,
            default_similarity=similarity,
        )
        debug_metrics = self._compute_eval_debug_metrics(model, similarity, text_ids, image_ids, image_features, text_features)
        return similarity_rows, text_ids, image_ids, debug_metrics, row_metadata

    def eval(self, model):
        similarity_rows, text_ids, image_ids, debug_metrics, row_metadata = self._compute_similarity(model)
        metrics_rows = [
            get_metrics(similarity=row_similarity, qids=text_ids, gids=image_ids, name=row_name)
            for row_name, row_similarity in similarity_rows
        ]
        if not metrics_rows:
            raise RuntimeError('Evaluation produced no similarity rows.')
        selected_source_row = None
        if self.selection_from_eval_subsets and self.eval_subset_selection_row_names:
            subset_rows = [
                row
                for row in metrics_rows
                if str(row.get('task', '')).strip() in self.eval_subset_selection_row_names
                and not self._is_host_only_selection_row(row.get('task', ''))
            ]
            if subset_rows:
                # When default fusion lambdas are omitted from config, select checkpoint/current-R1 from eval_subsets.
                metrics = max(subset_rows, key=lambda row: float(row['R1']))
                selected_source_row = str(metrics.get('task', '')).strip()
            else:
                eligible_rows = [row for row in metrics_rows if not self._is_host_only_selection_row(row.get('task', ''))]
                if not eligible_rows:
                    eligible_rows = metrics_rows
                preferred_pas_rows = [row for row in eligible_rows if self._is_pas_selection_row(row.get('task', ''))]
                if preferred_pas_rows:
                    metrics = max(preferred_pas_rows, key=lambda row: float(row['R1']))
                else:
                    metrics = max(eligible_rows, key=lambda row: float(row['R1']))
                selected_source_row = str(metrics.get('task', '')).strip()
        else:
            # Select checkpoint/current-R1 metric row by:
            # 1) excluding host-only rows,
            # 2) preferring pas-t2i when available,
            # 3) otherwise taking the best remaining row by R1.
            eligible_rows = [row for row in metrics_rows if not self._is_host_only_selection_row(row.get('task', ''))]
            if not eligible_rows:
                eligible_rows = metrics_rows
            preferred_pas_rows = [row for row in eligible_rows if self._is_pas_selection_row(row.get('task', ''))]
            if preferred_pas_rows:
                metrics = max(preferred_pas_rows, key=lambda row: float(row['R1']))
            else:
                metrics = max(eligible_rows, key=lambda row: float(row['R1']))
            selected_source_row = str(metrics.get('task', '')).strip()

        selected_display_row = metrics['task']
        if self.selection_from_eval_subsets and self.eval_subset_selection_row_names:
            if any(self._is_pas_selection_row(row.get('task', '')) for row in metrics_rows):
                selected_display_row = 'pas-t2i'

        authority_context = self.build_authority_context(
            metrics_rows=metrics_rows,
            row_metadata=row_metadata,
            selected_display_row=selected_display_row,
            selected_source_row=selected_source_row,
        )
        if authority_context.get('mismatch'):
            self.logger.warning(
                'Row provenance mismatch detected: display_row=%s source_row=%s selected_source_role=%s',
                authority_context.get('display_row'),
                authority_context.get('source_row'),
                authority_context.get('selected_source_role'),
            )
        self.latest_authority = authority_context

        table = PrettyTable(['task'] + list(self.requested_metrics))
        for row_metrics in metrics_rows:
            table.add_row([row_metrics['task']] + [row_metrics[metric_name] for metric_name in self.requested_metrics])
        for metric_name in self.requested_metrics:
            table.custom_format[metric_name] = lambda _, value: f'{value:.2f}'

        retrieval_metrics = {
            metric_name: metrics[metric_name]
            for metric_name in self.requested_metrics
        }
        self.latest_metrics = build_validation_retrieval_metrics(retrieval_metrics)
        self.latest_metrics['val/top1'] = metrics['R1']
        self.latest_metrics['val/top1_row'] = selected_display_row
        self.latest_metrics['val/top1_source_row'] = selected_source_row
        self.latest_metrics['val/top1_display_row'] = selected_display_row
        self.latest_metrics['val/authority/selected_source_role'] = authority_context.get('selected_source_role')
        self.latest_metrics['val/authority/selected_authority_row'] = authority_context.get('source_row')
        self.latest_metrics['val/authority/display_source_mismatch'] = float(bool(authority_context.get('mismatch')))
        candidates = authority_context.get('candidates', {}) if isinstance(authority_context, dict) else {}
        self.latest_metrics['val/authority/host_candidate_row'] = candidates.get('host')
        self.latest_metrics['val/authority/prototype_candidate_row'] = candidates.get('prototype')
        self.latest_metrics['val/authority/fused_candidate_row'] = candidates.get('fused')
        for row_metrics in metrics_rows:
            row_slug = self._metric_slug(row_metrics['task'])
            for metric_name in self.requested_metrics:
                self.latest_metrics[f'val/retrieval_sweep/{row_slug}/{metric_name}'] = float(row_metrics[metric_name])
        self.latest_metrics.update(debug_metrics)

        self.logger.info('\n' + str(table))
        if debug_metrics:
            positive_cos = debug_metrics.get('val/geometry/exact_positive_cosine_mean')
            hardest_negative = debug_metrics.get('val/geometry/exact_hardest_negative_cosine_mean')
            margin = debug_metrics.get('val/geometry/exact_margin_mean')
            if positive_cos is not None and hardest_negative is not None and margin is not None:
                self.logger.info(
                    'Retrieval sanity: positive_exact_cos=%.4f hardest_negative_exact_cos=%.4f margin=%.4f',
                    positive_cos,
                    hardest_negative,
                    margin,
                )
        self.logger.info('\ncurrent R1 = %s (%s)', str(metrics['R1']), selected_display_row)
        if selected_source_row and selected_source_row != selected_display_row:
            self.logger.info('current R1 source row = %s', selected_source_row)
        return metrics['R1']








