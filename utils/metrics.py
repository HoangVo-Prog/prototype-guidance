import fnmatch
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from prettytable import PrettyTable
import torch
import torch.nn.functional as F

from utils.precision import build_autocast_context
from utils.metric_logging import build_validation_debug_metrics, build_validation_retrieval_metrics


SUPPORTED_RETRIEVAL_METRICS = ('R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum')
BEST_ROW_TASK_NAME = 'best-row-t2i'
DEFAULT_ITSELF_ABLATION_ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.68, 0.32)


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True, stable_tiebreak: bool = False):
    if get_mAP:
        if stable_tiebreak:
            score = similarity.float()
            gids = g_pids.to(device=score.device, dtype=score.dtype).view(1, -1)
            tie_eps = torch.finfo(score.dtype).eps * 8.0
            stable_score = score - (gids * tie_eps)
            indices = torch.argsort(stable_score, dim=1, descending=True, stable=True)
        else:
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


def get_metrics(similarity, qids, gids, name, stable_tiebreak: bool = False):
    t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(
        similarity=similarity,
        q_pids=qids,
        g_pids=gids,
        max_rank=10,
        get_mAP=True,
        stable_tiebreak=stable_tiebreak,
    )
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


def _match_task_pattern(task_name: str, pattern: Optional[str]) -> bool:
    if not pattern:
        return True
    normalized_name = str(task_name or '')
    normalized_pattern = str(pattern).strip()
    if not normalized_pattern:
        return True
    lowered_name = normalized_name.lower()
    lowered_pattern = normalized_pattern.lower()
    if any(char in lowered_pattern for char in ('*', '?', '[')):
        return fnmatch.fnmatchcase(lowered_name, lowered_pattern)
    return lowered_name == lowered_pattern or lowered_pattern in lowered_name


def _extract_rows_and_roles(eval_result: Any) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    row_roles: Dict[str, str] = {}
    rows: List[Dict[str, Any]] = []
    authority_context = {}

    if hasattr(eval_result, 'latest_authority'):
        authority_context = getattr(eval_result, 'latest_authority') or {}
    elif isinstance(eval_result, dict):
        authority_context = eval_result.get('authority') or {}
    if isinstance(authority_context, dict):
        raw_roles = authority_context.get('row_roles', {})
        if isinstance(raw_roles, dict):
            row_roles = {str(name): str(role) for name, role in raw_roles.items()}

    if hasattr(eval_result, 'latest_eval_rows'):
        latest_rows = getattr(eval_result, 'latest_eval_rows') or []
        if isinstance(latest_rows, Sequence):
            rows = [dict(row) for row in latest_rows if isinstance(row, dict)]
    elif isinstance(eval_result, dict):
        if isinstance(eval_result.get('rows'), Sequence):
            rows = [dict(row) for row in eval_result.get('rows', []) if isinstance(row, dict)]
        elif isinstance(eval_result.get('row_metrics'), dict):
            rows = [
                {
                    'task': str(task_name),
                    **{
                        str(metric_name): metric_value
                        for metric_name, metric_value in metric_values.items()
                    },
                }
                for task_name, metric_values in eval_result['row_metrics'].items()
                if isinstance(metric_values, dict)
            ]

    return rows, row_roles


def collect_monitored_eval_rows(
    eval_result: Any,
    monitored_bucket: Optional[str] = 'host',
    monitored_task_pattern: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows, row_roles = _extract_rows_and_roles(eval_result)
    bucket_filter = None if monitored_bucket is None else str(monitored_bucket).strip().lower()
    if bucket_filter == '':
        bucket_filter = None

    monitored_rows: List[Dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row_name = str(row.get('task') or row.get('name') or row.get('row_name') or '')
        if not row_name:
            continue
        row_bucket = str(row.get('bucket') or row_roles.get(row_name, 'host')).strip().lower() or 'host'
        if bucket_filter is not None and row_bucket != bucket_filter:
            continue
        if not _match_task_pattern(row_name, monitored_task_pattern):
            continue
        monitored_rows.append(
            {
                'row_index': int(row_index),
                'row_name': row_name,
                'bucket': row_bucket,
                'row': row,
            }
        )
    return monitored_rows


def summarize_epoch_monitor(
    rows: Sequence[Dict[str, Any]],
    metric_name: str = 'R1',
    mode: str = 'max',
) -> Dict[str, Any]:
    normalized_metric_name = str(metric_name or 'R1')
    normalized_mode = str(mode or 'max').lower()
    if normalized_mode not in {'max', 'min'}:
        raise ValueError(f'Unsupported early-stopping mode: {mode!r}. Allowed values: max, min.')

    valid_rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []
    for item in rows:
        row = item.get('row', {}) if isinstance(item, dict) else {}
        if not isinstance(row, dict):
            invalid_rows.append(dict(item) if isinstance(item, dict) else {'row': row})
            continue
        metric_value = row.get(normalized_metric_name)
        if metric_value is None and isinstance(row.get('metrics'), dict):
            metric_value = row['metrics'].get(normalized_metric_name)
        try:
            metric_float = float(metric_value)
        except (TypeError, ValueError):
            invalid_rows.append(dict(item))
            continue
        if not math.isfinite(metric_float):
            invalid_rows.append(dict(item))
            continue
        row_entry = dict(item)
        row_entry['metric_value'] = metric_float
        valid_rows.append(row_entry)

    if not valid_rows:
        return {
            'metric_name': normalized_metric_name,
            'mode': normalized_mode,
            'best_value': None,
            'best_row_name': None,
            'best_row_full_metadata': None,
            'num_rows_total': int(len(rows)),
            'num_rows_considered': 0,
            'num_invalid_rows': int(len(invalid_rows)),
        }

    if normalized_mode == 'max':
        best_value = max(entry['metric_value'] for entry in valid_rows)
    else:
        best_value = min(entry['metric_value'] for entry in valid_rows)
    tied = [
        entry
        for entry in valid_rows
        if abs(float(entry['metric_value']) - float(best_value)) <= 1e-12
    ]
    best_row = sorted(
        tied,
        key=lambda entry: (
            str(entry.get('row_name', '')),
            int(entry.get('row_index', 0)),
        ),
    )[0]
    return {
        'metric_name': normalized_metric_name,
        'mode': normalized_mode,
        'best_value': float(best_value),
        'best_row_name': str(best_row.get('row_name', '')),
        'best_row_full_metadata': dict(best_row),
        'num_rows_total': int(len(rows)),
        'num_rows_considered': int(len(valid_rows)),
        'num_invalid_rows': int(len(invalid_rows)),
    }


class Evaluator:
    def __init__(self, img_loader, txt_loader, args):
        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.logger = logging.getLogger('pas.eval')
        self.args = args
        self.latest_metrics = {}
        self.latest_authority = {}
        self.latest_eval_rows = []

        requested_metrics = tuple(getattr(args, 'retrieval_metrics', SUPPORTED_RETRIEVAL_METRICS) or SUPPORTED_RETRIEVAL_METRICS)
        unknown_metrics = sorted(set(requested_metrics) - set(SUPPORTED_RETRIEVAL_METRICS))
        if unknown_metrics:
            raise ValueError(
                f'Unsupported evaluation.retrieval_metrics values: {unknown_metrics}. '
                f'Allowed values: {list(SUPPORTED_RETRIEVAL_METRICS)}'
            )
        self.requested_metrics = requested_metrics
        self.host_type = str(getattr(args, 'host_type', 'clip')).lower()
        self.itself_lambda_ablation_enabled = bool(getattr(args, 'itself_lambda_ablation_enabled', False))
        self.itself_lambda_ablation_include_default = bool(
            getattr(args, 'itself_lambda_ablation_include_default', True)
        )
        configured_alphas = getattr(args, 'itself_lambda_ablation_alphas', None)
        if configured_alphas in (None, []):
            self.itself_lambda_ablation_alphas = list(DEFAULT_ITSELF_ABLATION_ALPHAS)
        else:
            self.itself_lambda_ablation_alphas = [float(alpha) for alpha in configured_alphas]

    def _concat_feature_batches(self, batches):
        first = batches[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(batches, 0)
        if isinstance(first, dict):
            return {key: self._concat_feature_batches([batch[key] for batch in batches]) for key in first.keys()}
        if isinstance(first, (list, tuple)):
            return type(first)(self._concat_feature_batches(list(items)) for items in zip(*batches))
        return first

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

    def _compute_eval_debug_metrics(self, model, similarity, text_ids, image_ids, image_features):
        if not bool(getattr(self.args, 'log_debug_metrics', True)):
            return {}
        metrics = {}
        core_model = model.module if hasattr(model, 'module') else model
        similarity = similarity.detach().float().cpu()

        positive_mask = text_ids.view(-1, 1).eq(image_ids.view(1, -1))
        positive_counts = positive_mask.sum(dim=1)
        metrics['eval_positive_gallery_count_min'] = float(positive_counts.min().item())
        metrics['eval_positive_gallery_count_mean'] = float(positive_counts.float().mean().item())

        first_positive = positive_mask.to(dtype=torch.int64).argmax(dim=1)
        positive_scores = similarity.gather(1, first_positive.view(-1, 1)).squeeze(1)
        negative_mask = ~positive_mask
        hardest_negative = similarity.masked_fill(~negative_mask, float('-inf')).max(dim=1).values
        metrics['eval_positive_exact_cosine_mean'] = float(positive_scores.mean().item())
        metrics['eval_hardest_negative_exact_cosine_mean'] = float(hardest_negative.mean().item())
        metrics['eval_exact_margin_mean'] = float((positive_scores - hardest_negative).mean().item())

        image_projected = None
        if isinstance(image_features, dict):
            image_projected = image_features.get('host_image_projected', image_features.get('image_projected'))
        if isinstance(image_projected, torch.Tensor):
            image_norms = image_projected.detach().float().norm(dim=-1).cpu()
            metrics['eval_image_projected_norm_mean'] = float(image_norms.mean().item())
            metrics['eval_image_projected_norm_std'] = float(image_norms.std(unbiased=False).item())

        if hasattr(core_model, 'host_head') and hasattr(core_model.host_head, 'losses'):
            logit_scale = core_model.host_head.losses.get_logit_scale().detach().float().cpu()
            retrieval_temperature = core_model.host_head.losses.get_retrieval_temperature().detach().float().cpu()
            metrics['eval_logit_scale'] = float(logit_scale.item())
            metrics['eval_retrieval_temperature'] = float(retrieval_temperature.item())

        return build_validation_debug_metrics(metrics)

    def _compute_similarity(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        text_ids, image_ids = [], []
        text_batches, image_batches = [], []

        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                with build_autocast_context(self.args, device):
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

        text_features = self._feature_to_device(text_features, device)
        image_features = self._feature_to_device(image_features, device)

        with torch.no_grad():
            with build_autocast_context(self.args, device):
                similarity = model.compute_retrieval_similarity(image_features, text_features)

        expected_shape = (int(text_ids.numel()), int(image_ids.numel()))
        similarity = self._check_similarity_matrix(similarity, expected_shape, field_name='Host retrieval similarity')
        debug_metrics = self._compute_eval_debug_metrics(model, similarity, text_ids, image_ids, image_features)
        return similarity, text_ids, image_ids, debug_metrics, image_features, text_features

    def _itself_ablation_active(self) -> bool:
        return (
            self.host_type == 'itself'
            and self.itself_lambda_ablation_enabled
        )

    @staticmethod
    def _normalize_similarity(text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        text_embeddings = F.normalize(text_embeddings.float(), p=2, dim=1)
        image_embeddings = F.normalize(image_embeddings.float(), p=2, dim=1)
        return text_embeddings @ image_embeddings.t()

    def _build_itself_ablation_rows(
        self,
        similarity_default: torch.Tensor,
        image_features: Dict[str, object],
        text_features: Dict[str, object],
        text_ids: torch.Tensor,
        image_ids: torch.Tensor,
    ):
        rows = []
        global_image = image_features.get('global_image_embedding') if isinstance(image_features, dict) else None
        global_text = text_features.get('global_text_embedding') if isinstance(text_features, dict) else None
        grab_image = image_features.get('grab_image_embedding') if isinstance(image_features, dict) else None
        grab_text = text_features.get('grab_text_embedding') if isinstance(text_features, dict) else None

        if not (isinstance(global_image, torch.Tensor) and isinstance(global_text, torch.Tensor)):
            return [get_metrics(similarity=similarity_default, qids=text_ids, gids=image_ids, name=BEST_ROW_TASK_NAME)]

        sims_global = self._normalize_similarity(global_text, global_image).float().cpu()
        rows.append(get_metrics(similarity=sims_global, qids=text_ids, gids=image_ids, name='global-t2i'))

        if not (isinstance(grab_image, torch.Tensor) and isinstance(grab_text, torch.Tensor)):
            return rows

        sims_grab = self._normalize_similarity(grab_text, grab_image).float().cpu()
        rows.append(get_metrics(similarity=sims_grab, qids=text_ids, gids=image_ids, name='grab-t2i'))

        mix_alphas = list(self.itself_lambda_ablation_alphas)
        if self.itself_lambda_ablation_include_default:
            alpha_default = float(getattr(self.args, 'itself_score_weight_global', 0.68))
            if 0.0 <= alpha_default <= 1.0:
                mix_alphas.append(alpha_default)

        seen = set()
        for alpha in mix_alphas:
            rounded = round(float(alpha), 6)
            if rounded in seen:
                continue
            seen.add(rounded)
            alpha = float(alpha)
            if alpha < 0.0 or alpha > 1.0:
                continue
            sims_mix = (alpha * sims_global) + ((1.0 - alpha) * sims_grab)
            task_name = f'global+grab({alpha:.2f}'.rstrip('0').rstrip('.') + ')-t2i'
            rows.append(get_metrics(similarity=sims_mix, qids=text_ids, gids=image_ids, name=task_name))

        # The old static host row is removed; BEST_ROW_TASK_NAME is rebound to best row each eval epoch.
        return rows

    def eval(self, model):
        similarity, text_ids, image_ids, debug_metrics, image_features, text_features = self._compute_similarity(model)
        if self._itself_ablation_active():
            metric_rows = self._build_itself_ablation_rows(
                similarity_default=similarity,
                image_features=image_features,
                text_features=text_features,
                text_ids=text_ids,
                image_ids=image_ids,
            )
        else:
            metric_rows = [
                get_metrics(
                    similarity=similarity,
                    qids=text_ids,
                    gids=image_ids,
                    name=BEST_ROW_TASK_NAME,
                    stable_tiebreak=bool(getattr(self.args, 'repro_stable_eval_tiebreak', False)),
                )
            ]
        best_row = max(metric_rows, key=lambda row: float(row['R1']))
        metrics = dict(best_row)
        metrics['task'] = BEST_ROW_TASK_NAME
        metrics['source_task'] = best_row['task']

        row_by_task = {str(row.get('task', '')): dict(row) for row in metric_rows}
        row_by_task[BEST_ROW_TASK_NAME] = dict(metrics)
        metric_rows_with_host = list(row_by_task.values())

        authority_context: Dict[str, object] = {
            'display_row': BEST_ROW_TASK_NAME,
            'source_row': str(best_row['task']),
            'mismatch': False,
            'selected_source_role': 'host',
            'candidates': {'host': BEST_ROW_TASK_NAME},
            'row_roles': {str(row['task']): 'host' for row in metric_rows_with_host},
            'row_metrics': {
                row['task']: {
                    metric_name: float(row[metric_name])
                    for metric_name in SUPPORTED_RETRIEVAL_METRICS
                }
                for row in metric_rows_with_host
            },
        }
        self.latest_authority = authority_context
        self.latest_eval_rows = []
        for row in metric_rows_with_host:
            row_copy = {
                str(metric_key): (float(metric_value) if metric_key in SUPPORTED_RETRIEVAL_METRICS else metric_value)
                for metric_key, metric_value in row.items()
            }
            row_copy['bucket'] = str(authority_context['row_roles'].get(row['task'], 'host'))
            self.latest_eval_rows.append(row_copy)

        table = PrettyTable(['task'] + list(self.requested_metrics))
        for row in metric_rows_with_host:
            table.add_row([row['task']] + [row[metric_name] for metric_name in self.requested_metrics])
        for metric_name in self.requested_metrics:
            table.custom_format[metric_name] = lambda _, value: f'{value:.2f}'

        retrieval_metrics = {metric_name: best_row[metric_name] for metric_name in self.requested_metrics}
        self.latest_metrics = build_validation_retrieval_metrics(retrieval_metrics)
        self.latest_metrics['val/top1'] = best_row['R1']
        self.latest_metrics['val/top1_row'] = BEST_ROW_TASK_NAME
        self.latest_metrics['val/top1_source_row'] = best_row['task']
        self.latest_metrics['val/top1_display_row'] = BEST_ROW_TASK_NAME
        self.latest_metrics['val/authority/selected_source_role'] = 'host'
        self.latest_metrics['val/authority/selected_authority_row'] = BEST_ROW_TASK_NAME
        self.latest_metrics['val/authority/display_source_mismatch'] = 0.0
        self.latest_metrics['val/authority/host_candidate_row'] = BEST_ROW_TASK_NAME
        self.latest_metrics['val/authority/prototype_candidate_row'] = None
        self.latest_metrics.update(debug_metrics)

        self.logger.info('\n' + str(table))
        return best_row['R1']
