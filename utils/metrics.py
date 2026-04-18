import logging
from typing import Dict, Tuple

from prettytable import PrettyTable
import torch

from utils.precision import build_autocast_context, is_cuda_device
from utils.metric_logging import build_validation_debug_metrics, build_validation_retrieval_metrics


SUPPORTED_RETRIEVAL_METRICS = ('R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum')


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
        if bool(getattr(self.args, 'amp', False)) and not is_cuda_device(device):
            raise ValueError('training.amp=true requires a CUDA device.')

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
        return similarity, text_ids, image_ids, debug_metrics

    def eval(self, model):
        similarity, text_ids, image_ids, debug_metrics = self._compute_similarity(model)
        metrics = get_metrics(similarity=similarity, qids=text_ids, gids=image_ids, name='host-t2i')

        authority_context: Dict[str, object] = {
            'display_row': 'host-t2i',
            'source_row': 'host-t2i',
            'mismatch': False,
            'selected_source_role': 'host',
            'candidates': {'host': 'host-t2i'},
            'row_roles': {'host-t2i': 'host'},
            'row_metrics': {
                'host-t2i': {
                    metric_name: float(metrics[metric_name])
                    for metric_name in SUPPORTED_RETRIEVAL_METRICS
                }
            },
        }
        self.latest_authority = authority_context

        table = PrettyTable(['task'] + list(self.requested_metrics))
        table.add_row([metrics['task']] + [metrics[metric_name] for metric_name in self.requested_metrics])
        for metric_name in self.requested_metrics:
            table.custom_format[metric_name] = lambda _, value: f'{value:.2f}'

        retrieval_metrics = {metric_name: metrics[metric_name] for metric_name in self.requested_metrics}
        self.latest_metrics = build_validation_retrieval_metrics(retrieval_metrics)
        self.latest_metrics['val/top1'] = metrics['R1']
        self.latest_metrics['val/top1_row'] = 'host-t2i'
        self.latest_metrics['val/top1_source_row'] = 'host-t2i'
        self.latest_metrics['val/top1_display_row'] = 'host-t2i'
        self.latest_metrics['val/authority/selected_source_role'] = 'host'
        self.latest_metrics['val/authority/selected_authority_row'] = 'host-t2i'
        self.latest_metrics['val/authority/display_source_mismatch'] = 0.0
        self.latest_metrics['val/authority/host_candidate_row'] = 'host-t2i'
        self.latest_metrics['val/authority/prototype_candidate_row'] = None
        self.latest_metrics.update(debug_metrics)

        self.logger.info('\n' + str(table))
        return metrics['R1']
