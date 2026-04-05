import logging

from prettytable import PrettyTable
import torch
import torch.nn.functional as F

from utils.precision import build_autocast_context, is_cuda_device


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
        if not bool(getattr(args, 'use_prototype_bank', True)) and self.retrieval_scorer == 'approximate':
            self.logger.warning(
                'evaluation.retrieval_scorer=approximate with model.use_prototype_bank=false is accepted for config freedom; falling back to exact retrieval scoring.'
            )
            self.retrieval_scorer = 'exact'

    def _concat_feature_batches(self, batches):
        first = batches[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(batches, 0)
        if isinstance(first, dict):
            return {key: self._concat_feature_batches([batch[key] for batch in batches]) for key in first.keys()}
        raise TypeError(f'Unsupported feature batch type: {type(first)}')

    def _feature_batch_size(self, features):
        if isinstance(features, torch.Tensor):
            if features.ndim == 0:
                raise ValueError('Feature tensors must have a batch dimension.')
            return int(features.size(0))
        if isinstance(features, dict):
            if not features:
                raise ValueError('Feature dictionaries must not be empty.')
            first_key = next(iter(features))
            return self._feature_batch_size(features[first_key])
        raise TypeError(f'Unsupported feature container type: {type(features)}')

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
        metrics = {}
        core_model = model.module if hasattr(model, 'module') else model
        similarity = similarity.detach().float().cpu()
        positive_mask, positive_counts, first_positive = self._positive_gallery_structure(text_ids, image_ids)
        metrics['val/debug/eval_positive_gallery_count_min'] = float(positive_counts.min().item())
        metrics['val/debug/eval_positive_gallery_count_mean'] = float(positive_counts.float().mean().item())

        logit_scale_value = None
        if hasattr(core_model, 'prototype_head') and hasattr(core_model.prototype_head, 'losses'):
            logit_scale = core_model.prototype_head.losses.get_logit_scale().detach().float().cpu()
            retrieval_temperature = core_model.prototype_head.losses.get_retrieval_temperature().detach().float().cpu()
            logit_scale_value = float(logit_scale.item())
            metrics['val/debug/eval_logit_scale'] = logit_scale_value
            metrics['val/debug/eval_retrieval_temperature'] = float(retrieval_temperature.item())

        cosine_similarity = similarity
        if logit_scale_value is not None and logit_scale_value > 0.0:
            cosine_similarity = similarity / logit_scale_value

        positive_scores = cosine_similarity.gather(1, first_positive.view(-1, 1)).squeeze(1)
        negative_mask = ~positive_mask
        if negative_mask.any(dim=1).all():
            hardest_negative = cosine_similarity.masked_fill(~negative_mask, float('-inf')).max(dim=1).values
        else:
            hardest_negative = torch.zeros_like(positive_scores)
        metrics['val/debug/eval_positive_exact_cosine_mean'] = float(positive_scores.mean().item())
        metrics['val/debug/eval_hardest_negative_exact_cosine_mean'] = float(hardest_negative.mean().item())
        metrics['val/debug/eval_exact_margin_mean'] = float((positive_scores - hardest_negative).mean().item())

        image_projected = image_features.get('image_projected') if isinstance(image_features, dict) else None
        if isinstance(image_projected, torch.Tensor):
            image_norms = image_projected.detach().float().norm(dim=-1).cpu()
            metrics['val/debug/eval_image_projected_norm_mean'] = float(image_norms.mean().item())
            metrics['val/debug/eval_image_projected_norm_std'] = float(image_norms.std(unbiased=False).item())

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
                metrics['val/debug/eval_positive_exact_text_embed_norm_mean'] = float(exact_raw_norms.mean().item())
                metrics['val/debug/eval_positive_exact_text_embed_norm_std'] = float(exact_raw_norms.std(unbiased=False).item())
                metrics['val/debug/eval_positive_exact_text_embed_unit_norm_mean'] = float(exact_unit_norms.mean().item())
                metrics['val/debug/eval_positive_exact_pair_cosine_mean'] = float(paired_cosine.mean().item())

        return metrics

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
            text_batches.append({
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else {sub_key: sub_value.detach().cpu() for sub_key, sub_value in value.items()}
                for key, value in text_features.items()
            })

        for pid, image in self.img_loader:
            image = image.to(device)
            with torch.no_grad():
                with build_autocast_context(self.args, device):
                    image_features = model.encode_image_for_retrieval(image)
            image_ids.append(pid.view(-1))
            image_batches.append({key: value.detach().cpu() for key, value in image_features.items()})

        text_ids = torch.cat(text_ids, 0).cpu()
        image_ids = torch.cat(image_ids, 0).cpu()
        text_features = self._concat_feature_batches(text_batches)
        image_features = self._concat_feature_batches(image_batches)

        if self._feature_batch_size(text_features) != int(text_ids.numel()):
            raise ValueError('Text feature concatenation produced a batch size that does not match text_ids ordering.')
        if self._feature_batch_size(image_features) != int(image_ids.numel()):
            raise ValueError('Image feature concatenation produced a batch size that does not match image_ids ordering.')

        text_features = {
            key: value.to(device) if isinstance(value, torch.Tensor) else {sub_key: sub_value.to(device) for sub_key, sub_value in value.items()}
            for key, value in text_features.items()
        }
        image_features = {key: value.to(device) for key, value in image_features.items()}

        with torch.no_grad():
            with build_autocast_context(self.args, device):
                if self.retrieval_scorer == 'approximate':
                    similarity = model.compute_approximate_retrieval_similarity(image_features, text_features).cpu()
                else:
                    similarity = model.compute_retrieval_similarity(image_features, text_features).cpu()

        expected_shape = (int(text_ids.numel()), int(image_ids.numel()))
        if tuple(similarity.shape) != expected_shape:
            raise ValueError(
                f'Retrieval similarity has shape {tuple(similarity.shape)} but expected {expected_shape} '
                'from concatenated text/image ordering.'
            )
        if not torch.isfinite(similarity).all():
            raise FloatingPointError('Retrieval similarity contains NaN or Inf values after evaluation.')

        debug_metrics = self._compute_eval_debug_metrics(model, similarity, text_ids, image_ids, image_features, text_features)
        return similarity, text_ids, image_ids, debug_metrics

    def eval(self, model):
        similarity, text_ids, image_ids, debug_metrics = self._compute_similarity(model)
        metrics = get_metrics(similarity, text_ids, image_ids, 'pas-t2i')

        table = PrettyTable(['task'] + list(self.requested_metrics))
        table.add_row([metrics['task']] + [metrics[metric_name] for metric_name in self.requested_metrics])
        for metric_name in self.requested_metrics:
            table.custom_format[metric_name] = lambda _, value: f'{value:.2f}'

        self.latest_metrics = {
            f'val/pas/{metric_name}': metrics[metric_name]
            for metric_name in self.requested_metrics
        }
        self.latest_metrics['val/top1'] = metrics['R1']
        self.latest_metrics.update(debug_metrics)

        self.logger.info('\n' + str(table))
        if debug_metrics:
            positive_cos = debug_metrics.get('val/debug/eval_positive_exact_cosine_mean')
            hardest_negative = debug_metrics.get('val/debug/eval_hardest_negative_exact_cosine_mean')
            margin = debug_metrics.get('val/debug/eval_exact_margin_mean')
            if positive_cos is not None and hardest_negative is not None and margin is not None:
                self.logger.info(
                    'Retrieval sanity: positive_exact_cos=%.4f hardest_negative_exact_cos=%.4f margin=%.4f',
                    positive_cos,
                    hardest_negative,
                    margin,
                )
        self.logger.info('\ncurrent R1 = ' + str(metrics['R1']))
        return metrics['R1']








