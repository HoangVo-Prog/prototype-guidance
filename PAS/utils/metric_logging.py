from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F


TRAIN_LOSS_KEYS = (
    'loss_total',
    'loss_proxy',
    'loss_proxy_image',
    'loss_proxy_text',
    'loss_proxy_text_exact',
    'loss_align',
    'loss_diag',
    'loss_support',
    'loss_diversity',
    'loss_balance',
    'loss_proxy_weighted',
    'loss_align_weighted',
    'loss_diag_weighted',
    'loss_support_weighted',
    'loss_diversity_weighted',
    'loss_balance_weighted',
)


DEBUG_METRIC_MAP = {
    'logit_scale': 'debug/logit_scale',
    'proxy_temperature': 'debug/proxy_temperature',
    'retrieval_temperature': 'debug/retrieval_temperature',
    'surrogate_t_pool_norm': 'debug/surrogate_t_pool_norm',
    'exact_t_pool_norm': 'debug/exact_t_pool_norm',
    'surrogate_text_embed_norm': 'debug/surrogate_text_embed_norm',
    'surrogate_text_embed_norm_std': 'debug/surrogate_text_embed_norm_std',
    'surrogate_text_embed_norm_min': 'debug/surrogate_text_embed_norm_min',
    'surrogate_text_embed_norm_max': 'debug/surrogate_text_embed_norm_max',
    'exact_text_embed_norm': 'debug/exact_text_embed_norm',
    'exact_text_embed_norm_std': 'debug/exact_text_embed_norm_std',
    'exact_text_embed_norm_min': 'debug/exact_text_embed_norm_min',
    'exact_text_embed_norm_max': 'debug/exact_text_embed_norm_max',
    'image_embed_norm_std': 'debug/image_embed_norm_std',
    'image_embed_norm_min': 'debug/image_embed_norm_min',
    'image_embed_norm_max': 'debug/image_embed_norm_max',
    'text_embed_norm_std': 'debug/text_embed_norm_std',
    'text_embed_norm_min': 'debug/text_embed_norm_min',
    'text_embed_norm_max': 'debug/text_embed_norm_max',
    'prototype_usage_entropy': 'debug/prototype_usage_entropy',
    'prototype_usage_max': 'debug/prototype_usage_max',
    'prototype_dead_count': 'debug/prototype_dead_count',
    'routing_max_prob': 'debug/routing_max_prob',
    'routing_entropy': 'debug/routing_entropy',
    'routing_top1_usage_entropy': 'debug/routing_top1_usage_entropy',
    'routing_top1_usage_max': 'debug/routing_top1_usage_max',
    'routing_top1_dead_count': 'debug/routing_top1_dead_count',
    'prototype_assignment_entropy': 'debug/prototype_assignment_entropy',
    'routing_effective_support': 'debug/routing_effective_support',
    'routing_effective_support_ipr': 'debug/routing_effective_support_ipr',
    'image_proxy_logit_mean': 'debug/image_proxy_logit_mean',
    'image_proxy_logit_std': 'debug/image_proxy_logit_std',
    'image_proxy_logit_min': 'debug/image_proxy_logit_min',
    'image_proxy_logit_max': 'debug/image_proxy_logit_max',
    'text_proxy_logit_mean': 'debug/text_proxy_logit_mean',
    'text_proxy_logit_std': 'debug/text_proxy_logit_std',
    'text_proxy_logit_min': 'debug/text_proxy_logit_min',
    'text_proxy_logit_max': 'debug/text_proxy_logit_max',
    'image_positive_proxy_cosine_mean': 'debug/image_positive_proxy_cosine_mean',
    'image_positive_proxy_cosine_std': 'debug/image_positive_proxy_cosine_std',
    'image_hardest_negative_proxy_cosine_mean': 'debug/image_hardest_negative_proxy_cosine_mean',
    'image_hardest_negative_proxy_cosine_std': 'debug/image_hardest_negative_proxy_cosine_std',
    'image_proxy_margin_mean': 'debug/image_proxy_margin_mean',
    'image_proxy_margin_min': 'debug/image_proxy_margin_min',
    'text_positive_proxy_cosine_mean': 'debug/text_positive_proxy_cosine_mean',
    'text_positive_proxy_cosine_std': 'debug/text_positive_proxy_cosine_std',
    'text_hardest_negative_proxy_cosine_mean': 'debug/text_hardest_negative_proxy_cosine_mean',
    'text_hardest_negative_proxy_cosine_std': 'debug/text_hardest_negative_proxy_cosine_std',
    'text_proxy_margin_mean': 'debug/text_proxy_margin_mean',
    'text_proxy_margin_min': 'debug/text_proxy_margin_min',
    'text_exact_positive_proxy_cosine_mean': 'debug/text_exact_positive_proxy_cosine_mean',
    'text_exact_positive_proxy_cosine_std': 'debug/text_exact_positive_proxy_cosine_std',
    'text_exact_hardest_negative_proxy_cosine_mean': 'debug/text_exact_hardest_negative_proxy_cosine_mean',
    'text_exact_hardest_negative_proxy_cosine_std': 'debug/text_exact_hardest_negative_proxy_cosine_std',
    'text_exact_proxy_margin_mean': 'debug/text_exact_proxy_margin_mean',
    'text_exact_proxy_margin_min': 'debug/text_exact_proxy_margin_min',
    'class_proxy_norm_mean': 'debug/class_proxy_norm_mean',
    'class_proxy_norm_std': 'debug/class_proxy_norm_std',
    'class_proxy_norm_min': 'debug/class_proxy_norm_min',
    'class_proxy_norm_max': 'debug/class_proxy_norm_max',
    'class_proxy_norm_normalized_mean': 'debug/class_proxy_norm_normalized_mean',
    'class_proxy_norm_normalized_std': 'debug/class_proxy_norm_normalized_std',
    'class_proxy_norm_normalized_min': 'debug/class_proxy_norm_normalized_min',
    'class_proxy_norm_normalized_max': 'debug/class_proxy_norm_normalized_max',
    'grad_norm_class_proxies': 'debug/grad_norm_class_proxies',
    'grad_norm_image_projector': 'debug/grad_norm_image_projector',
    'grad_norm_text_projector': 'debug/grad_norm_text_projector',
    'grad_norm_prototype_bank': 'debug/grad_norm_prototype_bank',
    'grad_norm_image_backbone': 'debug/grad_norm_image_backbone',
    'grad_norm_text_backbone': 'debug/grad_norm_text_backbone',
    'grad_norm_image_projected_output': 'debug/grad_norm_image_projected_output',
    'grad_norm_surrogate_text_projected_output': 'debug/grad_norm_surrogate_text_projected_output',
    'grad_norm_exact_text_projected_output': 'debug/grad_norm_exact_text_projected_output',
    'grad_norm_total': 'debug/grad_norm_total',
    'token_pool_entropy': 'debug/token_pool_entropy',
    'beta_max_prob': 'debug/beta_max_prob',
    'token_special_mass': 'debug/token_special_mass',
    'token_valid_fraction': 'debug/token_valid_fraction',
    'valid_token_fraction': 'debug/valid_token_fraction',
    'prototype_pairwise_cosine_mean': 'debug/prototype_pairwise_cosine_mean',
    'prototype_pairwise_cosine_std': 'debug/prototype_pairwise_cosine_std',
    'prototype_pairwise_cosine_max': 'debug/prototype_pairwise_cosine_max',
    'contextualized_prototype_pairwise_cosine_mean': 'debug/contextualized_prototype_pairwise_cosine_mean',
    'contextualized_prototype_pairwise_cosine_std': 'debug/contextualized_prototype_pairwise_cosine_std',
    'contextualized_prototype_pairwise_cosine_max': 'debug/contextualized_prototype_pairwise_cosine_max',
    'prototype_contextualization_entropy': 'debug/prototype_contextualization_entropy',
    'q_norm': 'debug/q_norm',
    't_pool_norm': 'debug/t_pool_norm',
    'image_feature_norm': 'debug/image_feature_norm',
    'image_embed_norm': 'debug/image_embed_norm',
    'image_embed_norm_raw': 'debug/image_embed_norm_raw',
    'image_embed_unit_norm': 'debug/image_embed_unit_norm',
    'text_embed_norm': 'debug/text_embed_norm',
    'text_embed_norm_raw': 'debug/text_embed_norm_raw',
    'text_embed_unit_norm': 'debug/text_embed_unit_norm',
    'surrogate_text_embed_norm_raw': 'debug/surrogate_text_embed_norm_raw',
    'surrogate_text_embed_unit_norm': 'debug/surrogate_text_embed_unit_norm',
    'exact_text_embed_norm_raw': 'debug/exact_text_embed_norm_raw',
    'exact_text_embed_unit_norm': 'debug/exact_text_embed_unit_norm',
}

DEBUG_SKIP_KEYS = {
    'image_global',
    'text_tokens',
    'token_mask',
    'special_token_positions',
    'alpha',
    'beta',
    'Q',
    'Theta_v',
    'Theta_tilde',
    'T_pool',
    'Z_v',
    'Z_v_raw',
    'Z_t',
    'Z_t_raw',
    'Z_t_exact',
    'Z_t_exact_raw',
    'raw_prototypes',
    'contextualized_prototypes',
    'prototype_similarity',
    'contextualization_weights',
    'routing_similarity',
    'alpha_logits',
    'routing_logits',
    'routing_weights',
    'routing_top1_histogram',
    'prototype_summary',
    'token_similarity',
    'token_scores',
    'token_scores_scaled',
    'valid_mask',
    'token_valid_mask',
    'token_keep_mask',
    'masked_logits',
    'beta_logits_masked',
    'token_weights',
    'pooled_text',
    'basis_bank',
    'basis_token_scores',
    'basis_token_weights',
    'basis_beta_logits_masked',
    'projected_features',
    'projected_features_norm',
    'projected_features_pre_norm',
    'projected_features_raw',
    'prototype_usage',
    'contrastive_logits',
}

TRACKED_SCALAR_KEYS = TRAIN_LOSS_KEYS + tuple(sorted(set(DEBUG_METRIC_MAP.keys())))


def to_scalar(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, 'detach'):
        value = value.detach()
        if value.numel() != 1:
            return None
        return float(value.float().cpu().item())
    return None


def compute_prototype_geometry_metrics(debug_data: Dict[str, object]) -> Dict[str, float]:
    metrics = {}
    geometry_specs = (
        ('Theta_v', 'prototype_pairwise_cosine'),
        ('Theta_tilde', 'contextualized_prototype_pairwise_cosine'),
    )
    for tensor_key, metric_prefix in geometry_specs:
        theta = debug_data.get(tensor_key)
        if not isinstance(theta, torch.Tensor) or theta.ndim != 2 or theta.size(0) <= 1:
            continue
        normalized = F.normalize(theta.detach().float(), dim=-1)
        similarity = normalized @ normalized.t()
        mask = ~torch.eye(similarity.size(0), device=similarity.device, dtype=torch.bool)
        off_diagonal = similarity[mask]
        if off_diagonal.numel() == 0:
            continue
        metrics[f'{metric_prefix}_mean'] = float(off_diagonal.mean().cpu().item())
        metrics[f'{metric_prefix}_std'] = float(off_diagonal.std(unbiased=False).cpu().item())
        metrics[f'{metric_prefix}_max'] = float(off_diagonal.max().cpu().item())
    return metrics


def collect_loss_metrics(outputs: Dict[str, object]) -> Dict[str, float]:
    metrics = {}
    for key in TRAIN_LOSS_KEYS:
        scalar = to_scalar(outputs.get(key))
        if scalar is not None:
            metrics[key] = scalar
    return metrics


def collect_debug_metrics(outputs: Dict[str, object], include_debug_metrics: bool = True) -> Dict[str, float]:
    if not include_debug_metrics:
        return {}

    debug_data = outputs.get('debug', {}) if isinstance(outputs.get('debug', {}), dict) else {}
    metrics = {}
    metrics.update(compute_prototype_geometry_metrics(debug_data))

    for raw_key, mapped_key in DEBUG_METRIC_MAP.items():
        value = outputs.get(raw_key, debug_data.get(raw_key))
        scalar = to_scalar(value)
        if scalar is None:
            continue
        metrics[mapped_key] = scalar

    for raw_key, value in debug_data.items():
        if raw_key in DEBUG_SKIP_KEYS or isinstance(value, dict):
            continue
        mapped_key = DEBUG_METRIC_MAP.get(raw_key, f'debug/{raw_key}')
        scalar = to_scalar(value)
        if scalar is None:
            continue
        metrics[mapped_key] = scalar

    return metrics


def collect_scalar_metrics(outputs: Dict[str, object], include_debug_metrics: bool = True) -> Dict[str, float]:
    metrics = {}
    metrics.update(collect_loss_metrics(outputs))
    debug_metrics = collect_debug_metrics(outputs, include_debug_metrics=include_debug_metrics)
    for metric_name, metric_value in debug_metrics.items():
        if metric_name.startswith('debug/'):
            raw_name = metric_name.split('/', 1)[1]
            metrics[raw_name] = metric_value
        else:
            metrics[metric_name] = metric_value
    return metrics


def build_train_metrics(epoch: int, step: Optional[int], outputs: Dict[str, object], lr: float, include_debug_metrics: bool = True) -> Dict[str, float]:
    metrics = {
        'train/epoch': float(epoch),
        'train/lr': float(lr),
    }
    if step is not None:
        metrics['train/step'] = float(step)
    for key, value in collect_loss_metrics(outputs).items():
        metrics[f'train/{key}'] = value
    metrics.update(collect_debug_metrics(outputs, include_debug_metrics=include_debug_metrics))
    return metrics


def build_validation_metrics(epoch: int, evaluator=None, val_loss: Optional[float] = None) -> Dict[str, float]:
    metrics = {
        'val/epoch': float(epoch),
    }
    if val_loss is not None:
        metrics['val/loss_total'] = float(val_loss)
    if evaluator is not None and getattr(evaluator, 'latest_metrics', None):
        metrics.update(evaluator.latest_metrics)
    return metrics

