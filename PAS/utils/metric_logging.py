from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F


TRAIN_LOSS_KEYS = (
    'loss_total',
    'loss_infonce',
    'loss_diversity',
    'loss_balance',
    'loss_diversity_weighted',
    'loss_balance_weighted',
)

DEBUG_METRIC_MAP = {
    'logit_scale': 'debug/logit_scale',
    'prototype_usage_entropy': 'debug/prototype_usage_entropy',
    'prototype_usage_max': 'debug/prototype_usage_max',
    'prototype_dead_count': 'debug/prototype_dead_count',
    'routing_max_prob': 'debug/routing_max_prob',
    'routing_entropy': 'debug/routing_entropy',
    'prototype_assignment_entropy': 'debug/prototype_assignment_entropy',
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
    'text_embed_norm': 'debug/text_embed_norm',
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
    'raw_prototypes',
    'contextualized_prototypes',
    'prototype_similarity',
    'contextualization_weights',
    'routing_similarity',
    'alpha_logits',
    'routing_logits',
    'routing_weights',
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

    if 'temperature' in outputs and 'debug/logit_scale' not in metrics:
        temperature = to_scalar(outputs['temperature'])
        if temperature not in (None, 0.0):
            metrics['debug/logit_scale'] = 1.0 / temperature

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
