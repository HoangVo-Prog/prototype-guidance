from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


_LOGGER = logging.getLogger('pas.metrics.prototype')
_WARNED_EVENTS = set()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_EVENTS:
        return
    _WARNED_EVENTS.add(key)
    _LOGGER.warning(message)


def compute_proto_label_nmi(proto_ids: torch.Tensor, labels: torch.Tensor) -> float:
    try:
        from sklearn.metrics import normalized_mutual_info_score
    except Exception as exc:
        _warn_once('nmi_import_failed', f'Failed to import sklearn normalized_mutual_info_score ({exc!r}); logging NaN.')
        return float('nan')

    try:
        proto_ids_np = proto_ids.detach().cpu().numpy().reshape(-1)
        labels_np = labels.detach().cpu().numpy().reshape(-1)
        if proto_ids_np.shape[0] != labels_np.shape[0]:
            raise ValueError(f'NMI input length mismatch: proto_ids={proto_ids_np.shape}, labels={labels_np.shape}')
        if proto_ids_np.shape[0] == 0:
            return float('nan')
        if np.isnan(proto_ids_np).any() or np.isnan(labels_np).any():
            return float('nan')
        return float(normalized_mutual_info_score(labels_np, proto_ids_np))
    except Exception as exc:
        _warn_once('nmi_compute_failed', f'Failed to compute prototype-label NMI ({exc!r}); logging NaN.')
        return float('nan')


def compute_proto_cosine_offdiag(prototype_bank: torch.Tensor) -> torch.Tensor:
    if not isinstance(prototype_bank, torch.Tensor) or prototype_bank.ndim != 2:
        raise ValueError(f'prototype_bank must be 2D tensor, got {type(prototype_bank)!r} shape={getattr(prototype_bank, "shape", None)}')
    prototypes = F.normalize(prototype_bank.detach().float(), dim=-1)
    similarity = prototypes @ prototypes.t()
    count = similarity.size(0)
    mask = ~torch.eye(count, dtype=torch.bool, device=similarity.device)
    return similarity[mask]


def compute_proto_cosine_stats(prototype_bank: torch.Tensor) -> Dict[str, float]:
    offdiag = compute_proto_cosine_offdiag(prototype_bank)
    if offdiag.numel() == 0:
        return {'min': float('nan'), 'mean': float('nan'), 'max': float('nan')}
    return {
        'min': float(offdiag.min().item()),
        'mean': float(offdiag.mean().item()),
        'max': float(offdiag.max().item()),
    }


def compute_proto_high_sim_ratio_from_offdiag(offdiag: torch.Tensor, threshold: float = 0.8) -> float:
    if not isinstance(offdiag, torch.Tensor) or offdiag.numel() == 0:
        return float('nan')
    return float((offdiag > float(threshold)).float().mean().item())


def compute_prototype_activation_corr_stats(routing_probs: torch.Tensor) -> Dict[str, float]:
    if not isinstance(routing_probs, torch.Tensor) or routing_probs.ndim != 2:
        raise ValueError(f'routing_probs must be 2D tensor, got {type(routing_probs)!r} shape={getattr(routing_probs, "shape", None)}')

    activations = routing_probs.detach().float()
    batch_size, num_prototypes = activations.shape
    if batch_size < 2 or num_prototypes < 2:
        return {'mean': float('nan'), 'max': float('nan')}

    centered = activations - activations.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, keepdim=True).clamp_min(1e-8)
    normalized = centered / std
    corr = (normalized.t() @ normalized) / max(batch_size - 1, 1)
    mask = ~torch.eye(num_prototypes, dtype=torch.bool, device=corr.device)
    offdiag = corr[mask]
    if offdiag.numel() == 0:
        return {'mean': float('nan'), 'max': float('nan')}
    return {'mean': float(offdiag.mean().item()), 'max': float(offdiag.max().item())}


def summarize_proto_label_sanity(proto_ids: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    return {
        'proto_min': float(proto_ids.min().item()),
        'proto_max': float(proto_ids.max().item()),
        'label_min': float(labels.min().item()),
        'label_max': float(labels.max().item()),
        'proto_unique_count': float(torch.unique(proto_ids).numel()),
        'label_unique_count': float(torch.unique(labels).numel()),
    }


def summarize_routing_probs_sanity(routing_probs: torch.Tensor) -> Dict[str, float]:
    row_sums = routing_probs.sum(dim=1)
    return {
        'routing_row_sum_min': float(row_sums.min().item()),
        'routing_row_sum_mean': float(row_sums.mean().item()),
        'routing_row_sum_max': float(row_sums.max().item()),
        'routing_min': float(routing_probs.min().item()),
        'routing_max': float(routing_probs.max().item()),
        'routing_has_nan': float(torch.isnan(routing_probs).any().item()),
        'routing_has_inf': float(torch.isinf(routing_probs).any().item()),
    }


def summarize_prototype_bank_sanity(prototype_bank: torch.Tensor) -> Dict[str, float]:
    norms = prototype_bank.detach().float().norm(dim=-1)
    return {
        'prototype_norm_min': float(norms.min().item()),
        'prototype_norm_mean': float(norms.mean().item()),
        'prototype_norm_max': float(norms.max().item()),
        'prototype_has_nan': float(torch.isnan(prototype_bank).any().item()),
        'prototype_has_inf': float(torch.isinf(prototype_bank).any().item()),
    }


def resolve_prototype_bank_tensor(model) -> Optional[torch.Tensor]:
    core_model = model.module if hasattr(model, 'module') else model
    prototype_head = getattr(core_model, 'prototype_head', None)
    if prototype_head is None:
        return None
    prototype_bank = getattr(prototype_head, 'prototype_bank', None)
    if prototype_bank is None:
        return None
    if hasattr(prototype_bank, 'get_prototypes'):
        return prototype_bank.get_prototypes()
    if isinstance(prototype_bank, torch.Tensor):
        return prototype_bank
    return None


def collect_prototype_debug_metrics(
    *,
    proto_ids: torch.Tensor,
    labels: torch.Tensor,
    routing_probs: torch.Tensor,
    prototype_bank: torch.Tensor,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    offdiag = compute_proto_cosine_offdiag(prototype_bank)
    cosine_stats = compute_proto_cosine_stats(prototype_bank)
    activation_corr_stats = compute_prototype_activation_corr_stats(routing_probs)
    label_sanity = summarize_proto_label_sanity(proto_ids, labels)
    routing_sanity = summarize_routing_probs_sanity(routing_probs)
    bank_sanity = summarize_prototype_bank_sanity(prototype_bank)

    metrics = {
        'routing_proto_label_nmi': compute_proto_label_nmi(proto_ids, labels),
        'prototype_cosine_offdiag_min': cosine_stats['min'],
        'prototype_cosine_offdiag_mean': cosine_stats['mean'],
        'prototype_cosine_offdiag_max': cosine_stats['max'],
        'prototype_high_similarity_pair_ratio_07': compute_proto_high_sim_ratio_from_offdiag(offdiag, threshold=0.7),
        'prototype_high_similarity_pair_ratio_08': compute_proto_high_sim_ratio_from_offdiag(offdiag, threshold=0.8),
        'prototype_high_similarity_pair_ratio_09': compute_proto_high_sim_ratio_from_offdiag(offdiag, threshold=0.9),
        'prototype_activation_corr_mean': activation_corr_stats['mean'],
        'prototype_activation_corr_max': activation_corr_stats['max'],
        'debug_proto_unique_count_in_batch': label_sanity['proto_unique_count'],
        'debug_label_unique_count_in_batch': label_sanity['label_unique_count'],
        'debug_prototype_norm_mean': bank_sanity['prototype_norm_mean'],
        'debug_routing_prob_row_sum_mean': routing_sanity['routing_row_sum_mean'],
    }

    sanity = {}
    sanity.update(label_sanity)
    sanity.update(routing_sanity)
    sanity.update(bank_sanity)
    return metrics, sanity
