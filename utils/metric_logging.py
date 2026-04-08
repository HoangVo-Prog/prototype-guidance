from collections import deque
from typing import Deque, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


TRAIN_LOSS_KEYS = (
    'loss_total',
    'loss_host',
    'loss_host_ret',
    'loss_host_ret_i2t',
    'loss_host_ret_t2i',
    'loss_host_cid',
    'loss_proto_total',
    'loss_host_weighted',
    'lambda_host',
    'loss_proxy',
    'loss_proxy_image',
    'loss_proxy_text',
    'loss_proxy_text_exact',
    'loss_ret',
    'loss_align',
    'loss_diag',
    'loss_support',
    'loss_diversity',
    'loss_balance',
    'loss_proxy_image_weighted',
    'loss_proxy_text_weighted',
    'loss_proxy_text_exact_weighted',
    'loss_proxy_weighted',
    'loss_ret_weighted',
    'loss_align_weighted',
    'loss_diag_weighted',
    'loss_support_weighted',
    'loss_diversity_weighted',
    'loss_balance_weighted',
)


DEBUG_METRIC_MAP = {
    'logit_scale': 'debug/logit_scale',
    'host_logit_scale': 'debug/host_logit_scale',
    'host_retrieval_temperature': 'debug/host_retrieval_temperature',
    'fusion_coefficient': 'debug/fusion_coefficient',
    'host_loss_total': 'debug/host_loss_total',
    'host_loss_ret': 'debug/host_loss_ret',
    'proxy_temperature': 'debug/proxy_temperature',
    'retrieval_temperature': 'debug/retrieval_temperature',
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
    'routing_top1_active_count_window_100': 'debug/routing_top1_active_count_window_100',
    'routing_top1_active_count_window_500': 'debug/routing_top1_active_count_window_500',
    'routing_top1_dead_count_window_100': 'debug/routing_top1_dead_count_window_100',
    'routing_top1_dead_count_window_500': 'debug/routing_top1_dead_count_window_500',
    'routing_top1_usage_entropy_window_100': 'debug/routing_top1_usage_entropy_window_100',
    'routing_top1_usage_entropy_window_500': 'debug/routing_top1_usage_entropy_window_500',
    'routing_top1_usage_max_window_100': 'debug/routing_top1_usage_max_window_100',
    'routing_top1_usage_max_window_500': 'debug/routing_top1_usage_max_window_500',
    'prototype_assignment_entropy': 'debug/prototype_assignment_entropy',
    'routing_effective_support': 'debug/routing_effective_support',
    'routing_effective_support_ipr': 'debug/routing_effective_support_ipr',
    'routing_effective_support_ipr_p10': 'debug/routing_effective_support_ipr_p10',
    'routing_effective_support_ipr_p50': 'debug/routing_effective_support_ipr_p50',
    'routing_effective_support_ipr_p90': 'debug/routing_effective_support_ipr_p90',
    'routing_support_below_2_frac': 'debug/routing_support_below_2_frac',
    'routing_support_below_3_frac': 'debug/routing_support_below_3_frac',
    'routing_support_below_min_frac': 'debug/routing_support_below_min_frac',
    'routing_top1_minus_top2': 'debug/routing_top1_minus_top2',
    'routing_top2_mass': 'debug/routing_top2_mass',
    'routing_top4_mass': 'debug/routing_top4_mass',
    'diag_cos_full': 'debug/diag_cos_full',
    'diag_cos_top1': 'debug/diag_cos_top1',
    'diag_cos_top2': 'debug/diag_cos_top2',
    'diag_cos_top4': 'debug/diag_cos_top4',
    'loss_diag_full': 'debug/loss_diag_full',
    'loss_diag_top1': 'debug/loss_diag_top1',
    'loss_diag_top2': 'debug/loss_diag_top2',
    'loss_diag_top4': 'debug/loss_diag_top4',
    'prototype_active_count_eps_1e-3': 'debug/prototype_active_count_eps_1e-3',
    'prototype_active_count_eps_1e-2': 'debug/prototype_active_count_eps_1e-2',
    'prototype_active_count_eps_1e-3_window_100': 'debug/prototype_active_count_eps_1e-3_window_100',
    'prototype_active_count_eps_1e-3_window_500': 'debug/prototype_active_count_eps_1e-3_window_500',
    'prototype_active_count_eps_1e-2_window_100': 'debug/prototype_active_count_eps_1e-2_window_100',
    'prototype_active_count_eps_1e-2_window_500': 'debug/prototype_active_count_eps_1e-2_window_500',
    'prototype_usage_entropy_window_100': 'debug/prototype_usage_entropy_window_100',
    'prototype_usage_entropy_window_500': 'debug/prototype_usage_entropy_window_500',
    'prototype_usage_max_window_100': 'debug/prototype_usage_max_window_100',
    'prototype_usage_max_window_500': 'debug/prototype_usage_max_window_500',
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
    'image_surrogate_positive_cosine_mean': 'debug/image_surrogate_positive_cosine_mean',
    'image_surrogate_positive_cosine_std': 'debug/image_surrogate_positive_cosine_std',
    'image_surrogate_hardest_negative_cosine_mean': 'debug/image_surrogate_hardest_negative_cosine_mean',
    'image_surrogate_hardest_negative_cosine_std': 'debug/image_surrogate_hardest_negative_cosine_std',
    'image_surrogate_margin_mean': 'debug/image_surrogate_margin_mean',
    'image_surrogate_margin_min': 'debug/image_surrogate_margin_min',
    'image_surrogate_positive_logit_mean': 'debug/image_surrogate_positive_logit_mean',
    'image_surrogate_hardest_negative_logit_mean': 'debug/image_surrogate_hardest_negative_logit_mean',
    'image_exact_positive_cosine_mean': 'debug/image_exact_positive_cosine_mean',
    'image_exact_positive_cosine_std': 'debug/image_exact_positive_cosine_std',
    'image_exact_hardest_negative_cosine_mean': 'debug/image_exact_hardest_negative_cosine_mean',
    'image_exact_hardest_negative_cosine_std': 'debug/image_exact_hardest_negative_cosine_std',
    'image_exact_margin_mean': 'debug/image_exact_margin_mean',
    'image_exact_margin_min': 'debug/image_exact_margin_min',
    'image_exact_positive_logit_mean': 'debug/image_exact_positive_logit_mean',
    'image_exact_hardest_negative_logit_mean': 'debug/image_exact_hardest_negative_logit_mean',
    'surrogate_pairwise_positive_cosine_mean': 'debug/surrogate_pairwise_positive_cosine_mean',
    'surrogate_pairwise_positive_cosine_std': 'debug/surrogate_pairwise_positive_cosine_std',
    'surrogate_pairwise_hardest_negative_cosine_mean': 'debug/surrogate_pairwise_hardest_negative_cosine_mean',
    'surrogate_pairwise_hardest_negative_cosine_std': 'debug/surrogate_pairwise_hardest_negative_cosine_std',
    'surrogate_pairwise_margin_mean': 'debug/surrogate_pairwise_margin_mean',
    'surrogate_pairwise_margin_min': 'debug/surrogate_pairwise_margin_min',
    'surrogate_pairwise_positive_logit_mean': 'debug/surrogate_pairwise_positive_logit_mean',
    'surrogate_pairwise_hardest_negative_logit_mean': 'debug/surrogate_pairwise_hardest_negative_logit_mean',
    'surrogate_pairwise_logit_mean': 'debug/surrogate_pairwise_logit_mean',
    'surrogate_pairwise_logit_std': 'debug/surrogate_pairwise_logit_std',
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
    'surrogate_retrieval_grad_norm': 'debug/surrogate_retrieval_grad_norm',
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
    'surrogate_t_pool_norm': 'debug/surrogate_t_pool_norm',
    'exact_t_pool_norm': 'debug/exact_t_pool_norm',
    'image_feature_norm': 'debug/image_feature_norm',
    'image_embed_norm_raw': 'debug/image_embed_norm_raw',
    'image_embed_unit_norm': 'debug/image_embed_unit_norm',
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
    'surrogate_pairwise_logits',
    'surrogate_retrieval_logits',
    'basis_token_scores',
    'basis_token_weights',
    'basis_beta_logits_masked',
    'projected_features',
    'projected_features_norm',
    'projected_features_pre_norm',
    'projected_features_raw',
    'prototype_usage',
    't_pool_norm',
    'image_embed_norm',
    'text_embed_norm',
    'text_embed_norm_std',
    'text_embed_norm_min',
    'text_embed_norm_max',
    'surrogate_text_embed_norm',
    'surrogate_text_embed_norm_std',
    'surrogate_text_embed_norm_min',
    'surrogate_text_embed_norm_max',
    'exact_text_embed_norm',
    'exact_text_embed_norm_std',
    'exact_text_embed_norm_min',
    'exact_text_embed_norm_max',
    'contrastive_logits',
}

TRACKED_SCALAR_KEYS = TRAIN_LOSS_KEYS + tuple(sorted(set(DEBUG_METRIC_MAP.keys())))


class RoutingCoverageTracker:
    """Tracks rolling routing coverage and epoch summaries for logging only."""

    def __init__(
        self,
        window_sizes: Tuple[int, ...] = (100, 500),
        activity_epsilons: Tuple[float, ...] = (1e-3, 1e-2),
    ):
        self.window_sizes = tuple(int(size) for size in window_sizes)
        self.activity_epsilons = tuple(float(epsilon) for epsilon in activity_epsilons)
        self.num_prototypes: Optional[int] = None
        self._top1_windows: Dict[int, Deque[torch.Tensor]] = {size: deque() for size in self.window_sizes}
        self._mean_usage_windows: Dict[int, Deque[torch.Tensor]] = {size: deque() for size in self.window_sizes}
        self._top1_window_sums: Dict[int, Optional[torch.Tensor]] = {size: None for size in self.window_sizes}
        self._mean_usage_window_sums: Dict[int, Optional[torch.Tensor]] = {size: None for size in self.window_sizes}
        self._epoch_top1_counts: Optional[torch.Tensor] = None
        self._epoch_mean_usage_sum: Optional[torch.Tensor] = None
        self._epoch_top3_seen: Optional[torch.Tensor] = None
        self._epoch_batch_count = 0

    @staticmethod
    def _epsilon_label(epsilon: float) -> str:
        return format(epsilon, '.0e').replace('e-0', 'e-').replace('e+0', 'e+')

    def _ensure_initialized(self, num_prototypes: int) -> None:
        if self.num_prototypes is None:
            self.num_prototypes = int(num_prototypes)
            for size in self.window_sizes:
                self._top1_window_sums[size] = torch.zeros(self.num_prototypes, dtype=torch.float32)
                self._mean_usage_window_sums[size] = torch.zeros(self.num_prototypes, dtype=torch.float32)
            self.reset_epoch()
            return
        if self.num_prototypes != int(num_prototypes):
            raise ValueError(
                f'RoutingCoverageTracker expected {self.num_prototypes} prototypes but received {num_prototypes}.'
            )

    def reset_epoch(self) -> None:
        if self.num_prototypes is None:
            self._epoch_top1_counts = None
            self._epoch_mean_usage_sum = None
            self._epoch_top3_seen = None
            self._epoch_batch_count = 0
            return
        self._epoch_top1_counts = torch.zeros(self.num_prototypes, dtype=torch.float32)
        self._epoch_mean_usage_sum = torch.zeros(self.num_prototypes, dtype=torch.float32)
        self._epoch_top3_seen = torch.zeros(self.num_prototypes, dtype=torch.bool)
        self._epoch_batch_count = 0

    def _update_window(self, window: Deque[torch.Tensor], window_sum: torch.Tensor, value: torch.Tensor, size: int) -> None:
        if len(window) == size:
            window_sum.sub_(window.popleft())
        window.append(value)
        window_sum.add_(value)

    def update(self, alpha: torch.Tensor) -> None:
        if not isinstance(alpha, torch.Tensor) or alpha.ndim != 2:
            raise ValueError('RoutingCoverageTracker.update expects alpha with shape [B, N].')

        detached_alpha = alpha.detach().float()
        num_prototypes = detached_alpha.size(1)
        self._ensure_initialized(num_prototypes)

        top1_assignments = detached_alpha.argmax(dim=-1)
        top1_histogram = torch.bincount(top1_assignments, minlength=num_prototypes).to(dtype=torch.float32).cpu()
        mean_usage = detached_alpha.mean(dim=0).to(dtype=torch.float32).cpu()
        topk = min(3, num_prototypes)
        topk_indices = torch.topk(detached_alpha, k=topk, dim=-1).indices.reshape(-1).cpu().unique()

        for size in self.window_sizes:
            self._update_window(self._top1_windows[size], self._top1_window_sums[size], top1_histogram.clone(), size)
            self._update_window(self._mean_usage_windows[size], self._mean_usage_window_sums[size], mean_usage.clone(), size)

        self._epoch_top1_counts.add_(top1_histogram)
        self._epoch_mean_usage_sum.add_(mean_usage)
        self._epoch_top3_seen[topk_indices] = True
        self._epoch_batch_count += 1

    @staticmethod
    def _normalized_distribution(values: torch.Tensor) -> torch.Tensor:
        return values / values.sum().clamp_min(1e-12)

    @classmethod
    def _distribution_entropy(cls, values: torch.Tensor) -> float:
        probabilities = cls._normalized_distribution(values)
        return float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item())

    @classmethod
    def _distribution_max(cls, values: torch.Tensor) -> float:
        probabilities = cls._normalized_distribution(values)
        return float(probabilities.max().item())

    def get_debug_metrics(self) -> Dict[str, float]:
        if self.num_prototypes is None:
            return {}

        metrics = {}
        for size in self.window_sizes:
            # Windowed top-1 coverage shows whether sparse winners rotate across recent batches.
            top1_sum = self._top1_window_sums[size]
            active_count = float((top1_sum > 0).sum().item())
            metrics[f'routing_top1_active_count_window_{size}'] = active_count
            metrics[f'routing_top1_dead_count_window_{size}'] = float(self.num_prototypes - active_count)
            metrics[f'routing_top1_usage_entropy_window_{size}'] = self._distribution_entropy(top1_sum)
            metrics[f'routing_top1_usage_max_window_{size}'] = self._distribution_max(top1_sum)

            # Windowed mean-usage coverage catches diffuse cross-batch activity beyond top-1 winners.
            window_length = max(len(self._mean_usage_windows[size]), 1)
            mean_usage = self._mean_usage_window_sums[size] / float(window_length)
            metrics[f'prototype_usage_entropy_window_{size}'] = self._distribution_entropy(mean_usage)
            metrics[f'prototype_usage_max_window_{size}'] = self._distribution_max(mean_usage)
            for epsilon in self.activity_epsilons:
                epsilon_label = self._epsilon_label(epsilon)
                metrics[f'prototype_active_count_eps_{epsilon_label}_window_{size}'] = float((mean_usage > epsilon).sum().item())
        return metrics

    def flush_epoch_metrics(self, epoch: int) -> Dict[str, float]:
        metrics = {'train_epoch/epoch': float(epoch)}
        if self.num_prototypes is None or self._epoch_batch_count == 0:
            return metrics

        # Epoch winner coverage separates rotating sparse usage from persistent hub collapse.
        active_count = float((self._epoch_top1_counts > 0).sum().item())
        metrics['train_epoch/routing_top1_active_count'] = active_count
        metrics['train_epoch/routing_top1_dead_count'] = float(self.num_prototypes - active_count)
        metrics['train_epoch/routing_top1_usage_entropy'] = self._distribution_entropy(self._epoch_top1_counts)
        metrics['train_epoch/routing_top1_usage_max'] = self._distribution_max(self._epoch_top1_counts)

        # Epoch-averaged usage summarizes how much of the prototype bank stayed active overall.
        epoch_mean_usage = self._epoch_mean_usage_sum / float(self._epoch_batch_count)
        metrics['train_epoch/prototype_usage_entropy'] = self._distribution_entropy(epoch_mean_usage)
        metrics['train_epoch/prototype_usage_max'] = self._distribution_max(epoch_mean_usage)
        for epsilon in self.activity_epsilons:
            epsilon_label = self._epsilon_label(epsilon)
            metrics[f'train_epoch/prototype_active_count_eps_{epsilon_label}'] = float((epoch_mean_usage > epsilon).sum().item())
        metrics['train_epoch/routing_top3_active_count'] = float(self._epoch_top3_seen.sum().item())
        return metrics


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


def build_validation_metrics(
    epoch: int,
    evaluator=None,
    loss_metrics: Optional[Dict[str, float]] = None,
    val_loss: Optional[float] = None,
) -> Dict[str, float]:
    metrics = {
        'val/epoch': float(epoch),
    }
    merged_loss_metrics = dict(loss_metrics or {})
    if val_loss is not None and 'loss_total' not in merged_loss_metrics:
        merged_loss_metrics['loss_total'] = float(val_loss)
    for key, value in merged_loss_metrics.items():
        metrics[f'val/{key}'] = float(value)
    if evaluator is not None and getattr(evaluator, 'latest_metrics', None):
        metrics.update(evaluator.latest_metrics)
    return metrics






