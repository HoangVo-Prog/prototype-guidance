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
    'loss_proto',
    'loss_host_weighted',
    'lambda_host',
    'loss_proxy',
    'loss_proxy_image',
    'loss_proxy_text',
    'loss_proxy_text_exact',
    'loss_ret',
    'loss_semantic_pbt',
    'loss_weight_ret',
    'loss_align',
    'loss_diag',
    'loss_gap',
    'loss_support',
    'loss_diversity',
    'loss_balance',
    'loss_proxy_image_weighted',
    'loss_proxy_text_weighted',
    'loss_proxy_text_exact_weighted',
    'loss_proxy_weighted',
    'loss_ret_weighted',
    'loss_semantic_pbt_weighted',
    'loss_weight_ret_weighted',
    'loss_align_weighted',
    'loss_gap_weighted',
    'loss_diag_weighted',
    'loss_support_weighted',
    'loss_diversity_weighted',
    'loss_balance_weighted',
)


DEBUG_METRIC_MAP = {
    'logit_scale': 'debug/logit_scale',
    'host_logit_scale': 'debug/host_logit_scale',
    'host_retrieval_temperature': 'debug/host_retrieval_temperature',
    'host_loss_total': 'debug/host_loss_total',
    'host_loss_ret': 'debug/host_loss_ret',
    'proxy_temperature': 'debug/proxy_temperature',
    'diag_temperature': 'debug/diag_temperature',
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
    'routing_source_mode': 'debug/routing_source_mode',
    'local_routing_entropy': 'debug/local_routing_entropy',
    'local_routing_max_mean': 'debug/local_routing_max_mean',
    'local_routing_effective_support': 'debug/local_routing_effective_support',
    'routing_effective_support_mean': 'debug/routing_effective_support_mean',
    'routing_effective_support_std': 'debug/routing_effective_support_std',
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
    'routing_proto_label_nmi': 'debug/routing_proto_label_nmi',
    'diag_cos_full': 'debug/diag_cos_full',
    'diag_cos_top1': 'debug/diag_cos_top1',
    'diag_cos_top2': 'debug/diag_cos_top2',
    'diag_cos_top4': 'debug/diag_cos_top4',
    'diag_pos_cosine_mean': 'debug/diag_pos_cosine_mean',
    'diag_hardneg_cosine_mean': 'debug/diag_hardneg_cosine_mean',
    'diag_gap_margin_mean': 'debug/diag_gap_margin_mean',
    'diag_student_teacher_pos_mean': 'debug/diag_student_teacher_pos_mean',
    'diag_student_teacher_offdiag_mean': 'debug/diag_student_teacher_offdiag_mean',
    'diag_student_teacher_margin': 'debug/diag_student_teacher_margin',
    'loss_diag_row': 'debug/loss_diag_row',
    'loss_diag_col': 'debug/loss_diag_col',
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
    'host_margin_mean': 'debug/host_margin_mean',
    'host_margin_min': 'debug/host_margin_min',
    'host_weight_mean': 'debug/host_weight_mean',
    'host_weight_std': 'debug/host_weight_std',
    'proto_score_mean': 'debug/proto_score_mean',
    'proto_diag_mean': 'debug/proto_diag_mean',
    'proto_host_score_corr': 'debug/proto_host_score_corr',
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
    'prototype_cosine_offdiag_max': 'debug/prototype_cosine_offdiag_max',
    'prototype_high_similarity_pair_ratio': 'debug/prototype_high_similarity_pair_ratio',
    'prototype_assignment_overlap_mean': 'debug/prototype_assignment_overlap_mean',
    'prototype_method_role_semantic_structure': 'debug/prototype_method_role_semantic_structure',
    'prototype_semantic_enabled': 'debug/prototype_semantic_enabled',
    'semantic_structure_enabled': 'debug/semantic_structure_enabled',
    'prototype_source_recomputed': 'debug/prototype_source_recomputed',
    'semantic_recompute_count': 'debug/semantic_recompute_count',
    'semantic_recompute_last_epoch': 'debug/semantic_recompute_last_epoch',
    'semantic_recompute_last_step': 'debug/semantic_recompute_last_step',
    'semantic_recompute_triggered': 'debug/semantic_recompute_triggered',
    'semantic_recompute_skipped_no_features': 'debug/semantic_recompute_skipped_no_features',
    'semantic_empty_cluster_reseed_events': 'debug/semantic_empty_cluster_reseed_events',
    'semantic_active_cluster_count': 'debug/semantic_active_cluster_count',
    'semantic_empty_cluster_count': 'debug/semantic_empty_cluster_count',
    'semantic_assignment_entropy_image': 'debug/semantic_assignment_entropy_image',
    'semantic_assignment_entropy_teacher': 'debug/semantic_assignment_entropy_teacher',
    'semantic_target_entropy': 'debug/semantic_target_entropy',
    'semantic_pbt_valid_cluster_count': 'debug/semantic_pbt_valid_cluster_count',
    'semantic_pbt_empty_cluster_count': 'debug/semantic_pbt_empty_cluster_count',
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

PROTOTYPE_GEOMETRY_METRIC_KEYS = (
    'prototype_pairwise_cosine_mean',
    'prototype_pairwise_cosine_std',
    'prototype_pairwise_cosine_max',
    'contextualized_prototype_pairwise_cosine_mean',
    'contextualized_prototype_pairwise_cosine_std',
    'contextualized_prototype_pairwise_cosine_max',
)

_LOSS_BASE_SUFFIX_MAP = {
    'loss_total': 'total',
    'loss_host': 'host',
    'loss_host_ret': 'host_ret',
    'loss_host_ret_i2t': 'host_ret_i2t',
    'loss_host_ret_t2i': 'host_ret_t2i',
    'loss_host_cid': 'host_cid',
    'loss_proto_total': 'proto_total',
    'loss_proto': 'proto',
    'loss_proxy': 'proxy',
    'loss_proxy_image': 'proxy_image',
    'loss_proxy_text': 'proxy_text',
    'loss_proxy_text_exact': 'proxy_text_exact',
    'loss_ret': 'ret',
    'loss_semantic_pbt': 'semantic_pbt',
    'loss_weight_ret': 'weight_ret',
    'loss_align': 'align',
    'loss_dir': 'dir',
    'loss_gap': 'gap',
    'loss_sup': 'sup',
    'loss_diag': 'diag',
    'loss_support': 'support',
    'loss_diversity': 'diversity',
    'loss_balance': 'balance',
}

_LOSS_WEIGHTED_SUFFIX_MAP = {
    'loss_host_weighted': 'host',
    'loss_proxy_image_weighted': 'proxy_image',
    'loss_proxy_text_weighted': 'proxy_text',
    'loss_proxy_text_exact_weighted': 'proxy_text_exact',
    'loss_proxy_weighted': 'proxy',
    'loss_ret_weighted': 'ret',
    'loss_semantic_pbt_weighted': 'semantic_pbt',
    'loss_weight_ret_weighted': 'weight_ret',
    'loss_align_weighted': 'align',
    'loss_dir_weighted': 'dir',
    'loss_gap_weighted': 'gap',
    'loss_sup_weighted': 'sup',
    'loss_diag_weighted': 'diag',
    'loss_support_weighted': 'support',
    'loss_diversity_weighted': 'diversity',
    'loss_balance_weighted': 'balance',
}

_TRAIN_MODEL_KEYS = {
    'logit_scale',
    'host_logit_scale',
    'host_retrieval_temperature',
    'proxy_temperature',
    'diag_temperature',
    'retrieval_temperature',
    'host_loss_total',
    'host_loss_ret',
    'lambda_host',
}

_TRAIN_TOKEN_POOL_KEYS = {
    'token_pool_entropy',
    'beta_max_prob',
    'token_special_mass',
    'token_valid_fraction',
    'valid_token_fraction',
}

_TRAIN_NORM_KEYS = {
    'image_embed_norm_std',
    'image_embed_norm_min',
    'image_embed_norm_max',
    'text_embed_norm_std',
    'text_embed_norm_min',
    'text_embed_norm_max',
    'q_norm',
    'surrogate_t_pool_norm',
    'exact_t_pool_norm',
    'image_feature_norm',
    'image_embed_norm_raw',
    'image_embed_unit_norm',
    'surrogate_text_embed_norm_raw',
    'surrogate_text_embed_unit_norm',
    'exact_text_embed_norm_raw',
    'exact_text_embed_unit_norm',
}

_VAL_DEBUG_NAMESPACE_MAP = {
    'eval_positive_gallery_count_min': 'val/data/positive_gallery_count_min',
    'eval_positive_gallery_count_mean': 'val/data/positive_gallery_count_mean',
    'eval_logit_scale': 'val/model/logit_scale',
    'eval_retrieval_temperature': 'val/model/retrieval_temperature',
    'eval_positive_exact_cosine_mean': 'val/geometry/exact_positive_cosine_mean',
    'eval_hardest_negative_exact_cosine_mean': 'val/geometry/exact_hardest_negative_cosine_mean',
    'eval_exact_margin_mean': 'val/geometry/exact_margin_mean',
    'eval_positive_exact_pair_cosine_mean': 'val/geometry/exact_pair_cosine_mean',
    'eval_image_projected_norm_mean': 'val/norm/image_projected_norm_mean',
    'eval_image_projected_norm_std': 'val/norm/image_projected_norm_std',
    'eval_positive_exact_text_embed_norm_mean': 'val/norm/exact_text_embed_norm_mean',
    'eval_positive_exact_text_embed_norm_std': 'val/norm/exact_text_embed_norm_std',
    'eval_positive_exact_text_embed_unit_norm_mean': 'val/norm/exact_text_embed_unit_norm_mean',
}


def _map_train_loss_key(raw_key: str) -> str:
    if raw_key == 'loss_weight_ret':
        return 'train/loss_weight_ret'
    if raw_key == 'loss_weight_ret_weighted':
        return 'train/loss_weight_ret_weighted'
    if raw_key in _LOSS_BASE_SUFFIX_MAP:
        return f'train/loss/{_LOSS_BASE_SUFFIX_MAP[raw_key]}'
    if raw_key in _LOSS_WEIGHTED_SUFFIX_MAP:
        return f'train/loss_weighted/{_LOSS_WEIGHTED_SUFFIX_MAP[raw_key]}'
    if raw_key.startswith('loss_'):
        return f'train/loss/{raw_key[len("loss_"):]}'
    return f'train/model/{raw_key}'


def _map_val_loss_key(raw_key: str) -> str:
    if raw_key == 'loss_weight_ret':
        return 'val/loss_weight_ret'
    if raw_key == 'loss_weight_ret_weighted':
        return 'val/loss_weight_ret_weighted'
    if raw_key in _LOSS_BASE_SUFFIX_MAP:
        return f'val/loss/{_LOSS_BASE_SUFFIX_MAP[raw_key]}'
    if raw_key in _LOSS_WEIGHTED_SUFFIX_MAP:
        return f'val/loss_weighted/{_LOSS_WEIGHTED_SUFFIX_MAP[raw_key]}'
    if raw_key.startswith('loss_'):
        return f'val/loss/{raw_key[len("loss_"):]}'
    return f'val/model/{raw_key}'


def map_train_diagnostic_key(raw_key: str) -> str:
    if raw_key in {
        'routing_source_mode',
        'local_routing_entropy',
        'local_routing_max_mean',
        'local_routing_effective_support',
    }:
        return f'train/{raw_key}'
    if raw_key in {
        'host_margin_mean',
        'host_margin_min',
        'host_weight_mean',
        'host_weight_std',
        'proto_score_mean',
        'proto_diag_mean',
        'proto_host_score_corr',
    }:
        return f'train/{raw_key}'
    if raw_key in _TRAIN_MODEL_KEYS:
        return f'train/model/{raw_key}'
    if raw_key in _TRAIN_TOKEN_POOL_KEYS:
        return f'train/token_pool/{raw_key}'
    if raw_key in _TRAIN_NORM_KEYS:
        return f'train/norm/{raw_key}'
    if raw_key.startswith('prototype_usage_') or raw_key == 'prototype_dead_count' or raw_key.startswith('prototype_active_count_'):
        return f'train/prototype_usage/{raw_key}'
    if raw_key == 'prototype_assignment_entropy':
        return f'train/prototype_usage/{raw_key}'
    if raw_key.startswith('routing_top1_usage_') or raw_key == 'routing_top1_dead_count' or raw_key.startswith('routing_top1_dead_count_window_') or raw_key.startswith('routing_top1_active_count_window_'):
        return f'train/prototype_usage/{raw_key}'
    if raw_key.startswith('routing_'):
        return f'train/routing/{raw_key}'
    if raw_key.startswith('diag_cos_') or raw_key.startswith('loss_diag_') or raw_key.startswith('diag_student_teacher_'):
        return f'train/fidelity/{raw_key}'
    if raw_key.startswith('image_surrogate_') or raw_key.startswith('image_exact_') or raw_key.startswith('surrogate_pairwise_'):
        return f'train/geometry/{raw_key}'
    if raw_key.startswith('image_proxy_') or raw_key.startswith('text_proxy_') or raw_key.startswith('text_exact_proxy_') or raw_key.startswith('image_positive_proxy_') or raw_key.startswith('image_hardest_negative_proxy_') or raw_key.startswith('text_positive_proxy_') or raw_key.startswith('text_hardest_negative_proxy_') or raw_key.startswith('class_proxy_norm_'):
        return f'train/proxy/{raw_key}'
    if raw_key.startswith('prototype_pairwise_') or raw_key.startswith('contextualized_prototype_pairwise_') or raw_key == 'prototype_contextualization_entropy':
        return f'train/prototype_geometry/{raw_key}'
    if raw_key in {
        'prototype_cosine_offdiag_max',
        'prototype_high_similarity_pair_ratio',
        'prototype_assignment_overlap_mean',
    }:
        return f'train/prototype/{raw_key}'
    if raw_key.startswith('semantic_') or raw_key in {
        'prototype_source_recomputed',
        'prototype_semantic_enabled',
        'semantic_structure_enabled',
        'prototype_method_role_semantic_structure',
    }:
        return f'train/semantic/{raw_key}'
    if raw_key.startswith('grad_norm_') or raw_key == 'surrogate_retrieval_grad_norm':
        return f'train/grad/{raw_key}'
    return f'train/model/{raw_key}'


def map_train_scalar_to_wandb_key(raw_key: str) -> str:
    if raw_key in TRAIN_LOSS_KEYS:
        return _map_train_loss_key(raw_key)
    return map_train_diagnostic_key(raw_key)


def map_validation_metric_key(metric_key: str) -> str:
    if metric_key.startswith('val/pas/'):
        return f'val/retrieval/{metric_key.split("/", 2)[2]}'
    if metric_key.startswith('val/debug/'):
        debug_key = metric_key.split('/', 2)[2]
        return _VAL_DEBUG_NAMESPACE_MAP.get(debug_key, f'val/model/{debug_key}')
    return metric_key


def build_validation_debug_metrics(raw_debug_metrics: Dict[str, float]) -> Dict[str, float]:
    mapped = {}
    for raw_key, value in raw_debug_metrics.items():
        mapped_key = _VAL_DEBUG_NAMESPACE_MAP.get(raw_key, f'val/model/{raw_key}')
        mapped[mapped_key] = float(value)
    return mapped


def build_validation_retrieval_metrics(retrieval_metrics: Dict[str, float]) -> Dict[str, float]:
    return {f'val/retrieval/{metric_name}': float(metric_value) for metric_name, metric_value in retrieval_metrics.items()}


def collect_metric_namespace_collisions() -> Dict[str, Tuple[str, ...]]:
    tracked_keys = set(TRAIN_LOSS_KEYS) | set(DEBUG_METRIC_MAP.keys()) | set(PROTOTYPE_GEOMETRY_METRIC_KEYS)
    mapped_to_sources: Dict[str, list] = {}
    for raw_key in sorted(tracked_keys):
        mapped_key = map_train_scalar_to_wandb_key(raw_key)
        mapped_to_sources.setdefault(mapped_key, []).append(raw_key)
    return {
        mapped_key: tuple(source_keys)
        for mapped_key, source_keys in mapped_to_sources.items()
        if len(source_keys) > 1
    }


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


def build_train_metrics_from_scalars(
    epoch: int,
    step: Optional[int],
    scalar_metrics: Dict[str, float],
    lr: float,
) -> Dict[str, float]:
    metrics = {
        'train/epoch': float(epoch),
        'train/lr': float(lr),
    }
    if step is not None:
        metrics['train/step'] = float(step)

    for key, value in scalar_metrics.items():
        try:
            scalar_value = float(value)
        except (TypeError, ValueError):
            continue
        metrics[map_train_scalar_to_wandb_key(key)] = scalar_value
    return metrics


def build_train_metrics(epoch: int, step: Optional[int], outputs: Dict[str, object], lr: float, include_debug_metrics: bool = True) -> Dict[str, float]:
    scalar_metrics = collect_scalar_metrics(outputs, include_debug_metrics=include_debug_metrics)
    return build_train_metrics_from_scalars(
        epoch=epoch,
        step=step,
        scalar_metrics=scalar_metrics,
        lr=lr,
    )


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
        metrics[_map_val_loss_key(key)] = float(value)
    if evaluator is not None and getattr(evaluator, 'latest_metrics', None):
        for metric_key, metric_value in evaluator.latest_metrics.items():
            metrics[map_validation_metric_key(metric_key)] = metric_value
    return metrics
