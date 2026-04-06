import copy
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from utils.precision import AMP_DTYPE_ALIASES, BACKBONE_PRECISION_ALIASES, PROTOTYPE_PRECISION_ALIASES


PRIMARY_CONFIG_KEY_MAP: Dict[Tuple[str, ...], str] = {
    ('experiment', 'name'): 'name',
    ('experiment', 'output_dir'): 'output_dir',
    ('experiment', 'seed'): 'seed',
    ('experiment', 'local_rank'): 'local_rank',

    ('model', 'name'): 'model_name',
    ('model', 'variant'): 'model_variant',
    ('model', 'training_mode'): 'training_mode',
    ('model', 'pretrain_choice'): 'pretrain_choice',
    ('model', 'image_backbone'): 'image_backbone',
    ('model', 'text_backbone'): 'text_backbone',
    ('model', 'embedding_dim'): 'embedding_dim',
    ('model', 'projection_dim'): 'projection_dim',
    ('model', 'projector_hidden_dim'): 'projector_hidden_dim',
    ('model', 'projector_dropout'): 'projector_dropout',
    ('model', 'projector_type'): 'projector_type',
    ('model', 'normalize_projector_outputs'): 'normalize_projector_outputs',
    ('model', 'use_custom_projector'): 'use_custom_projector',
    ('model', 'backbone_precision'): 'backbone_precision',
    ('model', 'prototype_precision'): 'prototype_precision',
    ('model', 'temperature'): 'temperature',
    ('model', 'img_size'): 'img_size',
    ('model', 'stride_size'): 'stride_size',
    ('model', 'text_length'): 'text_length',
    ('model', 'vocab_size'): 'vocab_size',
    ('model', 'use_prototype_branch'): 'use_prototype_branch',
    ('model', 'use_prototype_bank'): 'use_prototype_bank',
    ('model', 'use_image_conditioned_pooling'): 'use_image_conditioned_pooling',
    ('host', 'type'): 'host_type',
    ('host', 'enabled'): 'use_host_loss',
    ('host', 'loss_weight'): 'lambda_host',
    ('host', 'use_custom_projector'): 'use_custom_projector',
    ('host', 'freeze_projectors'): 'freeze_host_projectors',
    ('host', 'itself_loss_names'): 'itself_loss_names',
    ('host', 'itself_only_global'): 'itself_only_global',
    ('host', 'itself_select_ratio'): 'itself_select_ratio',
    ('host', 'itself_grab_embed_dim'): 'itself_grab_embed_dim',
    ('host', 'itself_score_weight_global'): 'itself_score_weight_global',
    ('host', 'itself_tau'): 'itself_tau',
    ('host', 'itself_margin'): 'itself_margin',
    ('host', 'itself_return_all'): 'itself_return_all',
    ('host', 'itself_topk_type'): 'itself_topk_type',
    ('host', 'itself_layer_index'): 'itself_layer_index',
    ('host', 'itself_average_attn_weights'): 'itself_average_attn_weights',
    ('host', 'itself_modify_k'): 'itself_modify_k',
    ('model', 'return_debug_outputs'): 'return_debug_outputs',

    ('prototype', 'num_prototypes'): 'prototype_num_prototypes',
    ('prototype', 'prototype_dim'): 'prototype_dim',
    ('prototype', 'prototype_init'): 'prototype_init',
    ('prototype', 'prototype_init_path'): 'prototype_init_path',
    ('prototype', 'prototype_init_hybrid_ratio'): 'prototype_init_hybrid_ratio',
    ('prototype', 'prototype_init_max_iters'): 'prototype_init_max_iters',
    ('prototype', 'prototype_init_tol'): 'prototype_init_tol',
    ('prototype', 'prototype_init_seed'): 'prototype_init_seed',
    ('prototype', 'routing_type'): 'prototype_routing_type',
    ('prototype', 'routing_temperature'): 'prototype_temperature',
    ('prototype', 'contextualization_enabled'): 'prototype_contextualization_enabled',
    ('prototype', 'contextualization_type'): 'prototype_contextualization_type',
    ('prototype', 'contextualization_residual'): 'prototype_contextualization_residual',
    ('prototype', 'normalize_for_self_interaction'): 'normalize_for_self_interaction',
    ('prototype', 'normalize_for_routing'): 'normalize_for_routing',
    ('prototype', 'dead_prototype_threshold'): 'prototype_dead_threshold',
    ('fusion', 'enabled'): 'fusion_enabled',
    ('fusion', 'coefficient'): 'fusion_coefficient',
    ('fusion', 'coefficient_source'): 'fusion_coefficient_source',
    ('objectives', 'use_host_loss'): 'use_host_loss',
    ('objectives', 'lambda_host'): 'lambda_host',
    ('objectives', 'use_proto_loss_ret'): 'use_loss_ret',
    ('objectives', 'lambda_proto_ret'): 'lambda_ret',
    ('objectives', 'use_diag_fidelity'): 'use_loss_diag',
    ('objectives', 'lambda_diag'): 'lambda_diag',
    ('objectives', 'use_diversity'): 'use_diversity_loss',
    ('objectives', 'lambda_diversity'): 'diversity_loss_weight',
    ('objectives', 'use_balance'): 'use_balancing_loss',
    ('objectives', 'lambda_balance'): 'prototype_balance_loss_weight',

    ('loss', 'lambda_proxy'): 'lambda_proxy',
    ('loss', 'lambda_proxy_image'): 'lambda_proxy_image',
    ('loss', 'lambda_proxy_text'): 'lambda_proxy_text',
    ('loss', 'lambda_proxy_text_exact'): 'lambda_proxy_text_exact',
    ('loss', 'use_loss_proxy_image'): 'use_loss_proxy_image',
    ('loss', 'use_loss_proxy_text'): 'use_loss_proxy_text',
    ('loss', 'use_loss_proxy_text_exact'): 'use_loss_proxy_text_exact',
    ('loss', 'use_loss_align'): 'use_loss_align',
    ('loss', 'lambda_align'): 'lambda_align',
    ('loss', 'use_loss_diag'): 'use_loss_diag',
    ('loss', 'lambda_diag'): 'lambda_diag',
    ('loss', 'use_loss_ret'): 'use_loss_ret',
    ('loss', 'retrieval_mode'): 'retrieval_mode',
    ('loss', 'lambda_ret'): 'lambda_ret',
    ('loss', 'use_loss_support'): 'use_loss_support',
    ('loss', 'lambda_support'): 'lambda_support',
    ('loss', 'support_min'): 'support_min',
    ('loss', 'use_balancing_loss'): 'use_balancing_loss',
    ('loss', 'balance_loss_weight'): 'prototype_balance_loss_weight',
    ('loss', 'use_diversity_loss'): 'use_diversity_loss',
    ('loss', 'diversity_loss_weight'): 'diversity_loss_weight',

    ('text_pooling', 'token_policy'): 'token_policy',
    ('text_pooling', 'scoring_type'): 'token_scoring_type',
    ('text_pooling', 'normalize_for_token_scoring'): 'normalize_for_token_scoring',
    ('text_pooling', 'token_temperature'): 'token_pooling_temperature',
    ('text_pooling', 'special_token_ids'): 'special_token_ids',
    ('text_pooling', 'error_on_empty_kept_tokens'): 'error_on_empty_kept_tokens',

    ('training', 'batch_size'): 'batch_size',
    ('training', 'epochs'): 'num_epoch',
    ('training', 'log_period'): 'log_period',
    ('training', 'eval_frequency'): 'eval_period',
    ('training', 'save_interval'): 'save_interval',
    ('training', 'resume'): 'resume',
    ('training', 'resume_ckpt_file'): 'resume_ckpt_file',
    ('training', 'finetune'): 'finetune',
    ('training', 'pretrain'): 'pretrain',
    ('training', 'img_aug'): 'img_aug',
    ('training', 'txt_aug'): 'txt_aug',
    ('training', 'sampler'): 'sampler',
    ('training', 'num_instance'): 'num_instance',
    ('training', 'num_workers'): 'num_workers',
    ('training', 'training'): 'training',
    ('training', 'stage'): 'training_stage',
    ('training', 'freeze_image_backbone'): 'freeze_image_backbone',
    ('training', 'freeze_text_backbone'): 'freeze_text_backbone',
    ('training', 'freeze_prototype_side'): 'freeze_prototype_side',
    ('training', 'freeze_host_projectors'): 'freeze_host_projectors',
    ('training', 'grad_clip'): 'grad_clip',
    ('training', 'amp'): 'amp',
    ('training', 'amp_dtype'): 'amp_dtype',
    ('training', 'proxy_temperature'): 'proxy_temperature',

    ('optimizer', 'type'): 'optimizer',
    ('optimizer', 'lr'): 'lr',
    ('optimizer', 'lr_prototype_bank'): 'lr_prototype_bank',
    ('optimizer', 'lr_projectors'): 'lr_projectors',
    ('optimizer', 'lr_host_projectors'): 'lr_host_projectors',
    ('optimizer', 'lr_class_proxies'): 'lr_class_proxies',
    ('optimizer', 'lr_image_backbone'): 'lr_image_backbone',
    ('optimizer', 'lr_text_backbone'): 'lr_text_backbone',
    ('optimizer', 'weight_decay'): 'weight_decay',
    ('optimizer', 'weight_decay_prototype_bank'): 'weight_decay_prototype_bank',
    ('optimizer', 'weight_decay_projectors'): 'weight_decay_projectors',
    ('optimizer', 'weight_decay_host_projectors'): 'weight_decay_host_projectors',
    ('optimizer', 'weight_decay_class_proxies'): 'weight_decay_class_proxies',
    ('optimizer', 'weight_decay_image_backbone'): 'weight_decay_image_backbone',
    ('optimizer', 'weight_decay_text_backbone'): 'weight_decay_text_backbone',
    ('optimizer', 'momentum'): 'momentum',
    ('optimizer', 'alpha'): 'alpha',
    ('optimizer', 'beta'): 'beta',
    ('optimizer', 'milestones'): 'milestones',
    ('optimizer', 'gamma'): 'gamma',
    ('optimizer', 'warmup_factor'): 'warmup_factor',
    ('optimizer', 'warmup_epochs'): 'warmup_epochs',
    ('optimizer', 'warmup_method'): 'warmup_method',
    ('optimizer', 'scheduler'): 'lrscheduler',
    ('optimizer', 'target_lr'): 'target_lr',
    ('optimizer', 'power'): 'power',

    ('dataset', 'dataset_name'): 'dataset_name',
    ('dataset', 'root_dir'): 'root_dir',
    ('dataset', 'val_dataset'): 'val_dataset',
    ('dataset', 'num_workers'): 'num_workers',

    ('logging', 'use_wandb'): 'use_wandb',
    ('logging', 'project'): 'wandb_project',
    ('logging', 'entity'): 'wandb_entity',
    ('logging', 'run_name'): 'wandb_run_name',
    ('logging', 'group'): 'wandb_group',
    ('logging', 'mode'): 'wandb_mode',
    ('logging', 'tags'): 'wandb_tags',
    ('logging', 'notes'): 'wandb_notes',
    ('logging', 'log_interval'): 'wandb_log_interval',
    ('logging', 'log_code'): 'wandb_log_code',
    ('logging', 'log_debug_metrics'): 'log_debug_metrics',

    ('evaluation', 'checkpoint_path'): 'checkpoint',
    ('evaluation', 'device'): 'device',
    ('evaluation', 'cross_domain_generalization'): 'cross_domain_generalization',
    ('evaluation', 'target_domain'): 'target_domain',
    ('evaluation', 'retrieval_metrics'): 'retrieval_metrics',
    ('evaluation', 'batch_size'): 'test_batch_size',
    ('evaluation', 'prototype_image_chunk_size'): 'prototype_eval_image_chunk_size',
    ('evaluation', 'prototype_text_chunk_size'): 'prototype_eval_text_chunk_size',
    ('evaluation', 'retrieval_scorer'): 'retrieval_scorer',
}


READ_ALIAS_CONFIG_KEY_MAP: Dict[Tuple[str, ...], str] = {
    ('training', 'num_epoch'): 'num_epoch',
    ('training', 'eval_period'): 'eval_period',
    ('optimizer', 'optimizer'): 'optimizer',
    ('optimizer', 'lrscheduler'): 'lrscheduler',
    ('evaluation', 'checkpoint'): 'checkpoint',
    ('evaluation', 'test_batch_size'): 'test_batch_size',
    ('model', 'projector_output_dim'): 'projection_dim',
    ('model', 'use_prototype_contextualization'): 'use_prototype_contextualization',
    ('prototype', 'routing_similarity'): 'prototype_routing_type',
    ('prototype', 'tau_p'): 'prototype_temperature',
    ('prototype', 'use_balancing_loss'): 'use_balancing_loss',
    ('prototype', 'balance_loss_weight'): 'prototype_balance_loss_weight',
    ('prototype', 'use_diversity_loss'): 'use_diversity_loss',
    ('prototype', 'diversity_loss_weight'): 'diversity_loss_weight',
    ('prototype', 'lambda_bal'): 'prototype_balance_loss_weight',
    ('prototype', 'lambda_div'): 'diversity_loss_weight',
    ('training', 'lambda_proxy'): 'lambda_proxy',
    ('training', 'lambda_proxy_image'): 'lambda_proxy_image',
    ('training', 'lambda_proxy_text'): 'lambda_proxy_text',
    ('training', 'lambda_proxy_text_exact'): 'lambda_proxy_text_exact',
    ('training', 'use_loss_proxy_image'): 'use_loss_proxy_image',
    ('training', 'use_loss_proxy_text'): 'use_loss_proxy_text',
    ('training', 'use_loss_proxy_text_exact'): 'use_loss_proxy_text_exact',
    ('training', 'use_loss_align'): 'use_loss_align',
    ('training', 'lambda_align'): 'lambda_align',
    ('training', 'use_loss_diag'): 'use_loss_diag',
    ('training', 'lambda_diag'): 'lambda_diag',
    ('training', 'use_loss_ret'): 'use_loss_ret',
    ('training', 'lambda_ret'): 'lambda_ret',
    ('text_pooling', 'token_similarity'): 'token_scoring_type',
    ('text_pooling', 'tau_t'): 'token_pooling_temperature',
    ('optimizer', 'weight_decay_prototypes'): 'weight_decay_prototype_bank',
}


SECTION_TEMPLATE = {
    'experiment': {},
    'model': {},
    'host': {},
    'prototype': {},
    'fusion': {},
    'objectives': {},
    'loss': {},
    'text_pooling': {},
    'training': {},
    'optimizer': {},
    'dataset': {},
    'logging': {},
    'evaluation': {},
}


SECTION_KEYS = set(SECTION_TEMPLATE.keys())
SUPPORTED_SPECIAL_TOKEN_ID_KEYS = {
    'bos_token_id',
    'cls_token_id',
    'eos_token_id',
    'pad_token_id',
}
UNSUPPORTED_CONFIG_PATHS = {
    ('model', 'pooling_mode'): 'model.pooling_mode was removed because PAS only supports image-conditioned pooling.',
    ('text_pooling', 'exclude_special_tokens'): 'text_pooling.exclude_special_tokens was removed. Use text_pooling.token_policy.',
    ('text_pooling', 'eos_as_only_token'): 'text_pooling.eos_as_only_token was removed. Use text_pooling.token_policy.',
    ('text_pooling', 'mask_padding_tokens'): 'text_pooling.mask_padding_tokens was removed. Use text_pooling.special_token_ids / attention masks.',
    ('optimizer', 'lr_contextualizer'): 'optimizer.lr_contextualizer was removed because the contextualizer is parameter-free.',
    ('optimizer', 'weight_decay_contextualizer'): 'optimizer.weight_decay_contextualizer was removed because the contextualizer is parameter-free.',
    ('prototype', 'contextualization_num_layers'): 'prototype.contextualization_num_layers is not part of minimal PAS v1; contextualization is a single fixed self-interaction step.',
    ('prototype', 'prototype_normalize'): 'prototype.prototype_normalize was replaced by prototype.normalize_for_self_interaction in PAS v1.',
    ('prototype', 'assignment_sparse'): 'prototype.assignment_sparse is not part of minimal PAS v1; routing is always dense softmax.',
    ('prototype', 'assignment_topk'): 'prototype.assignment_topk is not part of minimal PAS v1; top-k routing is disabled.',
    ('logging', 'log_alpha_entropy'): 'logging.log_alpha_entropy is not a separate runtime toggle in PAS v1. Use logging.log_debug_metrics.',
    ('logging', 'log_beta_entropy'): 'logging.log_beta_entropy is not a separate runtime toggle in PAS v1. Use logging.log_debug_metrics.',
    ('logging', 'log_prototype_usage'): 'logging.log_prototype_usage is not a separate runtime toggle in PAS v1. Use logging.log_debug_metrics.',
    ('logging', 'log_pairwise_prototype_cosine'): 'logging.log_pairwise_prototype_cosine is not a separate runtime toggle in PAS v1. Use logging.log_debug_metrics.',
    ('logging', 'log_top_tokens'): 'logging.log_top_tokens is not a separate runtime toggle in PAS v1. Use logging.log_debug_metrics.',
    ('logging', 'log_tensor_norms'): 'logging.log_tensor_norms is not a separate runtime toggle in PAS v1. Use logging.log_debug_metrics.',
    ('logging', 'detect_dead_prototypes'): 'logging.detect_dead_prototypes is not a separate runtime toggle in PAS v1. Use prototype.dead_prototype_threshold and logging.log_debug_metrics.',
    ('logging', 'dead_prototype_threshold'): 'logging.dead_prototype_threshold moved to prototype.dead_prototype_threshold.',
    ('loss', 'use_loss_ret_exact'): 'loss.use_loss_ret_exact was removed. Use loss.use_loss_ret for row-wise surrogate image-to-text retrieval only.',
    ('loss', 'use_loss_ret_exact_image'): 'loss.use_loss_ret_exact_image was removed. Use loss.use_loss_ret for row-wise surrogate image-to-text retrieval only.',
    ('loss', 'use_loss_ret_exact_text'): 'loss.use_loss_ret_exact_text was removed because text-to-image retrieval is invalid for image-conditioned surrogate text embeddings.',
    ('loss', 'lambda_ret_exact'): 'loss.lambda_ret_exact was removed. Use loss.lambda_ret for row-wise surrogate image-to-text retrieval only.',
    ('loss', 'lambda_ret_exact_image'): 'loss.lambda_ret_exact_image was removed. Use loss.lambda_ret for row-wise surrogate image-to-text retrieval only.',
    ('loss', 'lambda_ret_exact_text'): 'loss.lambda_ret_exact_text was removed because text-to-image retrieval is invalid for image-conditioned surrogate text embeddings.',
    ('loss', 'ret_exact_temperature'): 'loss.ret_exact_temperature was removed. model.temperature defines surrogate retrieval scoring.',
    ('training', 'freeze_prototype'): 'training.freeze_prototype was replaced by training.freeze_prototype_side.',
    ('training', 'freeze_proxy'): 'training.freeze_proxy was replaced by training.freeze_prototype_side.',
    ('training', 'bidirectional_ret'): 'Bidirectional retrieval over the surrogate score matrix is invalid. Only row-wise image-to-text retrieval is supported.',
    ('training', 'clip_style_ret'): 'CLIP-style symmetric retrieval over the surrogate score matrix is invalid. Only row-wise image-to-text retrieval is supported.',
    ('training', 'symmetric_ret'): 'Symmetric retrieval over the surrogate score matrix is invalid. Only row-wise image-to-text retrieval is supported.',
    ('training', 't2i_ret'): 'Text-to-image retrieval over the surrogate score matrix is invalid because surrogate text embeddings depend on the image query.',
    ('model', 'learn_logit_scale'): 'model.learn_logit_scale was removed because the amortized PAS runtime always uses a fixed retrieval temperature.',
    ('model', 'logit_scale_init'): 'model.logit_scale_init is not exposed in minimal PAS v1.',
    ('model', 'logit_scale_max'): 'model.logit_scale_max is not exposed in minimal PAS v1.',
    ('optimizer', 'lr_logit_scale'): 'optimizer.lr_logit_scale was removed because PAS no longer optimizes a learnable logit scale.',
    ('optimizer', 'weight_decay_logit_scale'): 'optimizer.weight_decay_logit_scale was removed because PAS no longer optimizes a learnable logit scale.',
}


CONFIG_ENUM_CHOICES: Dict[Tuple[str, ...], Tuple[str, ...]] = {
    ('model', 'training_mode'): ('pas', 'vanilla_clip'),
    ('host', 'type'): ('clip', 'itself'),
    ('host', 'itself_topk_type'): ('mean', 'std', 'layer_index', 'custom'),
    ('model', 'projector_type'): ('mlp2', 'linear'),
    ('model', 'backbone_precision'): tuple(BACKBONE_PRECISION_ALIASES.keys()),
    ('model', 'prototype_precision'): tuple(PROTOTYPE_PRECISION_ALIASES.keys()),
    ('prototype', 'prototype_init'): (
        'normalized_random',
        'sampled_image_embeddings',
        'kmeans_centroids',
        'orthogonal_normalized_random',
        'spherical_kmeans_centroids',
        'hybrid_spherical_kmeans_random',
    ),
    ('prototype', 'routing_type'): ('cosine', 'dot'),
    ('prototype', 'contextualization_type'): ('self_attention', 'dense_self_attention', 'none'),
    ('text_pooling', 'token_policy'): ('content_only', 'content_plus_special', 'eos_only'),
    ('text_pooling', 'scoring_type'): ('cosine', 'dot'),
    ('loss', 'retrieval_mode'): ('surrogate_i2t', 'clip_bidirectional'),
    ('training', 'stage'): ('stage0', 'stage1', 'stage2', 'stage3', 'joint'),
    ('training', 'amp_dtype'): tuple(AMP_DTYPE_ALIASES.keys()),
    ('optimizer', 'type'): ('SGD', 'Adam', 'AdamW'),
    ('optimizer', 'scheduler'): ('step', 'exp', 'poly', 'cosine', 'linear'),
    ('dataset', 'dataset_name'): ('CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid'),
    ('dataset', 'val_dataset'): ('val', 'test'),
    ('evaluation', 'target_domain'): ('CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid'),
    ('evaluation', 'retrieval_scorer'): ('exact', 'approximate'),
}


RUNTIME_ENUM_CHOICES: Dict[str, Tuple[str, ...]] = {
    'training_mode': ('pas', 'vanilla_clip'),
    'host_type': ('clip', 'itself'),
    'itself_topk_type': ('mean', 'std', 'layer_index', 'custom'),
    'projector_type': ('mlp2', 'linear'),
    'backbone_precision': tuple(BACKBONE_PRECISION_ALIASES.keys()),
    'prototype_precision': tuple(PROTOTYPE_PRECISION_ALIASES.keys()),
    'prototype_init': (
        'normalized_random',
        'sampled_image_embeddings',
        'kmeans_centroids',
        'orthogonal_normalized_random',
        'spherical_kmeans_centroids',
        'hybrid_spherical_kmeans_random',
    ),
    'prototype_routing_type': ('cosine', 'dot'),
    'prototype_contextualization_type': ('self_attention', 'dense_self_attention', 'none'),
    'token_policy': ('content_only', 'content_plus_special', 'eos_only'),
    'token_scoring_type': ('cosine', 'dot'),
    'retrieval_mode': ('surrogate_i2t', 'clip_bidirectional'),
    'training_stage': ('stage0', 'stage1', 'stage2', 'stage3', 'joint'),
    'amp_dtype': tuple(AMP_DTYPE_ALIASES.keys()),
    'optimizer': ('SGD', 'Adam', 'AdamW'),
    'lrscheduler': ('step', 'exp', 'poly', 'cosine', 'linear'),
    'dataset_name': ('CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid'),
    'val_dataset': ('val', 'test'),
    'target_domain': ('CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid'),
    'retrieval_scorer': ('exact', 'approximate'),
}


def _format_allowed_values(allowed_values: Tuple[str, ...]) -> str:
    return '[' + ', '.join(repr(value) for value in allowed_values) + ']'


def _validate_enum_value(field_name: str, value: Any, allowed_values: Tuple[str, ...]) -> None:
    if str(value).lower() not in {allowed.lower() for allowed in allowed_values}:
        raise ValueError(
            f'Invalid value for {field_name}: {value!r}. Allowed values: {_format_allowed_values(allowed_values)}'
        )


def _validate_retrieval_metrics_value(field_name: str, value: Any) -> None:
    if not isinstance(value, list):
        raise ValueError(f'{field_name} must be a list of metric names.')
    allowed_values = ('R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum')
    invalid_values = [metric for metric in value if str(metric) not in allowed_values]
    if invalid_values:
        raise ValueError(
            f'Invalid value for {field_name}: {invalid_values!r}. Allowed values: {_format_allowed_values(allowed_values)}'
        )


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Config file must contain a mapping: {path}')
    return data


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _path_exists(config_data: Dict[str, Any], path: Tuple[str, ...]) -> bool:
    current: Any = config_data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _validate_known_sections(config_data: Dict[str, Any]) -> None:
    for key in config_data.keys():
        if key not in SECTION_KEYS:
            raise ValueError(f'Unknown config section `{key}`. Supported sections: {sorted(SECTION_KEYS)}')


def _validate_supported_keys(config_data: Dict[str, Any]) -> None:
    supported_paths = set(PRIMARY_CONFIG_KEY_MAP.keys()) | set(READ_ALIAS_CONFIG_KEY_MAP.keys())
    supported_leafs = {(path[0], path[1]) for path in supported_paths if len(path) == 2}

    for section_name, section_value in config_data.items():
        if not isinstance(section_value, dict):
            raise ValueError(f'Config section `{section_name}` must contain a mapping.')
        for key, value in section_value.items():
            path = (section_name, key)
            if path == ('text_pooling', 'special_token_ids'):
                if not isinstance(value, dict):
                    raise ValueError('text_pooling.special_token_ids must be a mapping of token names to ids.')
                unknown_keys = sorted(set(value.keys()) - SUPPORTED_SPECIAL_TOKEN_ID_KEYS)
                if unknown_keys:
                    raise ValueError(
                        'Unsupported text_pooling.special_token_ids keys: '
                        f'{unknown_keys}. Supported keys: {sorted(SUPPORTED_SPECIAL_TOKEN_ID_KEYS)}'
                    )
                continue
            if isinstance(value, dict):
                raise ValueError(f'Unsupported nested config mapping at `{section_name}.{key}`.')
            if path not in supported_leafs and path not in UNSUPPORTED_CONFIG_PATHS:
                raise ValueError(f'Unknown config key `{section_name}.{key}`.')


def validate_config_data(config_data: Dict[str, Any]) -> None:
    if not config_data:
        return
    _validate_known_sections(config_data)
    _validate_supported_keys(config_data)
    for path, message in UNSUPPORTED_CONFIG_PATHS.items():
        if _path_exists(config_data, path):
            raise ValueError(message)
    for path, allowed_values in CONFIG_ENUM_CHOICES.items():
        if not _path_exists(config_data, path):
            continue
        current = config_data
        for key in path:
            current = current[key]
        _validate_enum_value('.'.join(path), current, allowed_values)
    if _path_exists(config_data, ('evaluation', 'retrieval_metrics')):
        _validate_retrieval_metrics_value('evaluation.retrieval_metrics', config_data['evaluation']['retrieval_metrics'])
    training_mode = str(config_data.get('model', {}).get('training_mode', 'pas')).lower()
    host_type = str(config_data.get('host', {}).get('type', 'clip')).lower()
    use_prototype_branch = bool(config_data.get('model', {}).get('use_prototype_branch', training_mode != 'vanilla_clip'))
    retrieval_mode = str(config_data.get('loss', {}).get('retrieval_mode', 'surrogate_i2t')).lower()
    if training_mode == 'vanilla_clip' and host_type != 'clip':
        raise ValueError('model.training_mode=vanilla_clip requires host.type=clip.')
    if training_mode == 'vanilla_clip':
        if bool(config_data.get('model', {}).get('use_prototype_bank', True)):
            raise ValueError('model.training_mode=vanilla_clip requires model.use_prototype_bank=false.')
        if bool(config_data.get('model', {}).get('use_image_conditioned_pooling', True)):
            raise ValueError('model.training_mode=vanilla_clip requires model.use_image_conditioned_pooling=false.')
        if str(config_data.get('text_pooling', {}).get('token_policy', 'eos_only')).lower() != 'eos_only':
            raise ValueError('model.training_mode=vanilla_clip requires text_pooling.token_policy=eos_only.')
        if retrieval_mode != 'clip_bidirectional':
            raise ValueError('model.training_mode=vanilla_clip requires loss.retrieval_mode=clip_bidirectional.')
        if not bool(config_data.get('loss', {}).get('use_loss_ret', True)):
            raise ValueError('model.training_mode=vanilla_clip requires loss.use_loss_ret=true.')
        if str(config_data.get('evaluation', {}).get('retrieval_scorer', 'exact')).lower() != 'exact':
            raise ValueError('model.training_mode=vanilla_clip requires evaluation.retrieval_scorer=exact.')
        incompatible_flags = {
            'loss.use_loss_proxy_image': bool(config_data.get('loss', {}).get('use_loss_proxy_image', False)),
            'loss.use_loss_proxy_text': bool(config_data.get('loss', {}).get('use_loss_proxy_text', False)),
            'loss.use_loss_proxy_text_exact': bool(config_data.get('loss', {}).get('use_loss_proxy_text_exact', False)),
            'loss.use_loss_align': bool(config_data.get('loss', {}).get('use_loss_align', False)),
            'loss.use_loss_diag': bool(config_data.get('loss', {}).get('use_loss_diag', False)),
            'loss.use_loss_support': bool(config_data.get('loss', {}).get('use_loss_support', False)),
            'loss.use_balancing_loss': bool(config_data.get('loss', {}).get('use_balancing_loss', False)),
            'loss.use_diversity_loss': bool(config_data.get('loss', {}).get('use_diversity_loss', False)),
        }
        enabled_incompatible = sorted(name for name, enabled in incompatible_flags.items() if enabled)
        if enabled_incompatible:
            raise ValueError(
                'model.training_mode=vanilla_clip does not support prototype/auxiliary losses. '
                f'Disable: {enabled_incompatible}.'
            )
    elif retrieval_mode == 'clip_bidirectional':
        raise ValueError('loss.retrieval_mode=clip_bidirectional is only supported when model.training_mode=vanilla_clip.')
    if (
        training_mode != 'vanilla_clip'
        and use_prototype_branch
        and bool(config_data.get('model', {}).get('use_prototype_bank', True))
        and not bool(config_data.get('model', {}).get('use_image_conditioned_pooling', True))
    ):
        raise ValueError(
            'model.use_prototype_bank=true requires model.use_image_conditioned_pooling=true. '
            'Prototype-routed training with text-only pooling is no longer supported.'
        )

    training_stage = str(config_data.get('training', {}).get('stage', 'stage1')).lower()
    if training_stage == 'stage0' and use_prototype_branch:
        raise ValueError('training.stage=stage0 is reserved for host-only baselines and requires model.use_prototype_branch=false.')


def load_yaml_config(default_path: Optional[str] = None, override_path: Optional[str] = None) -> Dict[str, Any]:
    config = {}
    if default_path:
        config = deep_merge_dicts(config, _read_yaml(default_path))
    if override_path:
        config = deep_merge_dicts(config, _read_yaml(override_path))
    validate_config_data(config)
    return config


def validate_runtime_args_namespace(args) -> None:
    for field_name, allowed_values in RUNTIME_ENUM_CHOICES.items():
        if not hasattr(args, field_name):
            continue
        _validate_enum_value(field_name, getattr(args, field_name), allowed_values)
    retrieval_metrics = getattr(args, 'retrieval_metrics', None)
    if retrieval_metrics is not None:
        _validate_retrieval_metrics_value('retrieval_metrics', list(retrieval_metrics))
    training_mode = str(getattr(args, 'training_mode', 'pas')).lower()
    host_type = str(getattr(args, 'host_type', 'clip')).lower()
    use_prototype_branch = bool(getattr(args, 'use_prototype_branch', training_mode != 'vanilla_clip'))
    retrieval_mode = str(getattr(args, 'retrieval_mode', 'surrogate_i2t')).lower()
    if training_mode == 'vanilla_clip' and host_type != 'clip':
        raise ValueError('model.training_mode=vanilla_clip requires host.type=clip.')
    if training_mode == 'vanilla_clip':
        if use_prototype_branch:
            raise ValueError('model.training_mode=vanilla_clip requires model.use_prototype_branch=false.')
        if bool(getattr(args, 'use_prototype_bank', True)):
            raise ValueError('model.training_mode=vanilla_clip requires model.use_prototype_bank=false.')
        if bool(getattr(args, 'use_image_conditioned_pooling', True)):
            raise ValueError('model.training_mode=vanilla_clip requires model.use_image_conditioned_pooling=false.')
        if str(getattr(args, 'token_policy', 'eos_only')).lower() != 'eos_only':
            raise ValueError('model.training_mode=vanilla_clip requires text_pooling.token_policy=eos_only.')
        if retrieval_mode != 'clip_bidirectional':
            raise ValueError('model.training_mode=vanilla_clip requires loss.retrieval_mode=clip_bidirectional.')
        if not bool(getattr(args, 'use_loss_ret', True)):
            raise ValueError('model.training_mode=vanilla_clip requires loss.use_loss_ret=true.')
        if str(getattr(args, 'retrieval_scorer', 'exact')).lower() != 'exact':
            raise ValueError('model.training_mode=vanilla_clip requires evaluation.retrieval_scorer=exact.')
        incompatible_flags = {
            'loss.use_loss_proxy_image': bool(getattr(args, 'use_loss_proxy_image', False)),
            'loss.use_loss_proxy_text': bool(getattr(args, 'use_loss_proxy_text', False)),
            'loss.use_loss_proxy_text_exact': bool(getattr(args, 'use_loss_proxy_text_exact', False)),
            'loss.use_loss_align': bool(getattr(args, 'use_loss_align', False)),
            'loss.use_loss_diag': bool(getattr(args, 'use_loss_diag', False)),
            'loss.use_loss_support': bool(getattr(args, 'use_loss_support', False)),
            'loss.use_balancing_loss': bool(getattr(args, 'use_balancing_loss', False)),
            'loss.use_diversity_loss': bool(getattr(args, 'use_diversity_loss', False)),
        }
        enabled_incompatible = sorted(name for name, enabled in incompatible_flags.items() if enabled)
        if enabled_incompatible:
            raise ValueError(
                'model.training_mode=vanilla_clip does not support prototype/auxiliary losses. '
                f'Disable: {enabled_incompatible}.'
            )
    elif retrieval_mode == 'clip_bidirectional':
        raise ValueError('loss.retrieval_mode=clip_bidirectional is only supported when model.training_mode=vanilla_clip.')
    if training_mode != 'vanilla_clip' and use_prototype_branch and bool(getattr(args, 'use_prototype_bank', True)) and not bool(getattr(args, 'use_image_conditioned_pooling', True)):
        raise ValueError(
            'use_prototype_bank=true requires use_image_conditioned_pooling=true. '
            'Prototype-routed training with text-only pooling is no longer supported.'
        )

    if str(getattr(args, 'training_stage', 'stage1')).lower() == 'stage0' and use_prototype_branch:
        raise ValueError('training.stage=stage0 is reserved for host-only baselines and requires model.use_prototype_branch=false.')


def _normalize_value(dest: str, value: Any) -> Any:
    if dest == 'img_size' and isinstance(value, list):
        return tuple(value)
    return value


def _extract_cli_destinations(parser, argv: Iterable[str]) -> set:
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    destinations = set()
    for token in argv:
        if not token.startswith('--'):
            continue
        option = token.split('=', 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            destinations.add(dest)
    return destinations


def _iter_config_value_paths(config_key_map: Dict[Tuple[str, ...], str], config_data: Dict[str, Any]):
    for path, dest in config_key_map.items():
        current = config_data
        found = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                found = False
                break
            current = current[key]
        if found:
            yield path, dest, current


def apply_config_to_args(
    parser,
    args,
    config_data: Dict[str, Any],
    argv: Optional[List[str]] = None,
    override_config_data: Optional[Dict[str, Any]] = None,
):
    cli_dests = _extract_cli_destinations(parser, argv or [])
    args.cli_dests = set(cli_dests)
    applied_dests = set()

    def _apply_from_source(source_config: Optional[Dict[str, Any]]) -> None:
        if not source_config:
            return
        for mapping in (PRIMARY_CONFIG_KEY_MAP, READ_ALIAS_CONFIG_KEY_MAP):
            for _, dest, value in _iter_config_value_paths(mapping, source_config):
                if dest in cli_dests or dest in applied_dests:
                    continue
                setattr(args, dest, _normalize_value(dest, value))
                applied_dests.add(dest)

    # Explicit override-file entries must beat defaults, even when they arrive through
    # backward-compatible alias paths such as prototype.* or training.* legacy knobs.
    _apply_from_source(override_config_data)
    _apply_from_source(config_data)
    args.config_data = config_data
    return args


def build_runtime_config(args) -> Dict[str, Any]:
    config = copy.deepcopy(SECTION_TEMPLATE)
    for path, dest in PRIMARY_CONFIG_KEY_MAP.items():
        if not hasattr(args, dest):
            continue
        current = config
        for key in path[:-1]:
            current = current.setdefault(key, {})
        value = getattr(args, dest)
        if isinstance(value, tuple):
            value = list(value)
        current[path[-1]] = copy.deepcopy(value)
    return config


def flatten_config_dict(config_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config_data, dict):
        return {}
    if not SECTION_KEYS.intersection(config_data.keys()):
        return copy.deepcopy(config_data)

    flat = {}
    for mapping in (PRIMARY_CONFIG_KEY_MAP, READ_ALIAS_CONFIG_KEY_MAP):
        for _, dest, value in _iter_config_value_paths(mapping, config_data):
            if dest in flat:
                continue
            flat[dest] = _normalize_value(dest, value)
    return flat


def dump_yaml_config(path: str, config_data: Dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(config_data, handle, default_flow_style=False, sort_keys=False)


