import copy
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from utils.freeze_schedule import parse_freeze_schedule_config
from utils.module_group_registry import CHECKPOINT_GROUPS
from utils.precision import AMP_DTYPE_ALIASES, BACKBONE_PRECISION_ALIASES, PROTOTYPE_PRECISION_ALIASES


PRIMARY_CONFIG_KEY_MAP: Dict[Tuple[str, ...], str] = {
    ('experiment', 'name'): 'name',
    ('experiment', 'output_dir'): 'output_dir',
    ('experiment', 'seed'): 'seed',
    ('experiment', 'local_rank'): 'local_rank',

    ('model', 'name'): 'model_name',
    ('model', 'variant'): 'model_variant',
    ('model', 'training_mode'): 'training_mode',
    ('model', 'runtime_mode'): 'runtime_mode',
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
    ('model', 'return_debug_outputs'): 'return_debug_outputs',

    ('host', 'type'): 'host_type',
    ('host', 'freeze_projectors'): 'freeze_host_retrieval',
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
    ('host', 'itself_lambda1_weight'): 'lambda1_weight',
    ('host', 'itself_lambda2_weight'): 'lambda2_weight',

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
    ('prototype', 'routing_source'): 'prototype_routing_source',
    ('prototype', 'local_routing_temperature'): 'prototype_local_routing_temperature',
    ('prototype', 'local_routing_pooling'): 'prototype_local_routing_pooling',
    ('prototype', 'local_routing_use_adapter'): 'prototype_local_routing_use_adapter',
    ('prototype', 'local_routing_adapter_dim'): 'prototype_local_routing_adapter_dim',
    ('prototype', 'local_routing_normalize_inputs'): 'prototype_local_routing_normalize_inputs',
    ('prototype', 'contextualization_enabled'): 'prototype_contextualization_enabled',
    ('prototype', 'contextualization_type'): 'prototype_contextualization_type',
    ('prototype', 'contextualization_residual'): 'prototype_contextualization_residual',
    ('prototype', 'normalize_for_self_interaction'): 'normalize_for_self_interaction',
    ('prototype', 'normalize_for_routing'): 'normalize_for_routing',
    ('prototype', 'dead_prototype_threshold'): 'prototype_dead_threshold',

    ('fusion', 'enabled'): 'fusion_enabled',
    ('fusion', 'lambda_host'): 'fusion_lambda_host',
    ('fusion', 'lambda_prototype'): 'fusion_lambda_prototype',
    ('fusion', 'eval_subsets'): 'fusion_eval_subsets',
    ('fusion', 'coefficient_source'): 'fusion_coefficient_source',
    ('fusion', 'composer_calibration_enabled'): 'composer_calibration_enabled',

    ('objectives', 'objectives', 'use_host_loss'): 'use_host_loss',
    ('objectives', 'objectives', 'use_loss_proxy_image'): 'use_loss_proxy_image',
    ('objectives', 'objectives', 'use_loss_proxy_text'): 'use_loss_proxy_text',
    ('objectives', 'objectives', 'use_loss_proxy_text_exact'): 'use_loss_proxy_text_exact',
    ('objectives', 'objectives', 'use_loss_align'): 'use_loss_align',
    ('objectives', 'objectives', 'use_loss_diag'): 'use_loss_dir',
    ('objectives', 'objectives', 'use_loss_gap'): 'use_loss_gap',
    ('objectives', 'objectives', 'use_loss_support'): 'use_loss_sup',
    ('objectives', 'objectives', 'prototype_gap_margin'): 'prototype_gap_margin',
    ('objectives', 'objectives', 'prototype_support_target'): 'prototype_support_target',
    ('objectives', 'objectives', 'use_loss_ret'): 'use_loss_ret',
    ('objectives', 'objectives', 'use_loss_weight_ret'): 'use_loss_weight_ret',
    ('objectives', 'objectives', 'retrieval_mode'): 'retrieval_mode',
    ('objectives', 'objectives', 'weight_ret_margin_delta'): 'weight_ret_margin_delta',
    ('objectives', 'objectives', 'weight_ret_tau'): 'weight_ret_tau',
    ('objectives', 'objectives', 'weight_ret_detach_host'): 'weight_ret_detach_host',
    ('objectives', 'objectives', 'weight_ret_normalize_mean_one'): 'weight_ret_normalize_mean_one',
    ('objectives', 'objectives', 'use_balancing_loss'): 'use_balancing_loss',
    ('objectives', 'objectives', 'use_diversity_loss'): 'use_diversity_loss',

    ('objectives', 'lambda', 'host'): 'lambda_host',
    ('objectives', 'lambda', 'proxy'): 'lambda_proxy',
    ('objectives', 'lambda', 'proxy_image'): 'lambda_proxy_image',
    ('objectives', 'lambda', 'proxy_text'): 'lambda_proxy_text',
    ('objectives', 'lambda', 'proxy_text_exact'): 'lambda_proxy_text_exact',
    ('objectives', 'lambda', 'align'): 'lambda_align',
    ('objectives', 'lambda', 'diag'): 'lambda_dir',
    ('objectives', 'lambda', 'gap'): 'lambda_gap',
    ('objectives', 'lambda', 'support'): 'lambda_sup',
    ('objectives', 'lambda', 'ret'): 'lambda_ret',
    ('objectives', 'lambda', 'weight_ret'): 'lambda_weight_ret',
    ('objectives', 'lambda', 'balance'): 'prototype_balance_loss_weight',
    ('objectives', 'lambda', 'diversity'): 'diversity_loss_weight',

    ('text_pooling', 'token_policy'): 'token_policy',
    ('text_pooling', 'scoring_type'): 'token_scoring_type',
    ('text_pooling', 'normalize_for_token_scoring'): 'normalize_for_token_scoring',
    ('text_pooling', 'token_temperature'): 'token_pooling_temperature',
    ('text_pooling', 'special_token_ids'): 'special_token_ids',
    ('text_pooling', 'error_on_empty_kept_tokens'): 'error_on_empty_kept_tokens',

    ('training', 'batch_size'): 'batch_size',
    ('training', 'stage'): 'training_stage',
    ('training', 'epochs'): 'num_epoch',
    ('training', 'log_period'): 'log_period',
    ('training', 'eval_frequency'): 'eval_period',
    ('training', 'save_interval'): 'save_interval',
    ('training', 'prototype_selection_metric'): 'prototype_selection_metric',
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
    ('training', 'freeze_host_backbone'): 'freeze_host_backbone',
    ('training', 'freeze_host_retrieval'): 'freeze_host_retrieval',
    ('training', 'freeze_fusion'): 'freeze_fusion',
    ('training', 'freeze_prototype_bank'): 'freeze_prototype_bank',
    ('training', 'freeze_prototype_projector'): 'freeze_prototype_projector',
    ('training', 'freeze_routing'): 'freeze_routing',
    ('training', 'freeze_image_backbone'): 'freeze_image_backbone',
    ('training', 'freeze_text_backbone'): 'freeze_text_backbone',
    ('training', 'grad_clip'): 'grad_clip',
    ('training', 'amp'): 'amp',
    ('training', 'amp_dtype'): 'amp_dtype',
    ('training', 'proxy_temperature'): 'proxy_temperature',
    ('training', 'diag_temperature'): 'diag_temperature',

    ('optimizer', 'type'): 'optimizer',
    ('optimizer', 'lr'): 'lr',
    ('optimizer', 'lr_prototype_bank'): 'lr_prototype_bank',
    ('optimizer', 'lr_projectors'): 'lr_projectors',
    ('optimizer', 'lr_prototype_routing'): 'lr_prototype_routing',
    ('optimizer', 'lr_prototype_pooling'): 'lr_prototype_pooling',
    ('optimizer', 'lr_prototype_contextualization'): 'lr_prototype_contextualization',
    ('optimizer', 'lr_host_projectors'): 'lr_host_projectors',
    ('optimizer', 'lr_class_proxies'): 'lr_class_proxies',
    ('optimizer', 'lr_image_backbone'): 'lr_image_backbone',
    ('optimizer', 'lr_text_backbone'): 'lr_text_backbone',
    ('optimizer', 'weight_decay'): 'weight_decay',
    ('optimizer', 'weight_decay_prototype_bank'): 'weight_decay_prototype_bank',
    ('optimizer', 'weight_decay_projectors'): 'weight_decay_projectors',
    ('optimizer', 'weight_decay_prototype_routing'): 'weight_decay_prototype_routing',
    ('optimizer', 'weight_decay_prototype_pooling'): 'weight_decay_prototype_pooling',
    ('optimizer', 'weight_decay_prototype_contextualization'): 'weight_decay_prototype_contextualization',
    ('optimizer', 'weight_decay_host_projectors'): 'weight_decay_host_projectors',
    ('optimizer', 'weight_decay_class_proxies'): 'weight_decay_class_proxies',
    ('optimizer', 'weight_decay_image_backbone'): 'weight_decay_image_backbone',
    ('optimizer', 'weight_decay_text_backbone'): 'weight_decay_text_backbone',
    ('optimizer', 'momentum'): 'momentum',
    ('optimizer', 'alpha'): 'alpha',
    ('optimizer', 'beta'): 'beta',
    ('optimizer', 'lr_factor'): 'lr_factor',
    ('optimizer', 'bias_lr_factor'): 'bias_lr_factor',
    ('optimizer', 'weight_decay_bias'): 'weight_decay_bias',
    ('optimizer', 'optimizer_eps'): 'optimizer_eps',
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

# Backward-compatible alias paths (accepted by parser/loader) that are intentionally
# not part of canonical configs/base.yaml.
READ_ALIAS_CONFIG_KEY_MAP: Dict[Tuple[str, ...], str] = {
    ('training', 'training_stage'): 'training_stage',
    ('training', 'runtime_mode'): 'runtime_mode',
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
    ('prototype', 'routing_source'): 'prototype_routing_source',
    ('prototype', 'local_routing_temperature'): 'prototype_local_routing_temperature',
    ('prototype', 'local_routing_pooling'): 'prototype_local_routing_pooling',
    ('prototype', 'local_routing_use_adapter'): 'prototype_local_routing_use_adapter',
    ('prototype', 'local_routing_adapter_dim'): 'prototype_local_routing_adapter_dim',
    ('prototype', 'local_routing_normalize_inputs'): 'prototype_local_routing_normalize_inputs',
    ('prototype', 'use_balancing_loss'): 'use_balancing_loss',
    ('prototype', 'balance_loss_weight'): 'prototype_balance_loss_weight',
    ('prototype', 'use_diversity_loss'): 'use_diversity_loss',
    ('prototype', 'diversity_loss_weight'): 'diversity_loss_weight',
    ('prototype', 'lambda_bal'): 'prototype_balance_loss_weight',
    ('prototype', 'lambda_div'): 'diversity_loss_weight',

    ('objectives', 'use_host_loss'): 'use_host_loss',
    ('objectives', 'lambda_host'): 'lambda_host',
    ('objectives', 'objectives', 'use_loss_dir'): 'use_loss_dir',
    ('objectives', 'objectives', 'use_loss_sup'): 'use_loss_sup',
    ('objectives', 'objectives', 'use_loss_diag'): 'use_loss_dir',
    ('objectives', 'objectives', 'use_loss_support'): 'use_loss_sup',
    ('objectives', 'objectives', 'support_min'): 'prototype_support_target',
    ('objectives', 'lambda', 'dir'): 'lambda_dir',
    ('objectives', 'lambda', 'sup'): 'lambda_sup',
    ('objectives', 'lambda', 'diag'): 'lambda_dir',
    ('objectives', 'lambda', 'support'): 'lambda_sup',
    ('objectives', 'use_proto_loss_ret'): 'use_loss_ret',
    ('objectives', 'lambda_proto_ret'): 'lambda_ret',
    ('objectives', 'use_diag_fidelity'): 'use_loss_dir',
    ('objectives', 'lambda_diag'): 'lambda_dir',
    ('objectives', 'diag_temperature'): 'diag_temperature',
    ('objectives', 'use_diversity'): 'use_diversity_loss',
    ('objectives', 'lambda_diversity'): 'diversity_loss_weight',
    ('objectives', 'use_balance'): 'use_balancing_loss',
    ('objectives', 'lambda_balance'): 'prototype_balance_loss_weight',
    ('objectives', 'prototype_gap_margin'): 'prototype_gap_margin',
    ('objectives', 'prototype_support_target'): 'prototype_support_target',

    ('loss', 'lambda_proxy'): 'lambda_proxy',
    ('loss', 'lambda_proxy_image'): 'lambda_proxy_image',
    ('loss', 'lambda_proxy_text'): 'lambda_proxy_text',
    ('loss', 'lambda_proxy_text_exact'): 'lambda_proxy_text_exact',
    ('loss', 'use_loss_proxy_image'): 'use_loss_proxy_image',
    ('loss', 'use_loss_proxy_text'): 'use_loss_proxy_text',
    ('loss', 'use_loss_proxy_text_exact'): 'use_loss_proxy_text_exact',
    ('loss', 'use_loss_align'): 'use_loss_align',
    ('loss', 'lambda_align'): 'lambda_align',
    ('loss', 'use_loss_dir'): 'use_loss_dir',
    ('loss', 'lambda_dir'): 'lambda_dir',
    ('loss', 'use_loss_gap'): 'use_loss_gap',
    ('loss', 'lambda_gap'): 'lambda_gap',
    ('loss', 'prototype_gap_margin'): 'prototype_gap_margin',
    ('loss', 'use_loss_sup'): 'use_loss_sup',
    ('loss', 'lambda_sup'): 'lambda_sup',
    ('loss', 'prototype_support_target'): 'prototype_support_target',
    ('loss', 'use_loss_diag'): 'use_loss_dir',
    ('loss', 'lambda_diag'): 'lambda_dir',
    ('loss', 'diag_temperature'): 'diag_temperature',
    ('loss', 'use_loss_ret'): 'use_loss_ret',
    ('loss', 'use_loss_weight_ret'): 'use_loss_weight_ret',
    ('loss', 'retrieval_mode'): 'retrieval_mode',
    ('loss', 'lambda_ret'): 'lambda_ret',
    ('loss', 'lambda_weight_ret'): 'lambda_weight_ret',
    ('loss', 'weight_ret_margin_delta'): 'weight_ret_margin_delta',
    ('loss', 'weight_ret_tau'): 'weight_ret_tau',
    ('loss', 'weight_ret_detach_host'): 'weight_ret_detach_host',
    ('loss', 'weight_ret_normalize_mean_one'): 'weight_ret_normalize_mean_one',
    ('loss', 'use_loss_support'): 'use_loss_sup',
    ('loss', 'lambda_support'): 'lambda_sup',
    ('loss', 'support_min'): 'prototype_support_target',
    ('loss', 'use_balancing_loss'): 'use_balancing_loss',
    ('loss', 'balance_loss_weight'): 'prototype_balance_loss_weight',
    ('loss', 'use_diversity_loss'): 'use_diversity_loss',
    ('loss', 'diversity_loss_weight'): 'diversity_loss_weight',

    ('training', 'lambda_proxy'): 'lambda_proxy',
    ('training', 'lambda_proxy_image'): 'lambda_proxy_image',
    ('training', 'lambda_proxy_text'): 'lambda_proxy_text',
    ('training', 'lambda_proxy_text_exact'): 'lambda_proxy_text_exact',
    ('training', 'use_loss_proxy_image'): 'use_loss_proxy_image',
    ('training', 'use_loss_proxy_text'): 'use_loss_proxy_text',
    ('training', 'use_loss_proxy_text_exact'): 'use_loss_proxy_text_exact',
    ('training', 'use_loss_align'): 'use_loss_align',
    ('training', 'lambda_align'): 'lambda_align',
    ('training', 'use_loss_dir'): 'use_loss_dir',
    ('training', 'lambda_dir'): 'lambda_dir',
    ('training', 'use_loss_gap'): 'use_loss_gap',
    ('training', 'lambda_gap'): 'lambda_gap',
    ('training', 'prototype_gap_margin'): 'prototype_gap_margin',
    ('training', 'use_loss_sup'): 'use_loss_sup',
    ('training', 'lambda_sup'): 'lambda_sup',
    ('training', 'prototype_support_target'): 'prototype_support_target',
    ('training', 'use_loss_diag'): 'use_loss_dir',
    ('training', 'lambda_diag'): 'lambda_dir',
    ('training', 'diag_temperature'): 'diag_temperature',
    ('training', 'use_loss_ret'): 'use_loss_ret',
    ('training', 'lambda_ret'): 'lambda_ret',
    ('training', 'use_loss_weight_ret'): 'use_loss_weight_ret',
    ('training', 'lambda_weight_ret'): 'lambda_weight_ret',
    ('training', 'weight_ret_margin_delta'): 'weight_ret_margin_delta',
    ('training', 'weight_ret_tau'): 'weight_ret_tau',
    ('training', 'weight_ret_detach_host'): 'weight_ret_detach_host',
    ('training', 'weight_ret_normalize_mean_one'): 'weight_ret_normalize_mean_one',
    ('training', 'log_debug_metrics'): 'log_debug_metrics',

    ('text_pooling', 'token_similarity'): 'token_scoring_type',
    ('text_pooling', 'tau_t'): 'token_pooling_temperature',
    ('objectives', 'objectives', 'diag_temperature'): 'diag_temperature',
    ('optimizer', 'weight_decay_prototypes'): 'weight_decay_prototype_bank',
    ('host', 'loss_names'): 'itself_loss_names',
    ('host', 'only_global'): 'itself_only_global',
    ('host', 'select_ratio'): 'itself_select_ratio',
    ('host', 'tau'): 'itself_tau',
    ('host', 'margin'): 'itself_margin',
    ('host', 'return_all'): 'itself_return_all',
    ('host', 'topk_type'): 'itself_topk_type',
    ('host', 'layer_index'): 'itself_layer_index',
    ('host', 'average_attn_weights'): 'itself_average_attn_weights',
    ('host', 'modify_k'): 'itself_modify_k',
    ('host', 'lambda1_weight'): 'lambda1_weight',
    ('host', 'lambda2_weight'): 'lambda2_weight',
    ('host', 'use_custom_projector'): 'use_custom_projector',
    ('host', 'enabled'): 'use_host_loss',
    ('host', 'loss_weight'): 'lambda_host',
    ('training', 'freeze_host_projectors'): 'freeze_host_projectors',
    ('training', 'freeze_prototype_side'): 'freeze_prototype_side',
    ('training', 'val_dataset'): 'val_dataset',
    ('fusion', 'coefficient'): 'fusion_coefficient',
}

SECTION_TEMPLATE = {
    'experiment': {},
    'model': {},
    'host': {},
    'prototype': {},
    'fusion': {},
    'objectives': {
        'objectives': {},
        'lambda': {},
    },
    'text_pooling': {},
    'training': {},
    'optimizer': {},
    'dataset': {},
    'logging': {},
    'evaluation': {},
    'checkpointing': {},
}


# `loss` is retained as an alias-only section for backward compatibility.
AUXILIARY_SECTION_KEYS = {'loss'}
SECTION_KEYS = set(SECTION_TEMPLATE.keys()) | AUXILIARY_SECTION_KEYS
OBJECTIVES_NESTED_SECTION_KEYS = {'objectives', 'lambda'}
TRAINING_NESTED_SECTION_KEYS = {'freeze_schedule'}
CHECKPOINTING_GROUP_KEYS = set(CHECKPOINT_GROUPS)
CHECKPOINTING_METRIC_KEYS = {'name', 'mode'}
CHECKPOINTING_SAVE_KEYS = {'dir', 'save_latest', 'save_best', 'keep_last_n', 'artifacts'}
CHECKPOINTING_LOAD_KEYS = {'enabled', 'strict', 'sources'}
CHECKPOINTING_AUTHORITY_VALIDATION_KEYS = {'enabled', 'strict', 'warn_only', 'allow_fallback_row_name_classification'}
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
    ('training', 'freeze_prototype'): 'training.freeze_prototype was replaced by explicit module freeze flags (training.freeze_prototype_bank/training.freeze_prototype_projector/training.freeze_routing/training.freeze_fusion).',
    ('training', 'freeze_proxy'): 'training.freeze_proxy was replaced by explicit module freeze flags (training.freeze_prototype_bank/training.freeze_prototype_projector/training.freeze_routing/training.freeze_fusion).',
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
    ('model', 'runtime_mode'): ('auto', 'host_only', 'prototype_only', 'fused_external', 'joint_training', 'calibration_only'),
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
    ('objectives', 'objectives', 'retrieval_mode'): ('surrogate_i2t', 'clip_bidirectional'),
    ('loss', 'retrieval_mode'): ('surrogate_i2t', 'clip_bidirectional'),
    ('training', 'amp_dtype'): tuple(AMP_DTYPE_ALIASES.keys()),
    ('training', 'stage'): ('stage0', 'stage1', 'stage2', 'stage3', 'joint'),
    ('optimizer', 'type'): ('SGD', 'Adam', 'AdamW'),
    ('optimizer', 'scheduler'): ('step', 'exp', 'poly', 'cosine', 'linear'),
    ('dataset', 'dataset_name'): ('CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid'),
    ('dataset', 'val_dataset'): ('val', 'test'),
    ('evaluation', 'target_domain'): ('CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid'),
    ('evaluation', 'retrieval_scorer'): ('exact', 'approximate'),
}

RUNTIME_ENUM_CHOICES: Dict[str, Tuple[str, ...]] = {
    'training_mode': ('pas', 'vanilla_clip'),
    'runtime_mode': ('auto', 'host_only', 'prototype_only', 'fused_external', 'joint_training', 'calibration_only'),
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
    'amp_dtype': tuple(AMP_DTYPE_ALIASES.keys()),
    'training_stage': ('stage0', 'stage1', 'stage2', 'stage3', 'joint'),
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


FUSION_WEIGHT_SUM_TOLERANCE = 1e-6


def _coerce_float(field_name: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{field_name} must be a float, got {value!r}.')


def _validate_fusion_weight_value(field_name: str, value: Any) -> float:
    scalar = _coerce_float(field_name, value)
    if scalar < 0.0 or scalar > 1.0:
        raise ValueError(f'{field_name} must be within [0, 1], got {scalar}.')
    return scalar


def _validate_fusion_weight_pair(
    field_prefix: str,
    lambda_host: Any,
    lambda_prototype: Any,
    require_unit_sum: bool = True,
) -> Tuple[float, float]:
    host_weight = _validate_fusion_weight_value(f'{field_prefix}.lambda_host', lambda_host)
    prototype_weight = _validate_fusion_weight_value(f'{field_prefix}.lambda_prototype', lambda_prototype)
    if require_unit_sum:
        pair_sum = host_weight + prototype_weight
        if abs(pair_sum - 1.0) > FUSION_WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f'{field_prefix}.lambda_host + {field_prefix}.lambda_prototype must equal 1.0 '
                f'(tolerance={FUSION_WEIGHT_SUM_TOLERANCE}), got {pair_sum}.'
            )
    return host_weight, prototype_weight


def _validate_fusion_eval_subsets(field_name: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise ValueError(f'{field_name} must be a list of subset mappings.')
    for subset_index, subset in enumerate(value):
        subset_prefix = f'{field_name}[{subset_index}]'
        if not isinstance(subset, dict):
            raise ValueError(f'{subset_prefix} must be a mapping.')
        unknown_keys = sorted(set(subset.keys()) - {'name', 'lambda_host', 'lambda_prototype'})
        if unknown_keys:
            raise ValueError(
                f'Unknown keys in {subset_prefix}: {unknown_keys}. '
                'Allowed keys: ["name", "lambda_host", "lambda_prototype"]'
            )
        if 'name' in subset and subset['name'] is not None and not isinstance(subset['name'], str):
            raise ValueError(f'{subset_prefix}.name must be a string when provided.')
        if 'lambda_host' not in subset or 'lambda_prototype' not in subset:
            raise ValueError(f'{subset_prefix} must include both lambda_host and lambda_prototype.')
        _validate_fusion_weight_pair(
            field_prefix=subset_prefix,
            lambda_host=subset['lambda_host'],
            lambda_prototype=subset['lambda_prototype'],
            require_unit_sum=True,
        )


def _validate_fusion_config_data(config_data: Dict[str, Any]) -> None:
    fusion_cfg = config_data.get('fusion')
    if not isinstance(fusion_cfg, dict):
        return

    has_lambda_host = 'lambda_host' in fusion_cfg
    has_lambda_prototype = 'lambda_prototype' in fusion_cfg
    has_legacy_coefficient = 'coefficient' in fusion_cfg

    if has_lambda_host or has_lambda_prototype:
        if not (has_lambda_host and has_lambda_prototype):
            raise ValueError('fusion.lambda_host and fusion.lambda_prototype must be provided together.')
        _validate_fusion_weight_pair(
            field_prefix='fusion',
            lambda_host=fusion_cfg.get('lambda_host'),
            lambda_prototype=fusion_cfg.get('lambda_prototype'),
            require_unit_sum=True,
        )
    elif has_legacy_coefficient:
        _validate_fusion_weight_value('fusion.coefficient', fusion_cfg.get('coefficient'))

    _validate_fusion_eval_subsets('fusion.eval_subsets', fusion_cfg.get('eval_subsets'))


def _validate_runtime_fusion_args(args: Any) -> None:
    if bool(getattr(args, 'fusion_legacy_coefficient_mode', False)):
        if hasattr(args, 'fusion_coefficient') and getattr(args, 'fusion_coefficient') is not None:
            _validate_fusion_weight_value('fusion_coefficient', getattr(args, 'fusion_coefficient'))
        _validate_fusion_eval_subsets('fusion_eval_subsets', getattr(args, 'fusion_eval_subsets', None))
        return

    has_lambda_host = hasattr(args, 'fusion_lambda_host') and getattr(args, 'fusion_lambda_host') is not None
    has_lambda_prototype = hasattr(args, 'fusion_lambda_prototype') and getattr(args, 'fusion_lambda_prototype') is not None
    has_legacy_coefficient = hasattr(args, 'fusion_coefficient') and getattr(args, 'fusion_coefficient') is not None

    if has_lambda_host or has_lambda_prototype:
        if not (has_lambda_host and has_lambda_prototype):
            raise ValueError('fusion_lambda_host and fusion_lambda_prototype must be provided together.')
        _validate_fusion_weight_pair(
            field_prefix='fusion',
            lambda_host=getattr(args, 'fusion_lambda_host'),
            lambda_prototype=getattr(args, 'fusion_lambda_prototype'),
            require_unit_sum=True,
        )
    elif has_legacy_coefficient:
        _validate_fusion_weight_value('fusion_coefficient', getattr(args, 'fusion_coefficient'))

    _validate_fusion_eval_subsets('fusion_eval_subsets', getattr(args, 'fusion_eval_subsets', None))


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


def _validate_checkpointing_section(checkpointing_value: Any) -> None:
    if not isinstance(checkpointing_value, dict):
        raise ValueError('checkpointing must contain a mapping.')

    allowed_root_keys = {'metric', 'groups', 'save', 'load', 'authority_validation'}
    unknown_root_keys = sorted(set(checkpointing_value.keys()) - allowed_root_keys)
    if unknown_root_keys:
        raise ValueError(
            f'Unknown checkpointing keys: {unknown_root_keys}. Allowed keys: {sorted(allowed_root_keys)}'
        )

    metric_cfg = checkpointing_value.get('metric')
    if metric_cfg is not None:
        if not isinstance(metric_cfg, dict):
            raise ValueError('checkpointing.metric must be a mapping.')
        unknown_metric_keys = sorted(set(metric_cfg.keys()) - CHECKPOINTING_METRIC_KEYS)
        if unknown_metric_keys:
            raise ValueError(
                f'Unknown checkpointing.metric keys: {unknown_metric_keys}. Allowed keys: {sorted(CHECKPOINTING_METRIC_KEYS)}'
            )
        metric_name = metric_cfg.get('name')
        if metric_name is not None and not isinstance(metric_name, str):
            raise ValueError('checkpointing.metric.name must be a string.')
        metric_mode = metric_cfg.get('mode')
        if metric_mode is not None and str(metric_mode).lower() not in {'max', 'min'}:
            raise ValueError('checkpointing.metric.mode must be one of: max, min.')

    groups_cfg = checkpointing_value.get('groups')
    if groups_cfg is not None:
        if not isinstance(groups_cfg, dict):
            raise ValueError('checkpointing.groups must be a mapping.')
        unknown_groups = sorted(set(groups_cfg.keys()) - CHECKPOINTING_GROUP_KEYS)
        if unknown_groups:
            raise ValueError(
                f'Unknown checkpointing.groups keys: {unknown_groups}. Allowed keys: {sorted(CHECKPOINTING_GROUP_KEYS)}'
            )
        for group_name, group_cfg in groups_cfg.items():
            if not isinstance(group_cfg, dict):
                raise ValueError(f'checkpointing.groups.{group_name} must be a mapping.')
            unknown_group_keys = sorted(set(group_cfg.keys()) - {'enabled'})
            if unknown_group_keys:
                raise ValueError(
                    f'Unknown checkpointing.groups.{group_name} keys: {unknown_group_keys}. Allowed keys: ["enabled"]'
                )

    save_cfg = checkpointing_value.get('save')
    if save_cfg is not None:
        if not isinstance(save_cfg, dict):
            raise ValueError('checkpointing.save must be a mapping.')
        unknown_save_keys = sorted(set(save_cfg.keys()) - CHECKPOINTING_SAVE_KEYS)
        if unknown_save_keys:
            raise ValueError(
                f'Unknown checkpointing.save keys: {unknown_save_keys}. Allowed keys: {sorted(CHECKPOINTING_SAVE_KEYS)}'
            )
        artifacts_cfg = save_cfg.get('artifacts')
        if artifacts_cfg is not None:
            if not isinstance(artifacts_cfg, dict):
                raise ValueError('checkpointing.save.artifacts must be a mapping.')
            unknown_artifact_groups = sorted(set(artifacts_cfg.keys()) - CHECKPOINTING_GROUP_KEYS)
            if unknown_artifact_groups:
                raise ValueError(
                    f'Unknown checkpointing.save.artifacts groups: {unknown_artifact_groups}. '
                    f'Allowed keys: {sorted(CHECKPOINTING_GROUP_KEYS)}'
                )
            for group_name, artifact_cfg in artifacts_cfg.items():
                if not isinstance(artifact_cfg, dict):
                    raise ValueError(f'checkpointing.save.artifacts.{group_name} must be a mapping.')
                unknown_artifact_keys = sorted(set(artifact_cfg.keys()) - {'enabled', 'filename_latest', 'filename_best'})
                if unknown_artifact_keys:
                    raise ValueError(
                        f'Unknown checkpointing.save.artifacts.{group_name} keys: {unknown_artifact_keys}. '
                        'Allowed keys: ["enabled", "filename_latest", "filename_best"]'
                    )

    load_cfg = checkpointing_value.get('load')
    if load_cfg is not None:
        if not isinstance(load_cfg, dict):
            raise ValueError('checkpointing.load must be a mapping.')
        unknown_load_keys = sorted(set(load_cfg.keys()) - CHECKPOINTING_LOAD_KEYS)
        if unknown_load_keys:
            raise ValueError(
                f'Unknown checkpointing.load keys: {unknown_load_keys}. Allowed keys: {sorted(CHECKPOINTING_LOAD_KEYS)}'
            )
        sources_cfg = load_cfg.get('sources')
        if sources_cfg is not None:
            if not isinstance(sources_cfg, dict):
                raise ValueError('checkpointing.load.sources must be a mapping.')
            unknown_source_groups = sorted(set(sources_cfg.keys()) - CHECKPOINTING_GROUP_KEYS)
            if unknown_source_groups:
                raise ValueError(
                    f'Unknown checkpointing.load.sources groups: {unknown_source_groups}. '
                    f'Allowed keys: {sorted(CHECKPOINTING_GROUP_KEYS)}'
                )
            for group_name, source_cfg in sources_cfg.items():
                if not isinstance(source_cfg, dict):
                    raise ValueError(f'checkpointing.load.sources.{group_name} must be a mapping.')
                unknown_source_keys = sorted(set(source_cfg.keys()) - {'enabled', 'path'})
                if unknown_source_keys:
                    raise ValueError(
                        f'Unknown checkpointing.load.sources.{group_name} keys: {unknown_source_keys}. '
                        'Allowed keys: ["enabled", "path"]'
                    )

    authority_cfg = checkpointing_value.get('authority_validation')
    if authority_cfg is not None:
        if not isinstance(authority_cfg, dict):
            raise ValueError('checkpointing.authority_validation must be a mapping.')
        unknown_authority_keys = sorted(set(authority_cfg.keys()) - CHECKPOINTING_AUTHORITY_VALIDATION_KEYS)
        if unknown_authority_keys:
            raise ValueError(
                f'Unknown checkpointing.authority_validation keys: {unknown_authority_keys}. '
                f'Allowed keys: {sorted(CHECKPOINTING_AUTHORITY_VALIDATION_KEYS)}'
            )
        for key in CHECKPOINTING_AUTHORITY_VALIDATION_KEYS:
            if key in authority_cfg and not isinstance(authority_cfg[key], bool):
                raise ValueError(f'checkpointing.authority_validation.{key} must be a boolean.')


def _validate_supported_keys(config_data: Dict[str, Any]) -> None:
    supported_paths = set(PRIMARY_CONFIG_KEY_MAP.keys()) | set(READ_ALIAS_CONFIG_KEY_MAP.keys())
    supported_leafs = {(path[0], path[1]) for path in supported_paths if len(path) == 2}

    for section_name, section_value in config_data.items():
        if not isinstance(section_value, dict):
            raise ValueError(f'Config section `{section_name}` must contain a mapping.')

        for key, value in section_value.items():
            path = (section_name, key)

            if section_name == 'checkpointing':
                # Full nested validation is handled at section-level because this config is intentionally hierarchical.
                _validate_checkpointing_section(section_value)
                break

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

            if section_name == 'objectives' and key in OBJECTIVES_NESTED_SECTION_KEYS:
                if not isinstance(value, dict):
                    raise ValueError(f'objectives.{key} must be a mapping.')
                for nested_key, nested_value in value.items():
                    nested_path = (section_name, key, nested_key)
                    if isinstance(nested_value, dict):
                        raise ValueError(f'Unsupported nested config mapping at `objectives.{key}.{nested_key}`.')
                    if nested_path not in supported_paths and nested_path not in UNSUPPORTED_CONFIG_PATHS:
                        raise ValueError(f'Unknown config key `objectives.{key}.{nested_key}`.')
                continue

            if section_name == 'training' and key in TRAINING_NESTED_SECTION_KEYS:
                if key == 'freeze_schedule':
                    if value is None:
                        continue
                    if not isinstance(value, list):
                        raise ValueError('training.freeze_schedule must be a list of phase mappings.')
                    for phase_index, phase in enumerate(value):
                        if not isinstance(phase, dict):
                            raise ValueError(
                                f'training.freeze_schedule[{phase_index}] must be a mapping.'
                            )
                continue

            if section_name == 'fusion' and key == 'eval_subsets':
                _validate_fusion_eval_subsets('fusion.eval_subsets', value)
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
    _validate_fusion_config_data(config_data)

    flat = flatten_config_dict(config_data)
    host_type = str(flat.get('host_type', 'clip')).lower()
    use_prototype_branch = bool(flat.get('use_prototype_branch', False))
    if not use_prototype_branch and (
        bool(flat.get('use_prototype_bank', False))
        or bool(flat.get('use_image_conditioned_pooling', False))
    ):
        use_prototype_branch = True
    use_prototype_bank = bool(flat.get('use_prototype_bank', use_prototype_branch))
    use_image_conditioned_pooling = bool(flat.get('use_image_conditioned_pooling', use_prototype_branch))
    retrieval_mode = str(flat.get('retrieval_mode', 'surrogate_i2t')).lower()

    if not use_prototype_branch:
        if use_prototype_bank:
            raise ValueError('model.use_prototype_branch=false requires model.use_prototype_bank=false.')
        if use_image_conditioned_pooling:
            raise ValueError('model.use_prototype_branch=false requires model.use_image_conditioned_pooling=false.')

    if host_type == 'clip' and not use_prototype_branch:
        if str(flat.get('token_policy', 'eos_only')).lower() != 'eos_only':
            raise ValueError('host.type=clip with model.use_prototype_branch=false requires text_pooling.token_policy=eos_only.')
        if retrieval_mode != 'clip_bidirectional':
            raise ValueError('host.type=clip with model.use_prototype_branch=false requires objectives.objectives.retrieval_mode=clip_bidirectional.')

        if str(flat.get('retrieval_scorer', 'exact')).lower() != 'exact':
            raise ValueError('host.type=clip with model.use_prototype_branch=false requires evaluation.retrieval_scorer=exact.')
        incompatible_flags = {
            'objectives.objectives.use_loss_proxy_image': bool(flat.get('use_loss_proxy_image', False)),
            'objectives.objectives.use_loss_proxy_text': bool(flat.get('use_loss_proxy_text', False)),
            'objectives.objectives.use_loss_proxy_text_exact': bool(flat.get('use_loss_proxy_text_exact', False)),
            'objectives.objectives.use_loss_align': bool(flat.get('use_loss_align', False)),
            'objectives.objectives.use_loss_dir': bool(flat.get('use_loss_dir', False)),
            'objectives.objectives.use_loss_gap': bool(flat.get('use_loss_gap', False)),
            'objectives.objectives.use_loss_sup': bool(flat.get('use_loss_sup', False)),
            'objectives.objectives.use_loss_diag': bool(flat.get('use_loss_diag', False)),
            'objectives.objectives.use_loss_support': bool(flat.get('use_loss_support', False)),
            'objectives.objectives.use_balancing_loss': bool(flat.get('use_balancing_loss', False)),
            'objectives.objectives.use_diversity_loss': bool(flat.get('use_diversity_loss', False)),
        }
        enabled_incompatible = sorted(name for name, enabled in incompatible_flags.items() if enabled)
        if enabled_incompatible:
            raise ValueError(
                'host.type=clip with model.use_prototype_branch=false does not support prototype/auxiliary losses. '
                f'Disable: {enabled_incompatible}.'
            )
    if use_prototype_branch and use_prototype_bank and not use_image_conditioned_pooling:
        raise ValueError(
            'model.use_prototype_bank=true requires model.use_image_conditioned_pooling=true. '
            'Prototype-routed training with text-only pooling is no longer supported.'
        )

    training_config = config_data.get('training', {})
    if isinstance(training_config, dict) and 'freeze_schedule' in training_config:
        parse_freeze_schedule_config(
            training_config.get('freeze_schedule'),
            num_epoch=flat.get('num_epoch'),
        )



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
    _validate_runtime_fusion_args(args)
    parse_freeze_schedule_config(
        getattr(args, 'freeze_schedule', None),
        num_epoch=getattr(args, 'num_epoch', None),
    )

    host_type = str(getattr(args, 'host_type', 'clip')).lower()
    use_prototype_branch = bool(getattr(args, 'use_prototype_branch', False))
    if not use_prototype_branch and (
        bool(getattr(args, 'use_prototype_bank', False))
        or bool(getattr(args, 'use_image_conditioned_pooling', False))
    ):
        use_prototype_branch = True
    use_prototype_bank = bool(getattr(args, 'use_prototype_bank', use_prototype_branch))
    use_image_conditioned_pooling = bool(getattr(args, 'use_image_conditioned_pooling', use_prototype_branch))
    retrieval_mode = str(getattr(args, 'retrieval_mode', 'surrogate_i2t')).lower()

    if not use_prototype_branch:
        if use_prototype_bank:
            raise ValueError('use_prototype_branch=false requires use_prototype_bank=false.')
        if use_image_conditioned_pooling:
            raise ValueError('use_prototype_branch=false requires use_image_conditioned_pooling=false.')

    if host_type == 'clip' and not use_prototype_branch:
        if str(getattr(args, 'token_policy', 'eos_only')).lower() != 'eos_only':
            raise ValueError('host_type=clip with use_prototype_branch=false requires token_policy=eos_only.')
        if retrieval_mode != 'clip_bidirectional':
            raise ValueError('host_type=clip with use_prototype_branch=false requires retrieval_mode=clip_bidirectional.')
        if str(getattr(args, 'retrieval_scorer', 'exact')).lower() != 'exact':
            raise ValueError('host_type=clip with use_prototype_branch=false requires retrieval_scorer=exact.')
        incompatible_flags = {
            'use_loss_proxy_image': bool(getattr(args, 'use_loss_proxy_image', False)),
            'use_loss_proxy_text': bool(getattr(args, 'use_loss_proxy_text', False)),
            'use_loss_proxy_text_exact': bool(getattr(args, 'use_loss_proxy_text_exact', False)),
            'use_loss_align': bool(getattr(args, 'use_loss_align', False)),
            'use_loss_dir': bool(getattr(args, 'use_loss_dir', False)),
            'use_loss_gap': bool(getattr(args, 'use_loss_gap', False)),
            'use_loss_sup': bool(getattr(args, 'use_loss_sup', False)),
            'use_loss_diag': bool(getattr(args, 'use_loss_diag', False)),
            'use_loss_support': bool(getattr(args, 'use_loss_support', False)),
            'use_balancing_loss': bool(getattr(args, 'use_balancing_loss', False)),
            'use_diversity_loss': bool(getattr(args, 'use_diversity_loss', False)),
        }
        enabled_incompatible = sorted(name for name, enabled in incompatible_flags.items() if enabled)
        if enabled_incompatible:
            raise ValueError(
                'host_type=clip with use_prototype_branch=false does not support prototype/auxiliary losses. '
                f'Disable: {enabled_incompatible}.'
            )

    if use_prototype_branch and use_prototype_bank and not use_image_conditioned_pooling:
        raise ValueError(
            'use_prototype_bank=true requires use_image_conditioned_pooling=true. '
            'Prototype-routed training with text-only pooling is no longer supported.'
        )



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
    freeze_schedule = getattr(args, 'freeze_schedule', None)
    if freeze_schedule is not None:
        config.setdefault('training', {})['freeze_schedule'] = copy.deepcopy(freeze_schedule)
    checkpointing = getattr(args, 'checkpointing', None)
    if isinstance(checkpointing, dict) and checkpointing:
        config['checkpointing'] = copy.deepcopy(checkpointing)
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




