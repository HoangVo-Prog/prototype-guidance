import copy
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


PRIMARY_CONFIG_KEY_MAP: Dict[Tuple[str, ...], str] = {
    ('experiment', 'name'): 'name',
    ('experiment', 'output_dir'): 'output_dir',
    ('experiment', 'seed'): 'seed',
    ('experiment', 'local_rank'): 'local_rank',

    ('model', 'name'): 'model_name',
    ('model', 'variant'): 'model_variant',
    ('model', 'pretrain_choice'): 'pretrain_choice',
    ('model', 'image_backbone'): 'image_backbone',
    ('model', 'text_backbone'): 'text_backbone',
    ('model', 'embedding_dim'): 'embedding_dim',
    ('model', 'projection_dim'): 'projection_dim',
    ('model', 'projector_hidden_dim'): 'projector_hidden_dim',
    ('model', 'projector_dropout'): 'projector_dropout',
    ('model', 'temperature'): 'temperature',
    ('model', 'pooling_mode'): 'pooling_mode',
    ('model', 'img_size'): 'img_size',
    ('model', 'stride_size'): 'stride_size',
    ('model', 'text_length'): 'text_length',
    ('model', 'vocab_size'): 'vocab_size',
    ('model', 'use_prototype_bank'): 'use_prototype_bank',
    ('model', 'use_image_conditioned_pooling'): 'use_image_conditioned_pooling',
    ('model', 'use_prototype_contextualization'): 'use_prototype_contextualization',
    ('model', 'return_debug_outputs'): 'return_debug_outputs',

    ('prototype', 'num_prototypes'): 'prototype_num_prototypes',
    ('prototype', 'prototype_dim'): 'prototype_dim',
    ('prototype', 'prototype_init'): 'prototype_init',
    ('prototype', 'prototype_init_path'): 'prototype_init_path',
    ('prototype', 'routing_type'): 'prototype_routing_type',
    ('prototype', 'routing_temperature'): 'prototype_temperature',
    ('prototype', 'contextualization_enabled'): 'prototype_contextualization_enabled',
    ('prototype', 'contextualization_type'): 'prototype_contextualization_type',
    ('prototype', 'contextualization_residual'): 'prototype_contextualization_residual',
    ('prototype', 'contextualization_num_layers'): 'prototype_contextualization_num_layers',
    ('prototype', 'prototype_normalize'): 'prototype_normalize',
    ('prototype', 'assignment_sparse'): 'prototype_sparse_assignment',
    ('prototype', 'assignment_topk'): 'prototype_sparse_topk',
    ('prototype', 'balance_loss_weight'): 'prototype_balance_loss_weight',
    ('prototype', 'dead_prototype_threshold'): 'prototype_dead_threshold',
    ('prototype', 'use_diversity_loss'): 'use_diversity_loss',
    ('prototype', 'diversity_loss_weight'): 'diversity_loss_weight',

    ('text_pooling', 'token_policy'): 'token_policy',
    ('text_pooling', 'scoring_type'): 'token_scoring_type',
    ('text_pooling', 'token_temperature'): 'token_pooling_temperature',
    ('text_pooling', 'exclude_special_tokens'): 'exclude_special_tokens',
    ('text_pooling', 'eos_as_only_token'): 'eos_as_only_token',
    ('text_pooling', 'mask_padding_tokens'): 'mask_padding_tokens',

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
    ('training', 'freeze_image_backbone'): 'freeze_image_backbone',
    ('training', 'freeze_text_backbone'): 'freeze_text_backbone',
    ('training', 'grad_clip'): 'grad_clip',
    ('training', 'amp'): 'amp',

    ('optimizer', 'type'): 'optimizer',
    ('optimizer', 'lr'): 'lr',
    ('optimizer', 'lr_prototype_bank'): 'lr_prototype_bank',
    ('optimizer', 'lr_contextualizer'): 'lr_contextualizer',
    ('optimizer', 'lr_projectors'): 'lr_projectors',
    ('optimizer', 'lr_logit_scale'): 'lr_logit_scale',
    ('optimizer', 'lr_image_backbone'): 'lr_image_backbone',
    ('optimizer', 'lr_text_backbone'): 'lr_text_backbone',
    ('optimizer', 'weight_decay'): 'weight_decay',
    ('optimizer', 'weight_decay_prototype_bank'): 'weight_decay_prototype_bank',
    ('optimizer', 'weight_decay_contextualizer'): 'weight_decay_contextualizer',
    ('optimizer', 'weight_decay_projectors'): 'weight_decay_projectors',
    ('optimizer', 'weight_decay_logit_scale'): 'weight_decay_logit_scale',
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
}


READ_ALIAS_CONFIG_KEY_MAP: Dict[Tuple[str, ...], str] = {
    ('training', 'num_epoch'): 'num_epoch',
    ('training', 'eval_period'): 'eval_period',
    ('optimizer', 'optimizer'): 'optimizer',
    ('optimizer', 'lrscheduler'): 'lrscheduler',
    ('evaluation', 'checkpoint'): 'checkpoint',
    ('evaluation', 'test_batch_size'): 'test_batch_size',
    ('model', 'projector_output_dim'): 'projection_dim',
    ('prototype', 'routing_similarity'): 'prototype_routing_type',
    ('prototype', 'tau_p'): 'prototype_temperature',
    ('prototype', 'lambda_bal'): 'prototype_balance_loss_weight',
    ('prototype', 'lambda_div'): 'diversity_loss_weight',
    ('text_pooling', 'token_similarity'): 'token_scoring_type',
    ('text_pooling', 'tau_t'): 'token_pooling_temperature',
    ('optimizer', 'weight_decay_prototypes'): 'weight_decay_prototype_bank',
    ('optimizer', 'weight_decay_projectors'): 'weight_decay_projectors',
}


SECTION_TEMPLATE = {
    'experiment': {},
    'model': {},
    'prototype': {},
    'text_pooling': {},
    'training': {},
    'optimizer': {},
    'dataset': {},
    'logging': {},
    'evaluation': {},
}


SECTION_KEYS = set(SECTION_TEMPLATE.keys())


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


def load_yaml_config(default_path: Optional[str] = None, override_path: Optional[str] = None) -> Dict[str, Any]:
    config = {}
    if default_path:
        config = deep_merge_dicts(config, _read_yaml(default_path))
    if override_path:
        config = deep_merge_dicts(config, _read_yaml(override_path))
    return config


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


def apply_config_to_args(parser, args, config_data: Dict[str, Any], argv: Optional[List[str]] = None):
    cli_dests = _extract_cli_destinations(parser, argv or [])
    applied_dests = set()
    for mapping in (PRIMARY_CONFIG_KEY_MAP, READ_ALIAS_CONFIG_KEY_MAP):
        for _, dest, value in _iter_config_value_paths(mapping, config_data):
            if dest in cli_dests or dest in applied_dests:
                continue
            setattr(args, dest, _normalize_value(dest, value))
            applied_dests.add(dest)
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
