import argparse
import copy
import os
import sys

from utils.config import apply_config_to_args, load_yaml_config, validate_runtime_args_namespace


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, 'base.yaml')
DEFAULT_RETRIEVAL_METRICS = ['R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum']
LEGACY_RETRIEVAL_FLAGS = {
    '--use_loss_ret_exact': 'Legacy exact retrieval training is removed. Use --use_loss_ret for row-wise surrogate image-to-text retrieval.',
    '--use_loss_ret_exact_image': 'Legacy exact image-side retrieval training is removed. Use --use_loss_ret for row-wise surrogate image-to-text retrieval.',
    '--use_loss_ret_exact_text': 'Legacy text-to-image retrieval training is invalid because surrogate text embeddings are image-conditioned.',
    '--lambda_ret_exact': 'Legacy exact retrieval weighting is removed. Use --lambda_ret for row-wise surrogate image-to-text retrieval.',
    '--lambda_ret_exact_image': 'Legacy exact image-side retrieval weighting is removed. Use --lambda_ret for row-wise surrogate image-to-text retrieval.',
    '--lambda_ret_exact_text': 'Legacy text-side retrieval weighting is removed. Column-wise text retrieval is invalid for image-conditioned text embeddings.',
    '--ret_exact_temperature': 'Legacy exact retrieval temperature is removed. Use --temperature for surrogate retrieval scoring.',
    '--use_loss_ret_text': 'Text-to-image retrieval loss is not supported because surrogate text embeddings are image-conditioned.',
    '--lambda_ret_text': 'Text-to-image retrieval weighting is not supported because surrogate text embeddings are image-conditioned.',
    '--bidirectional_ret': 'Bidirectional retrieval over the surrogate score matrix is invalid. Only row-wise image-to-text retrieval is supported.',
    '--clip_style_ret': 'CLIP-style symmetric retrieval over the surrogate score matrix is invalid. Only row-wise image-to-text retrieval is supported.',
    '--symmetric_ret': 'Symmetric retrieval over the surrogate score matrix is invalid. Only row-wise image-to-text retrieval is supported.',
    '--t2i_ret': 'Text-to-image retrieval over the surrogate score matrix is invalid because surrogate text embeddings depend on the image query.',
    '--freeze_prototype': 'Legacy --freeze_prototype was replaced by --freeze_prototype_side.',
    '--freeze_proxy': 'Legacy --freeze_proxy was replaced by --freeze_prototype_side.',
}


def _str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def _prevalidate_removed_cli_flags(argv):
    for token in argv or []:
        option = token.split('=', 1)[0]
        if option in LEGACY_RETRIEVAL_FLAGS:
            raise ValueError(LEGACY_RETRIEVAL_FLAGS[option])


def build_parser():
    parser = argparse.ArgumentParser(description='PAS training and evaluation')
    parser.add_argument('--config_file', default='', help='Optional YAML config override file')

    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--output_dir', default='runs')
    parser.add_argument('--name', default='pas_v1', help='Experiment name')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--model_name', default='PAS')
    parser.add_argument('--model_variant', default='pas_v1')
    parser.add_argument('--training_mode', type=str, default='pas')
    parser.add_argument('--pretrain_choice', default='ViT-B/16')
    parser.add_argument('--image_backbone', default='clip_visual')
    parser.add_argument('--text_backbone', default='clip_text_transformer')
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--projector_output_dim', '--projection_dim', dest='projection_dim', type=int, default=256)
    parser.add_argument('--projector_hidden_dim', type=int, default=512)
    parser.add_argument('--projector_dropout', type=float, default=0.0)
    parser.add_argument('--projector_type', type=str, default='mlp2')
    parser.add_argument('--normalize_projector_outputs', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_custom_projector', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--backbone_precision', type=str, default='fp16')
    parser.add_argument('--prototype_precision', type=str, default='fp32')
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--proxy_temperature', type=float, default=0.07)
    parser.add_argument('--host_type', type=str, default='clip')
    parser.add_argument('--use_host_loss', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--lambda_host', type=float, default=1.0)
    parser.add_argument('--itself_loss_names', type=str, default='tal+cid')
    parser.add_argument('--loss_names', dest='itself_loss_names', type=str)
    parser.add_argument('--itself_only_global', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--only_global', dest='itself_only_global', action='store_true')
    parser.add_argument('--itself_select_ratio', type=float, default=0.4)
    parser.add_argument('--select_ratio', dest='itself_select_ratio', type=float)
    parser.add_argument('--itself_grab_embed_dim', type=int, default=4096)
    parser.add_argument('--itself_score_weight_global', type=float, default=0.68)
    parser.add_argument('--itself_tau', type=float, default=0.015)
    parser.add_argument('--tau', dest='itself_tau', type=float)
    parser.add_argument('--itself_margin', type=float, default=0.1)
    parser.add_argument('--margin', dest='itself_margin', type=float)
    parser.add_argument('--itself_return_all', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--return_all', dest='itself_return_all', action='store_true')
    parser.add_argument('--itself_topk_type', type=str, default='mean')
    parser.add_argument('--topk_type', dest='itself_topk_type', type=str)
    parser.add_argument('--itself_layer_index', type=int, default=-1)
    parser.add_argument('--layer_index', dest='itself_layer_index', type=int)
    parser.add_argument('--itself_average_attn_weights', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--average_attn_weights', dest='itself_average_attn_weights', type=_str2bool, nargs='?', const=True)
    parser.add_argument('--itself_modify_k', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--modify_k', dest='itself_modify_k', action='store_true')
    parser.add_argument('--lambda1_weight', type=float, default=0.5)
    parser.add_argument('--lambda2_weight', type=float, default=3.5)
    parser.add_argument('--lambda_proxy', type=float, default=1.0)
    parser.add_argument('--lambda_proxy_image', type=float, default=None)
    parser.add_argument('--lambda_proxy_text', type=float, default=None)
    parser.add_argument('--lambda_proxy_text_exact', type=float, default=None)
    parser.add_argument('--use_loss_proxy_image', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_loss_proxy_text', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_loss_proxy_text_exact', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_loss_align', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--lambda_align', type=float, default=1.0)
    parser.add_argument('--use_loss_dir', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--lambda_dir', type=float, default=None)
    parser.add_argument('--use_loss_gap', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--lambda_gap', type=float, default=0.5)
    parser.add_argument('--prototype_gap_margin', type=float, default=0.05)
    parser.add_argument('--use_loss_sup', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--lambda_sup', type=float, default=None)
    parser.add_argument('--prototype_support_target', type=float, default=4.0)
    parser.add_argument('--use_loss_diag', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--lambda_diag', type=float, default=1.0)
    parser.add_argument('--diag_temperature', type=float, default=0.07)
    parser.add_argument('--use_loss_ret', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--retrieval_mode', type=str, default='surrogate_i2t')
    parser.add_argument('--lambda_ret', type=float, default=0.5)
    parser.add_argument('--use_loss_weight_ret', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--lambda_weight_ret', type=float, default=0.0)
    parser.add_argument('--weight_ret_margin_delta', type=float, default=0.0)
    parser.add_argument('--weight_ret_tau', type=float, default=0.5)
    parser.add_argument('--weight_ret_detach_host', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--weight_ret_normalize_mean_one', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_loss_support', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--lambda_support', type=float, default=0.1)
    parser.add_argument('--support_min', type=float, default=2.0)
    parser.add_argument('--img_size', type=int, nargs=2, default=(384, 128))
    parser.add_argument('--stride_size', type=int, default=16)
    parser.add_argument('--text_length', type=int, default=77)
    parser.add_argument('--vocab_size', type=int, default=49408)
    parser.add_argument('--use_prototype_branch', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--use_prototype_bank', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_image_conditioned_pooling', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--fusion_enabled', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--fusion_lambda_host', type=float, default=None)
    parser.add_argument('--fusion_lambda_prototype', type=float, default=None)
    parser.add_argument('--fusion_coefficient', type=float, default=None)
    parser.add_argument('--fusion_coefficient_source', type=str, default='fixed')
    parser.add_argument('--use_prototype_contextualization', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--return_debug_outputs', type=_str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--prototype_num_prototypes', type=int, default=32)
    parser.add_argument('--prototype_dim', type=int, default=512)
    parser.add_argument('--prototype_init', type=str, default='normalized_random')
    parser.add_argument('--prototype_init_path', type=str, default='')
    parser.add_argument('--prototype_init_hybrid_ratio', type=float, default=0.5)
    parser.add_argument('--prototype_init_max_iters', type=int, default=50)
    parser.add_argument('--prototype_init_tol', type=float, default=1e-4)
    parser.add_argument('--prototype_init_seed', type=int, default=None)
    parser.add_argument('--routing_similarity', '--prototype_routing_type', dest='prototype_routing_type', type=str, default='cosine')
    parser.add_argument('--tau_p', '--prototype_temperature', dest='prototype_temperature', type=float, default=0.07)
    parser.add_argument('--prototype_contextualization_enabled', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--prototype_contextualization_type', type=str, default='self_attention')
    parser.add_argument('--prototype_contextualization_residual', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--normalize_for_self_interaction', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--normalize_for_routing', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_balancing_loss', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--lambda_bal', '--prototype_balance_loss_weight', dest='prototype_balance_loss_weight', type=float, default=0.0)
    parser.add_argument('--prototype_dead_threshold', type=float, default=0.005)
    parser.add_argument('--use_diversity_loss', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--lambda_div', '--diversity_loss_weight', dest='diversity_loss_weight', type=float, default=0.01)

    parser.add_argument('--token_policy', type=str, default='content_only')
    parser.add_argument('--token_similarity', '--token_scoring_type', dest='token_scoring_type', type=str, default='cosine')
    parser.add_argument('--normalize_for_token_scoring', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--tau_t', '--token_pooling_temperature', dest='token_pooling_temperature', type=float, default=0.07)
    parser.add_argument('--error_on_empty_kept_tokens', type=_str2bool, nargs='?', const=True, default=True)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--stage', '--training_stage', dest='training_stage', type=str, default='joint')
    parser.add_argument('--epochs', '--num_epoch', dest='num_epoch', type=int, default=60)
    parser.add_argument('--log_period', default=50, type=int)
    parser.add_argument('--eval_frequency', '--eval_period', dest='eval_period', default=1, type=int)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument(
        '--prototype-selection-metric',
        dest='prototype_selection_metric',
        type=str,
        default=None,
        help='Deprecated. Use checkpointing.metric/checkpointing.save for modular best/latest checkpointing.',
    )
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--amp', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--amp_dtype', type=str, default='fp16')
    parser.add_argument('--resume', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--resume_ckpt_file', default='', help='Resume from checkpoint path')
    parser.add_argument('--finetune', type=str, default='')
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--img_aug', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--txt_aug', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--sampler', default='identity', help='choose sampler from [identity, random]')
    parser.add_argument('--num_instance', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--freeze_host_projectors', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--freeze_image_backbone', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--freeze_text_backbone', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--freeze_prototype_side', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--test', dest='training', default=True, action='store_false')

    parser.add_argument('--optimizer_type', '--optimizer', dest='optimizer', type=str, default='AdamW', help='[SGD, Adam, AdamW]')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_prototype_bank', type=float, default=1e-3)
    parser.add_argument('--lr_projectors', type=float, default=1e-3)
    parser.add_argument('--lr_prototype_routing', type=float, default=None)
    parser.add_argument('--lr_prototype_pooling', type=float, default=None)
    parser.add_argument('--lr_prototype_contextualization', type=float, default=None)
    parser.add_argument('--lr_host_projectors', type=float, default=None)
    parser.add_argument('--lr_image_backbone', type=float, default=0.0)
    parser.add_argument('--lr_text_backbone', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--weight_decay_prototype_bank', type=float, default=1e-2)
    parser.add_argument('--weight_decay_projectors', type=float, default=5e-2)
    parser.add_argument('--weight_decay_prototype_routing', type=float, default=None)
    parser.add_argument('--weight_decay_prototype_pooling', type=float, default=None)
    parser.add_argument('--weight_decay_prototype_contextualization', type=float, default=None)
    parser.add_argument('--weight_decay_host_projectors', type=float, default=None)
    parser.add_argument('--lr_class_proxies', type=float, default=None)
    parser.add_argument('--weight_decay_class_proxies', type=float, default=None)
    parser.add_argument('--weight_decay_image_backbone', type=float, default=0.0)
    parser.add_argument('--weight_decay_text_backbone', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.999)
    parser.add_argument('--lr_factor', type=float, default=5.0)
    parser.add_argument('--bias_lr_factor', type=float, default=2.0)
    parser.add_argument('--weight_decay_bias', type=float, default=0.0)
    parser.add_argument('--optimizer_eps', type=float, default=1e-8)
    parser.add_argument('--milestones', type=int, nargs='+', default=(40, 50))
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--warmup_factor', type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_method', type=str, default='linear')
    parser.add_argument('--lr_scheduler', '--lrscheduler', dest='lrscheduler', type=str, default='cosine')
    parser.add_argument('--target_lr', type=float, default=0.0)
    parser.add_argument('--power', type=float, default=0.9)

    parser.add_argument('--dataset_name', default='CUHK-PEDES', help='[CUHK-PEDES, ICFG-PEDES, RSTPReid]')
    parser.add_argument('--root_dir', default='data')
    parser.add_argument('--val_dataset', default='test')
    parser.add_argument('--test_batch_size', type=int, default=512)

    parser.add_argument('--checkpoint_path', '--checkpoint', dest='checkpoint', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cross_domain_generalization', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--target_domain', default='RSTPReid')
    parser.add_argument('--retrieval_metrics', nargs='+', default=list(DEFAULT_RETRIEVAL_METRICS))
    parser.add_argument('--prototype_eval_image_chunk_size', type=int, default=32)
    parser.add_argument('--prototype_eval_text_chunk_size', type=int, default=128)
    parser.add_argument('--retrieval_scorer', type=str, default='exact')

    parser.add_argument('--use_wandb', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--wandb_project', default='PAS')
    parser.add_argument('--wandb_entity', default=None)
    parser.add_argument('--wandb_run_name', default=None)
    parser.add_argument('--nohup', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--wandb_group', default=None)
    parser.add_argument('--wandb_mode', default='online')
    parser.add_argument('--wandb_tags', nargs='*', default=[])
    parser.add_argument('--wandb_notes', default=None)
    parser.add_argument('--wandb_log_interval', type=int, default=50)
    parser.add_argument('--wandb_log_code', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--log_debug_metrics', type=_str2bool, nargs='?', const=True, default=True)

    parser.set_defaults(special_token_ids=None)
    return parser


def _finalize_args(args):
    if isinstance(args.img_size, list):
        args.img_size = tuple(args.img_size)
    if isinstance(args.milestones, tuple):
        args.milestones = list(args.milestones)
    if not args.prototype_init_path:
        args.prototype_init_path = None
    if not args.retrieval_metrics:
        args.retrieval_metrics = list(DEFAULT_RETRIEVAL_METRICS)
    args.host_type = str(getattr(args, 'host_type', 'clip')).lower()
    override_config_data = getattr(args, 'override_config_data', {}) or {}
    model_config = override_config_data.get('model', {}) if isinstance(override_config_data.get('model', {}), dict) else {}
    cli_dests = getattr(args, 'cli_dests', set())
    if args.use_prototype_branch is None:
        args.use_prototype_branch = bool(
            getattr(args, 'use_prototype_bank', False)
            or getattr(args, 'use_image_conditioned_pooling', False)
        )
    elif not bool(args.use_prototype_branch) and (
        bool(getattr(args, 'use_prototype_bank', False))
        or bool(getattr(args, 'use_image_conditioned_pooling', False))
    ):
        args.use_prototype_branch = True
    args.use_prototype_branch = bool(args.use_prototype_branch)
    args.use_prototype_bank = bool(args.use_prototype_bank) if args.use_prototype_branch else False
    args.use_image_conditioned_pooling = bool(args.use_image_conditioned_pooling) if args.use_prototype_branch else False
    args.use_custom_projector = bool(getattr(args, 'use_custom_projector', True))
    args.training_mode = str(getattr(args, 'training_mode', 'pas')).lower()
    args.training_stage = str(getattr(args, 'training_stage', 'joint')).lower()
    config_data = getattr(args, 'config_data', {}) or {}
    training_config = config_data.get('training', {}) if isinstance(config_data.get('training', {}), dict) else {}
    checkpointing_config = config_data.get('checkpointing', {}) if isinstance(config_data.get('checkpointing', {}), dict) else {}
    args.checkpointing = copy.deepcopy(checkpointing_config)
    freeze_schedule = training_config.get('freeze_schedule')
    args.freeze_schedule = copy.deepcopy(freeze_schedule) if freeze_schedule is not None else None
    selection_metric = getattr(args, 'prototype_selection_metric', None)
    if selection_metric is not None:
        if isinstance(selection_metric, (list, tuple, set)):
            normalized_metrics = [str(metric).strip() for metric in selection_metric if str(metric).strip()]
            args.prototype_selection_metric = normalized_metrics if normalized_metrics else None
        else:
            selection_metric = str(selection_metric).strip()
            args.prototype_selection_metric = selection_metric if selection_metric else None
    args.retrieval_mode = str(getattr(args, 'retrieval_mode', 'surrogate_i2t')).lower()
    if args.fusion_enabled is None:
        args.fusion_enabled = args.use_prototype_branch
    args.fusion_enabled = bool(args.fusion_enabled) and args.use_prototype_branch
    fusion_lambda_host = getattr(args, 'fusion_lambda_host', None)
    fusion_lambda_prototype = getattr(args, 'fusion_lambda_prototype', None)
    fusion_coefficient = getattr(args, 'fusion_coefficient', None)
    if fusion_lambda_host is None and fusion_lambda_prototype is not None:
        fusion_lambda_host = 1.0 - float(fusion_lambda_prototype)
    elif fusion_lambda_prototype is None and fusion_lambda_host is not None:
        fusion_lambda_prototype = 1.0 - float(fusion_lambda_host)

    if fusion_lambda_host is not None or fusion_lambda_prototype is not None:
        args.fusion_lambda_host = float(fusion_lambda_host)
        args.fusion_lambda_prototype = float(fusion_lambda_prototype)
        args.fusion_legacy_coefficient_mode = False
    elif fusion_coefficient is not None:
        args.fusion_lambda_host = 1.0
        args.fusion_lambda_prototype = float(fusion_coefficient)
        args.fusion_legacy_coefficient_mode = True
    else:
        args.fusion_lambda_host = 1.0
        args.fusion_lambda_prototype = 0.0
        args.fusion_legacy_coefficient_mode = False
    if fusion_coefficient is None:
        args.fusion_coefficient = float(args.fusion_lambda_prototype)
    args.fusion_eval_subsets = copy.deepcopy(getattr(args, 'fusion_eval_subsets', None))

    legacy_contextualization = getattr(args, 'use_prototype_contextualization', None)
    authoritative_contextualization = getattr(args, 'prototype_contextualization_enabled', None)
    prototype_config = override_config_data.get('prototype', {}) if isinstance(override_config_data.get('prototype', {}), dict) else {}
    authoritative_from_config = 'contextualization_enabled' in prototype_config
    legacy_from_config = 'use_prototype_contextualization' in model_config
    authoritative_from_cli = 'prototype_contextualization_enabled' in cli_dests
    if authoritative_from_cli:
        authoritative_contextualization = bool(authoritative_contextualization)
    elif authoritative_contextualization is None:
        authoritative_contextualization = True if legacy_contextualization is None else bool(legacy_contextualization)
    elif legacy_from_config and not authoritative_from_config:
        authoritative_contextualization = bool(legacy_contextualization)
    else:
        authoritative_contextualization = bool(authoritative_contextualization)
    args.prototype_contextualization_enabled = authoritative_contextualization
    args.use_prototype_contextualization = authoritative_contextualization
    args.freeze_backbones = bool(args.freeze_image_backbone and args.freeze_text_backbone)
    args.lambda_proxy_image = args.lambda_proxy if args.lambda_proxy_image is None else float(args.lambda_proxy_image)
    args.lambda_proxy_text = args.lambda_proxy if args.lambda_proxy_text is None else float(args.lambda_proxy_text)
    args.lambda_proxy_text_exact = args.lambda_proxy if args.lambda_proxy_text_exact is None else float(args.lambda_proxy_text_exact)
    args.lambda_dir = float(args.lambda_diag) if getattr(args, 'lambda_dir', None) is None else float(args.lambda_dir)
    args.use_loss_dir = bool(args.lambda_dir > 0.0) if getattr(args, 'use_loss_dir', None) is None else bool(args.use_loss_dir)
    args.use_loss_diag = bool(args.use_loss_dir)
    args.lambda_diag = float(args.lambda_dir)
    args.diag_temperature = float(getattr(args, 'diag_temperature', 0.07))
    args.lambda_sup = float(args.lambda_support) if getattr(args, 'lambda_sup', None) is None else float(args.lambda_sup)
    args.use_loss_sup = bool(args.lambda_sup > 0.0) if getattr(args, 'use_loss_sup', None) is None else bool(args.use_loss_sup)
    args.use_loss_support = bool(args.use_loss_sup)
    args.lambda_support = float(args.lambda_sup)
    args.use_loss_gap = bool(getattr(args, 'use_loss_gap', True))
    args.lambda_gap = float(getattr(args, 'lambda_gap', 0.5))
    args.prototype_gap_margin = float(getattr(args, 'prototype_gap_margin', 0.05))
    args.prototype_support_target = float(getattr(args, 'prototype_support_target', getattr(args, 'support_min', 4.0)))
    args.support_min = float(args.prototype_support_target)
    args.use_loss_ret = bool(args.use_loss_ret)
    args.lambda_ret = float(args.lambda_ret)
    args.use_loss_weight_ret = bool(getattr(args, 'use_loss_weight_ret', False))
    args.lambda_weight_ret = float(getattr(args, 'lambda_weight_ret', 0.0))
    args.weight_ret_margin_delta = float(getattr(args, 'weight_ret_margin_delta', 0.0))
    args.weight_ret_tau = float(getattr(args, 'weight_ret_tau', 0.5))
    args.weight_ret_detach_host = bool(getattr(args, 'weight_ret_detach_host', True))
    args.weight_ret_normalize_mean_one = bool(getattr(args, 'weight_ret_normalize_mean_one', True))
    args.image_backbone = args.image_backbone or args.pretrain_choice
    args.text_backbone = args.text_backbone or 'clip_text_transformer'
    return args


def get_args(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    _prevalidate_removed_cli_flags(argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    default_config_path = DEFAULT_CONFIG_PATH if os.path.exists(DEFAULT_CONFIG_PATH) else None
    override_config_path = args.config_file or None
    config_data = load_yaml_config(default_config_path, override_config_path)
    override_config_data = load_yaml_config(None, override_config_path) if override_config_path else {}
    args = apply_config_to_args(
        parser,
        args,
        config_data,
        argv,
        override_config_data=override_config_data,
    )
    args.override_config_data = override_config_data
    args = _finalize_args(args)
    validate_runtime_args_namespace(args)
    return args

