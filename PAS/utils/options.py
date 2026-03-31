import argparse
import os
import sys

from utils.config import apply_config_to_args, load_yaml_config, validate_runtime_args_namespace


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, 'default.yaml')
DEFAULT_RETRIEVAL_METRICS = ['R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum']


def _str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def build_parser():
    parser = argparse.ArgumentParser(description='PAS training and evaluation')
    parser.add_argument('--config_file', default='', help='Optional YAML config override file')

    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--output_dir', default='runs')
    parser.add_argument('--name', default='pas_v1', help='Experiment name')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--model_name', default='PAS')
    parser.add_argument('--model_variant', default='pas_v1')
    parser.add_argument('--pretrain_choice', default='ViT-B/16')
    parser.add_argument('--image_backbone', default='clip_visual')
    parser.add_argument('--text_backbone', default='clip_text_transformer')
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--projector_output_dim', '--projection_dim', dest='projection_dim', type=int, default=256)
    parser.add_argument('--projector_hidden_dim', type=int, default=512)
    parser.add_argument('--projector_dropout', type=float, default=0.0)
    parser.add_argument('--projector_type', type=str, default='mlp2')
    parser.add_argument('--normalize_projector_outputs', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--backbone_precision', type=str, default='fp16')
    parser.add_argument('--prototype_precision', type=str, default='fp32')
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--learn_logit_scale', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--img_size', type=int, nargs=2, default=(384, 128))
    parser.add_argument('--stride_size', type=int, default=16)
    parser.add_argument('--text_length', type=int, default=77)
    parser.add_argument('--vocab_size', type=int, default=49408)
    parser.add_argument('--use_prototype_bank', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_image_conditioned_pooling', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_prototype_contextualization', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--return_debug_outputs', type=_str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--prototype_num_prototypes', type=int, default=32)
    parser.add_argument('--prototype_dim', type=int, default=512)
    parser.add_argument('--prototype_init', type=str, default='normalized_random')
    parser.add_argument('--prototype_init_path', type=str, default='')
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
    parser.add_argument('--epochs', '--num_epoch', dest='num_epoch', type=int, default=60)
    parser.add_argument('--log_period', default=20, type=int)
    parser.add_argument('--eval_frequency', '--eval_period', dest='eval_period', default=1, type=int)
    parser.add_argument('--save_interval', type=int, default=1)
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
    parser.add_argument('--freeze_image_backbone', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--freeze_text_backbone', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--test', dest='training', default=True, action='store_false')

    parser.add_argument('--optimizer_type', '--optimizer', dest='optimizer', type=str, default='AdamW', help='[SGD, Adam, AdamW]')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_prototype_bank', type=float, default=1e-3)
    parser.add_argument('--lr_projectors', type=float, default=1e-3)
    parser.add_argument('--lr_logit_scale', type=float, default=1e-4)
    parser.add_argument('--lr_image_backbone', type=float, default=0.0)
    parser.add_argument('--lr_text_backbone', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--weight_decay_prototype_bank', type=float, default=1e-2)
    parser.add_argument('--weight_decay_projectors', type=float, default=5e-2)
    parser.add_argument('--weight_decay_logit_scale', type=float, default=0.0)
    parser.add_argument('--weight_decay_image_backbone', type=float, default=0.0)
    parser.add_argument('--weight_decay_text_backbone', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.999)
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

    parser.add_argument('--use_wandb', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--wandb_project', default='PAS')
    parser.add_argument('--wandb_entity', default=None)
    parser.add_argument('--wandb_run_name', default=None)
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
    args.use_prototype_bank = bool(args.use_prototype_bank)
    args.use_image_conditioned_pooling = bool(args.use_image_conditioned_pooling)
    legacy_contextualization = getattr(args, 'use_prototype_contextualization', None)
    authoritative_contextualization = getattr(args, 'prototype_contextualization_enabled', None)
    override_config_data = getattr(args, 'override_config_data', {}) or {}
    model_config = override_config_data.get('model', {}) if isinstance(override_config_data.get('model', {}), dict) else {}
    prototype_config = override_config_data.get('prototype', {}) if isinstance(override_config_data.get('prototype', {}), dict) else {}
    authoritative_from_config = 'contextualization_enabled' in prototype_config
    legacy_from_config = 'use_prototype_contextualization' in model_config
    cli_dests = getattr(args, 'cli_dests', set())
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
    args.image_backbone = args.image_backbone or args.pretrain_choice
    args.text_backbone = args.text_backbone or 'clip_text_transformer'
    return args


def get_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    default_config_path = DEFAULT_CONFIG_PATH if os.path.exists(DEFAULT_CONFIG_PATH) else None
    override_config_path = args.config_file or None
    config_data = load_yaml_config(default_config_path, override_config_path)
    override_config_data = load_yaml_config(None, override_config_path) if override_config_path else {}
    args = apply_config_to_args(parser, args, config_data, argv if argv is not None else sys.argv[1:])
    args.override_config_data = override_config_data
    args = _finalize_args(args)
    validate_runtime_args_namespace(args)
    return args

