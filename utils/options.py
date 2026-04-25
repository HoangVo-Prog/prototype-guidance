import argparse
import copy
import os
import sys

from utils.config import apply_config_to_args, load_yaml_config, validate_runtime_args_namespace


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, 'base.yaml')
DEFAULT_RETRIEVAL_METRICS = ['R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum']
LEGACY_RETRIEVAL_FLAGS = {
    '--use_loss_ret_exact': 'Legacy exact retrieval training is removed.',
    '--use_loss_ret_exact_image': 'Legacy exact image-side retrieval training is removed.',
    '--use_loss_ret_exact_text': 'Legacy text-to-image retrieval training is invalid because surrogate text embeddings are image-conditioned.',
    '--lambda_ret_exact': 'Legacy exact retrieval weighting is removed.',
    '--lambda_ret_exact_image': 'Legacy exact image-side retrieval weighting is removed.',
    '--lambda_ret_exact_text': 'Legacy text-side retrieval weighting is removed. Column-wise text retrieval is invalid for image-conditioned text embeddings.',
    '--ret_exact_temperature': 'Legacy exact retrieval temperature is removed. Use --temperature for surrogate retrieval scoring.',
    '--use_loss_ret_text': 'Text-to-image retrieval loss is not supported because surrogate text embeddings are image-conditioned.',
    '--lambda_ret_text': 'Text-to-image retrieval weighting is not supported because surrogate text embeddings are image-conditioned.',
    '--bidirectional_ret': 'Bidirectional retrieval over the surrogate score matrix is invalid. Only row-wise image-to-text retrieval is supported.',
    '--clip_style_ret': 'CLIP-style symmetric retrieval over the surrogate score matrix is invalid. Only row-wise image-to-text retrieval is supported.',
    '--symmetric_ret': 'Symmetric retrieval over the surrogate score matrix is invalid. Only row-wise image-to-text retrieval is supported.',
    '--t2i_ret': 'Text-to-image retrieval over the surrogate score matrix is invalid because surrogate text embeddings depend on the image query.',
    '--freeze_prototype': 'Legacy --freeze_prototype was replaced by explicit module freeze flags (freeze_prototype_bank/freeze_prototype_projector/freeze_routing/freeze_fusion).',
    '--freeze_proxy': 'Legacy --freeze_proxy was replaced by explicit module freeze flags (freeze_prototype_bank/freeze_prototype_projector/freeze_routing/freeze_fusion).',
    '--fusion_enabled': 'Fusion-based retrieval was removed. HostCore is the only retrieval scorer.',
    '--fusion_lambda_host': 'Fusion-based retrieval was removed. HostCore is the only retrieval scorer.',
    '--fusion_lambda_prototype': 'Fusion-based retrieval was removed. HostCore is the only retrieval scorer.',
    '--fusion_coefficient': 'Fusion-based retrieval was removed. HostCore is the only retrieval scorer.',
    '--fusion_coefficient_source': 'Fusion-based retrieval was removed. HostCore is the only retrieval scorer.',
    '--composer_calibration_enabled': 'Composer calibration was removed from active runtime semantics.',
    '--retrieval_scorer': 'evaluation.retrieval_scorer was removed. Retrieval scoring is always exact host-only.',
    '--prototype_inference_mode': 'prototype_inference_mode was removed. Retrieval inference is always host_only.',
    '--lambda_proxy': 'loss_proxy was removed from runtime semantics.',
    '--lambda_proxy_image': 'loss_proxy was removed from runtime semantics.',
    '--lambda_proxy_text': 'loss_proxy was removed from runtime semantics.',
    '--lambda_proxy_text_exact': 'loss_proxy was removed from runtime semantics.',
    '--use_loss_proxy_image': 'loss_proxy was removed from runtime semantics.',
    '--use_loss_proxy_text': 'loss_proxy was removed from runtime semantics.',
    '--use_loss_proxy_text_exact': 'loss_proxy was removed from runtime semantics.',
    '--use_loss_dir': 'loss_dir was removed from runtime semantics.',
    '--lambda_dir': 'loss_dir was removed from runtime semantics.',
    '--use_loss_gap': 'loss_gap was removed from runtime semantics.',
    '--lambda_gap': 'loss_gap was removed from runtime semantics.',
    '--prototype_gap_margin': 'loss_gap was removed from runtime semantics.',
    '--use_loss_sup': 'loss_sup/loss_support were removed from runtime semantics.',
    '--lambda_sup': 'loss_sup/loss_support were removed from runtime semantics.',
    '--use_loss_support': 'loss_sup/loss_support were removed from runtime semantics.',
    '--lambda_support': 'loss_sup/loss_support were removed from runtime semantics.',
    '--prototype_support_target': 'loss_sup/loss_support were removed from runtime semantics.',
    '--support_min': 'loss_sup/loss_support were removed from runtime semantics.',
    '--use_loss_ret': 'Prototype-side loss_ret was removed from runtime semantics.',
    '--lambda_ret': 'Prototype-side loss_ret was removed from runtime semantics.',
    '--use_loss_align': 'loss_align was removed from runtime semantics.',
    '--lambda_align': 'loss_align was removed from runtime semantics.',
    '--use_loss_weight_ret': 'loss_weight_ret was removed from runtime semantics.',
    '--lambda_weight_ret': 'loss_weight_ret was removed from runtime semantics.',
    '--weight_ret_margin_delta': 'loss_weight_ret was removed from runtime semantics.',
    '--weight_ret_tau': 'loss_weight_ret was removed from runtime semantics.',
    '--weight_ret_detach_host': 'loss_weight_ret was removed from runtime semantics.',
    '--weight_ret_normalize_mean_one': 'loss_weight_ret was removed from runtime semantics.',
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


def _parse_float_list_csv(value):
    if value in (None, ''):
        return []
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    tokens = [token.strip() for token in str(value).split(',') if token.strip()]
    return [float(token) for token in tokens]


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
    parser.add_argument('--runtime_mode', '--model.runtime_mode', dest='runtime_mode', type=str, default='auto')
    parser.add_argument('--lr_ablation_enabled', '--lr_ablation.enabled', dest='lr_ablation_enabled', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--lr_ablation_base_lrs', '--lr_ablation.base_lrs', dest='lr_ablation_base_lrs', default='')
    parser.add_argument('--lr_ablation_num_epochs', '--lr_ablation.num_epochs', dest='lr_ablation_num_epochs', type=int, default=2)
    parser.add_argument('--lr_ablation_selection_metric', '--lr_ablation.selection_metric', dest='lr_ablation_selection_metric', type=str, default='val_r1')
    parser.add_argument('--lr_ablation_selection_task', '--lr_ablation.selection_task', dest='lr_ablation_selection_task', type=str, default='host-t2i')
    parser.add_argument('--lr_ablation_save_each_run', '--lr_ablation.save_each_run', dest='lr_ablation_save_each_run', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument(
        '--lr_ablation_restore_initial_state_each_run',
        '--lr_ablation.restore_initial_state_each_run',
        dest='lr_ablation_restore_initial_state_each_run',
        type=_str2bool,
        nargs='?',
        const=True,
        default=True,
    )
    parser.add_argument('--lr_ablation_write_summary_json', '--lr_ablation.write_summary_json', dest='lr_ablation_write_summary_json', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--lr_ablation_summary_path', '--lr_ablation.summary_path', dest='lr_ablation_summary_path', type=str, default='outputs/lr_ablation_summary.json')
    parser.add_argument('--prototype_method_role', type=str, default='semantic_structure')
    parser.add_argument('--prototype_semantic_enabled', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--prototype_recompute_enabled', type=_str2bool, nargs='?', const=True, default=None)
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
    parser.add_argument('--use_loss_diag', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--lambda_diag', type=float, default=1.0)
    parser.add_argument('--diag_temperature', type=float, default=0.07)
    parser.add_argument('--retrieval_mode', type=str, default='surrogate_i2t')
    parser.add_argument('--use_loss_semantic_pbt', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--lambda_semantic_pbt', type=float, default=0.0)
    parser.add_argument('--use_loss_semantic_hardneg_margin', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--lambda_semantic_hardneg_margin', type=float, default=0.0)
    parser.add_argument('--semantic_hardneg_margin', type=float, default=0.05)
    parser.add_argument('--semantic_hardneg_eps', type=float, default=1e-8)
    parser.add_argument('--use_loss_semantic_hosthard_weighted', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--lambda_semantic_hosthard_weighted', type=float, default=0.0)
    parser.add_argument('--semantic_hosthard_margin_ref', type=float, default=0.0)
    parser.add_argument('--semantic_hosthard_tau', type=float, default=0.1)
    parser.add_argument('--semantic_hosthard_eps', type=float, default=1e-8)
    parser.add_argument('--semantic_hosthard_normalize_weights', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--img_size', type=int, nargs=2, default=(384, 128))
    parser.add_argument('--stride_size', type=int, default=16)
    parser.add_argument('--text_length', type=int, default=77)
    parser.add_argument('--vocab_size', type=int, default=49408)
    parser.add_argument('--use_prototype_branch', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--use_prototype_bank', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_image_conditioned_pooling', type=_str2bool, nargs='?', const=True, default=True)
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
    parser.add_argument('--prototype_routing_source', type=str, default='global')
    parser.add_argument('--prototype_local_routing_temperature', type=float, default=None)
    parser.add_argument('--prototype_local_routing_pooling', type=str, default='logsumexp')
    parser.add_argument('--prototype_local_routing_use_adapter', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--prototype_local_routing_adapter_dim', type=int, default=None)
    parser.add_argument('--prototype_local_routing_normalize_inputs', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--prototype_contextualization_enabled', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--prototype_contextualization_type', type=str, default='self_attention')
    parser.add_argument('--prototype_contextualization_residual', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--prototype_bank_source', type=str, default='learnable_legacy')
    parser.add_argument('--prototype_use_base_for_semantic_targets', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--normalize_for_self_interaction', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--normalize_for_routing', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--semantic_structure_enabled', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--semantic_feature_space', type=str, default='prototype_projected')
    parser.add_argument('--semantic_pbt_enabled', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--semantic_soft_target_enabled', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--semantic_target_temperature', type=float, default=0.01)
    parser.add_argument('--semantic_pred_temperature', type=float, default=0.07)
    parser.add_argument('--semantic_recompute_schedule', type=str, default='epoch')
    parser.add_argument('--semantic_recompute_interval', type=int, default=1)
    parser.add_argument('--semantic_min_cluster_count_for_pbt', type=float, default=1.0)
    parser.add_argument('--semantic_empty_cluster_policy', type=str, default='skip')
    parser.add_argument('--semantic_text_teacher_source', type=str, default='exact_diagonal')
    parser.add_argument('--semantic_text_student_source', type=str, default='surrogate_diagonal')
    parser.add_argument('--semantic_image_student_source', type=str, default='image_semantic_feature')
    parser.add_argument('--semantic_recompute_start_epoch', type=int, default=0)
    parser.add_argument('--semantic_recompute_start_step', type=int, default=0)
    parser.add_argument('--semantic_loss_ramp_start_epoch', type=int, default=0)
    parser.add_argument('--semantic_loss_ramp_start_step', type=int, default=0)
    parser.add_argument('--semantic_loss_ramp_epochs', type=int, default=0)
    parser.add_argument('--semantic_loss_ramp_steps', type=int, default=0)
    parser.add_argument('--semantic_ramp_loss_diag', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--semantic_ramp_loss_semantic_pbt', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--semantic_ramp_loss_semantic_hardneg_margin', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--semantic_ramp_loss_semantic_hosthard_weighted', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--semantic_ramp_use_prototype', type=_str2bool, nargs='?', const=True, default=False)
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
    parser.add_argument('--early_stopping_enabled', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--early_stopping_metric', type=str, default='R1')
    parser.add_argument('--early_stopping_mode', type=str, default='max')
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0)
    parser.add_argument('--early_stopping_start_epoch', type=int, default=1)
    parser.add_argument('--early_stopping_monitored_bucket', type=str, default='host')
    parser.add_argument('--early_stopping_monitored_task_pattern', type=str, default=None)
    parser.add_argument('--early_stopping_stop_on_nan', type=_str2bool, nargs='?', const=True, default=False)
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
    parser.add_argument('--resume_strict', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--resume_restore_rng', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--finetune', type=str, default='')
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--img_aug', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--txt_aug', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--sampler', default='identity', help='choose sampler from [identity, random]')
    parser.add_argument('--num_instance', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--freeze_host_backbone', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--freeze_host_retrieval', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--freeze_fusion', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--freeze_prototype_bank', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--freeze_prototype_projector', type=_str2bool, nargs='?', const=True, default=None)
    parser.add_argument('--freeze_routing', type=_str2bool, nargs='?', const=True, default=None)
    # Deprecated coarse freeze aliases; retained as compatibility fallbacks.
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
    parser.add_argument('--itself_lambda_ablation_enabled', type=_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--itself_lambda_ablation_alphas', type=float, nargs='+', default=None)
    parser.add_argument('--itself_lambda_ablation_include_default', type=_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--prototype_eval_image_chunk_size', type=int, default=32)
    parser.add_argument('--prototype_eval_text_chunk_size', type=int, default=128)

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


def _coerce_optional_bool(value):
    if value is None:
        return None
    return bool(value)


def _resolve_freeze_controls(args):
    deprecated_controls = []

    legacy_freeze_host_projectors = bool(getattr(args, 'freeze_host_projectors', False))
    legacy_freeze_prototype_side = bool(getattr(args, 'freeze_prototype_side', False))

    freeze_host_backbone = _coerce_optional_bool(getattr(args, 'freeze_host_backbone', None))
    freeze_host_retrieval = _coerce_optional_bool(getattr(args, 'freeze_host_retrieval', None))
    freeze_fusion = _coerce_optional_bool(getattr(args, 'freeze_fusion', None))
    freeze_prototype_bank = _coerce_optional_bool(getattr(args, 'freeze_prototype_bank', None))
    freeze_prototype_projector = _coerce_optional_bool(getattr(args, 'freeze_prototype_projector', None))
    freeze_routing = _coerce_optional_bool(getattr(args, 'freeze_routing', None))

    if freeze_host_backbone is None:
        freeze_host_backbone = bool(getattr(args, 'freeze_image_backbone', True) and getattr(args, 'freeze_text_backbone', True))
    else:
        args.freeze_image_backbone = bool(freeze_host_backbone)
        args.freeze_text_backbone = bool(freeze_host_backbone)

    if freeze_host_retrieval is None:
        freeze_host_retrieval = legacy_freeze_host_projectors
        if legacy_freeze_host_projectors:
            deprecated_controls.append('training.freeze_host_projectors')

    if freeze_fusion is None:
        freeze_fusion = legacy_freeze_prototype_side
        if legacy_freeze_prototype_side:
            deprecated_controls.append('training.freeze_prototype_side')
    if freeze_prototype_bank is None:
        freeze_prototype_bank = legacy_freeze_prototype_side
        if legacy_freeze_prototype_side:
            deprecated_controls.append('training.freeze_prototype_side')
    if freeze_prototype_projector is None:
        freeze_prototype_projector = legacy_freeze_prototype_side
        if legacy_freeze_prototype_side:
            deprecated_controls.append('training.freeze_prototype_side')
    if freeze_routing is None:
        freeze_routing = legacy_freeze_prototype_side
        if legacy_freeze_prototype_side:
            deprecated_controls.append('training.freeze_prototype_side')

    args.freeze_host_backbone = bool(freeze_host_backbone)
    args.freeze_host_retrieval = bool(freeze_host_retrieval)
    args.freeze_fusion = bool(freeze_fusion)
    args.freeze_prototype_bank = bool(freeze_prototype_bank)
    args.freeze_prototype_projector = bool(freeze_prototype_projector)
    args.freeze_routing = bool(freeze_routing)

    # Keep legacy attributes synchronized as backward-compatible aliases only.
    args.freeze_host_projectors = bool(args.freeze_host_retrieval)
    args.freeze_prototype_side = bool(
        args.freeze_fusion
        and args.freeze_prototype_bank
        and args.freeze_prototype_projector
        and args.freeze_routing
    )
    args.freeze_image_backbone = bool(getattr(args, 'freeze_image_backbone', True))
    args.freeze_text_backbone = bool(getattr(args, 'freeze_text_backbone', True))
    args.freeze_backbones = bool(args.freeze_image_backbone and args.freeze_text_backbone)
    args.deprecated_freeze_controls = sorted(set(deprecated_controls))
    return args


def _finalize_args(args):
    if isinstance(args.img_size, list):
        args.img_size = tuple(args.img_size)
    if isinstance(args.milestones, tuple):
        args.milestones = list(args.milestones)
    if not args.prototype_init_path:
        args.prototype_init_path = None
    if not args.retrieval_metrics:
        args.retrieval_metrics = list(DEFAULT_RETRIEVAL_METRICS)
    args.itself_lambda_ablation_enabled = bool(getattr(args, 'itself_lambda_ablation_enabled', False))
    args.itself_lambda_ablation_include_default = bool(
        getattr(args, 'itself_lambda_ablation_include_default', True)
    )
    ablation_alphas = getattr(args, 'itself_lambda_ablation_alphas', None)
    if ablation_alphas is None:
        args.itself_lambda_ablation_alphas = []
    else:
        normalized_alphas = []
        seen_alphas = set()
        for raw_alpha in list(ablation_alphas):
            alpha = float(raw_alpha)
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError(
                    f'itself_lambda_ablation_alphas must be within [0, 1]. Got {alpha}.'
                )
            rounded = round(alpha, 6)
            if rounded in seen_alphas:
                continue
            seen_alphas.add(rounded)
            normalized_alphas.append(alpha)
        args.itself_lambda_ablation_alphas = normalized_alphas
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
    args.runtime_mode = str(getattr(args, 'runtime_mode', 'auto')).lower()
    args.lr_ablation_enabled = bool(getattr(args, 'lr_ablation_enabled', False))
    try:
        args.lr_ablation_base_lrs = _parse_float_list_csv(getattr(args, 'lr_ablation_base_lrs', ''))
    except (TypeError, ValueError) as exc:
        raise ValueError('lr_ablation.base_lrs must be a comma-separated list of floats.') from exc
    args.lr_ablation_num_epochs = max(int(getattr(args, 'lr_ablation_num_epochs', 2)), 1)
    args.lr_ablation_selection_metric = str(getattr(args, 'lr_ablation_selection_metric', 'val_r1') or 'val_r1').strip().lower()
    args.lr_ablation_selection_task = str(getattr(args, 'lr_ablation_selection_task', 'host-t2i') or 'host-t2i').strip()
    args.lr_ablation_save_each_run = bool(getattr(args, 'lr_ablation_save_each_run', True))
    args.lr_ablation_restore_initial_state_each_run = bool(
        getattr(args, 'lr_ablation_restore_initial_state_each_run', True)
    )
    args.lr_ablation_write_summary_json = bool(getattr(args, 'lr_ablation_write_summary_json', True))
    args.lr_ablation_summary_path = str(
        getattr(args, 'lr_ablation_summary_path', 'outputs/lr_ablation_summary.json') or ''
    ).strip() or 'outputs/lr_ablation_summary.json'
    if args.runtime_mode == 'lr_ablation':
        args.lr_ablation_enabled = True
    args.training_stage = str(getattr(args, 'training_stage', 'joint')).lower()
    args.resume = bool(getattr(args, 'resume', False))
    args.resume_ckpt_file = str(getattr(args, 'resume_ckpt_file', '') or '').strip()
    args.resume_strict = bool(getattr(args, 'resume_strict', False))
    args.resume_restore_rng = bool(getattr(args, 'resume_restore_rng', True))
    args.early_stopping_enabled = bool(getattr(args, 'early_stopping_enabled', False))
    args.early_stopping_metric = str(getattr(args, 'early_stopping_metric', 'R1') or 'R1')
    args.early_stopping_mode = str(getattr(args, 'early_stopping_mode', 'max') or 'max').lower()
    args.early_stopping_patience = int(getattr(args, 'early_stopping_patience', 5))
    args.early_stopping_min_delta = float(getattr(args, 'early_stopping_min_delta', 0.0) or 0.0)
    args.early_stopping_start_epoch = int(getattr(args, 'early_stopping_start_epoch', 1))
    monitored_bucket = getattr(args, 'early_stopping_monitored_bucket', 'host')
    if monitored_bucket is None:
        args.early_stopping_monitored_bucket = None
    else:
        monitored_bucket = str(monitored_bucket).strip()
        args.early_stopping_monitored_bucket = monitored_bucket if monitored_bucket else None
    monitored_task_pattern = getattr(args, 'early_stopping_monitored_task_pattern', None)
    if monitored_task_pattern is None:
        args.early_stopping_monitored_task_pattern = None
    else:
        monitored_task_pattern = str(monitored_task_pattern).strip()
        args.early_stopping_monitored_task_pattern = monitored_task_pattern if monitored_task_pattern else None
    args.early_stopping_stop_on_nan = bool(getattr(args, 'early_stopping_stop_on_nan', False))
    args.prototype_method_role = str(getattr(args, 'prototype_method_role', 'semantic_structure')).lower()
    if args.prototype_method_role != 'semantic_structure':
        raise ValueError(
            'model.prototype_method_role=retrieval_branch is removed. '
            'PrototypePlugin is structure-only and retrieval is host-only.'
        )
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

    def _has_override_path(section_name, key_name):
        section = override_config_data.get(section_name, {})
        return isinstance(section, dict) and key_name in section

    def _has_nested_override_path(section_name, nested_name, key_name):
        section = override_config_data.get(section_name, {})
        if not isinstance(section, dict):
            return False
        nested = section.get(nested_name, {})
        return isinstance(nested, dict) and key_name in nested

    def _is_explicit_bool(dest_name, section_name, key_name, default_value):
        if dest_name in cli_dests:
            return bool(getattr(args, dest_name))
        if _has_override_path(section_name, key_name):
            return bool(getattr(args, dest_name))
        value = getattr(args, dest_name)
        if value is None:
            return bool(default_value)
        return bool(value)

    semantic_mode_selected = args.prototype_method_role == 'semantic_structure'
    args.prototype_semantic_enabled = _is_explicit_bool(
        'prototype_semantic_enabled',
        'model',
        'prototype_semantic_enabled',
        default_value=semantic_mode_selected,
    )
    args.semantic_structure_enabled = _is_explicit_bool(
        'semantic_structure_enabled',
        'semantic_structure',
        'enabled',
        default_value=args.prototype_semantic_enabled,
    )
    args.prototype_recompute_enabled = _is_explicit_bool(
        'prototype_recompute_enabled',
        'model',
        'prototype_recompute_enabled',
        default_value=(semantic_mode_selected and args.semantic_structure_enabled),
    )
    args.prototype_bank_source = str(getattr(args, 'prototype_bank_source', 'learnable_legacy')).lower()
    if args.prototype_bank_source in {'', 'auto'}:
        args.prototype_bank_source = 'recomputed_kmeans' if semantic_mode_selected else 'learnable_legacy'
    args.prototype_use_base_for_semantic_targets = bool(
        getattr(args, 'prototype_use_base_for_semantic_targets', True)
    )
    args.semantic_feature_space = str(getattr(args, 'semantic_feature_space', 'prototype_projected')).lower()
    args.semantic_pbt_enabled = bool(getattr(args, 'semantic_pbt_enabled', True))
    args.semantic_soft_target_enabled = bool(getattr(args, 'semantic_soft_target_enabled', True))
    args.semantic_target_temperature = float(getattr(args, 'semantic_target_temperature', 0.01))
    args.semantic_pred_temperature = float(getattr(args, 'semantic_pred_temperature', 0.07))
    args.semantic_recompute_schedule = str(getattr(args, 'semantic_recompute_schedule', 'epoch')).lower()
    args.semantic_recompute_interval = int(getattr(args, 'semantic_recompute_interval', 1))
    args.semantic_min_cluster_count_for_pbt = float(getattr(args, 'semantic_min_cluster_count_for_pbt', 1.0))
    args.semantic_empty_cluster_policy = str(getattr(args, 'semantic_empty_cluster_policy', 'skip')).lower()
    args.semantic_text_teacher_source = str(getattr(args, 'semantic_text_teacher_source', 'exact_diagonal')).lower()
    args.semantic_text_student_source = str(getattr(args, 'semantic_text_student_source', 'surrogate_diagonal')).lower()
    args.semantic_image_student_source = str(getattr(args, 'semantic_image_student_source', 'image_semantic_feature')).lower()
    args.semantic_recompute_start_epoch = max(int(getattr(args, 'semantic_recompute_start_epoch', 0)), 0)
    args.semantic_recompute_start_step = max(int(getattr(args, 'semantic_recompute_start_step', 0)), 0)
    args.semantic_loss_ramp_start_epoch = max(int(getattr(args, 'semantic_loss_ramp_start_epoch', 0)), 0)
    args.semantic_loss_ramp_start_step = max(int(getattr(args, 'semantic_loss_ramp_start_step', 0)), 0)
    args.semantic_loss_ramp_epochs = max(int(getattr(args, 'semantic_loss_ramp_epochs', 0)), 0)
    args.semantic_loss_ramp_steps = max(int(getattr(args, 'semantic_loss_ramp_steps', 0)), 0)
    args.semantic_ramp_loss_diag = bool(getattr(args, 'semantic_ramp_loss_diag', False))
    args.semantic_ramp_loss_semantic_pbt = bool(getattr(args, 'semantic_ramp_loss_semantic_pbt', True))
    args.semantic_ramp_loss_semantic_hardneg_margin = bool(
        getattr(args, 'semantic_ramp_loss_semantic_hardneg_margin', True)
    )
    args.semantic_ramp_loss_semantic_hosthard_weighted = bool(
        getattr(args, 'semantic_ramp_loss_semantic_hosthard_weighted', True)
    )
    args.semantic_ramp_use_prototype = bool(getattr(args, 'semantic_ramp_use_prototype', False))
    args.prototype_inference_mode = 'host_only'

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
    args = _resolve_freeze_controls(args)
    args.diag_temperature = float(getattr(args, 'diag_temperature', 0.07))
    args.use_loss_diag = bool(getattr(args, 'use_loss_diag', True))
    args.lambda_diag = float(getattr(args, 'lambda_diag', 1.0))
    args.use_loss_semantic_pbt = bool(getattr(args, 'use_loss_semantic_pbt', False))
    args.lambda_semantic_pbt = float(getattr(args, 'lambda_semantic_pbt', 0.0))
    args.use_loss_semantic_hardneg_margin = bool(getattr(args, 'use_loss_semantic_hardneg_margin', False))
    args.lambda_semantic_hardneg_margin = float(getattr(args, 'lambda_semantic_hardneg_margin', 0.0))
    args.semantic_hardneg_margin = float(getattr(args, 'semantic_hardneg_margin', 0.05))
    args.semantic_hardneg_eps = float(getattr(args, 'semantic_hardneg_eps', 1e-8))
    args.use_loss_semantic_hosthard_weighted = bool(getattr(args, 'use_loss_semantic_hosthard_weighted', False))
    args.lambda_semantic_hosthard_weighted = float(getattr(args, 'lambda_semantic_hosthard_weighted', 0.0))
    args.semantic_hosthard_margin_ref = float(getattr(args, 'semantic_hosthard_margin_ref', 0.0))
    args.semantic_hosthard_tau = float(getattr(args, 'semantic_hosthard_tau', 0.1))
    args.semantic_hosthard_eps = float(getattr(args, 'semantic_hosthard_eps', 1e-8))
    args.semantic_hosthard_normalize_weights = bool(getattr(args, 'semantic_hosthard_normalize_weights', True))
    semantic_loss_explicit = (
        ('use_loss_semantic_pbt' in cli_dests)
        or _has_override_path('loss', 'use_loss_semantic_pbt')
        or _has_nested_override_path('objectives', 'objectives', 'use_loss_semantic_pbt')
    )
    semantic_lambda_explicit = (
        ('lambda_semantic_pbt' in cli_dests)
        or _has_override_path('loss', 'lambda_semantic_pbt')
        or _has_nested_override_path('objectives', 'lambda', 'semantic_pbt')
    )
    if semantic_mode_selected and not semantic_loss_explicit:
        args.use_loss_semantic_pbt = bool(args.semantic_structure_enabled and args.semantic_pbt_enabled)
    if semantic_mode_selected and not semantic_lambda_explicit:
        args.lambda_semantic_pbt = 1.0 if args.use_loss_semantic_pbt else 0.0
    if not args.use_loss_semantic_pbt:
        args.lambda_semantic_pbt = 0.0
    if not args.use_loss_semantic_hardneg_margin:
        args.lambda_semantic_hardneg_margin = 0.0
    if not args.use_loss_semantic_hosthard_weighted:
        args.lambda_semantic_hosthard_weighted = 0.0

    args.prototype_routing_source = str(getattr(args, 'prototype_routing_source', 'global')).lower()
    args.prototype_local_routing_temperature = getattr(args, 'prototype_local_routing_temperature', None)
    if args.prototype_local_routing_temperature in ('', None):
        args.prototype_local_routing_temperature = None
    else:
        args.prototype_local_routing_temperature = float(args.prototype_local_routing_temperature)
    args.prototype_local_routing_pooling = str(getattr(args, 'prototype_local_routing_pooling', 'logsumexp')).lower()
    args.prototype_local_routing_use_adapter = bool(getattr(args, 'prototype_local_routing_use_adapter', True))
    local_adapter_dim = getattr(args, 'prototype_local_routing_adapter_dim', None)
    if local_adapter_dim in ('', None, 0):
        args.prototype_local_routing_adapter_dim = None
    else:
        args.prototype_local_routing_adapter_dim = int(local_adapter_dim)
    args.prototype_local_routing_normalize_inputs = bool(getattr(args, 'prototype_local_routing_normalize_inputs', True))
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

