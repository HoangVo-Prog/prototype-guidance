from typing import Optional

import torch
import torch.nn as nn

from .direct_head import DirectImageConditionedTextHead
from .head import PrototypeConditionedTextHead


def build_prototype_head(
    args,
    input_dim: int,
    num_classes: int,
    image_adapter: Optional[nn.Module] = None,
    text_adapter: Optional[nn.Module] = None,
    prototype_init_features: Optional[torch.Tensor] = None,
):
    contextualization_enabled = bool(getattr(args, 'prototype_contextualization_enabled', True))
    contextualization_type = getattr(args, 'prototype_contextualization_type', 'none') if contextualization_enabled else 'none'
    contextualization_residual = getattr(args, 'prototype_contextualization_residual', True) if contextualization_enabled else False
    projector_output_dim = getattr(args, 'projector_output_dim', getattr(args, 'projection_dim', 256))
    token_scoring_type = getattr(args, 'token_similarity', getattr(args, 'token_scoring_type', 'cosine'))
    token_temperature = getattr(args, 'tau_t', getattr(args, 'token_pooling_temperature', 0.07))
    use_image_conditioned_pooling = bool(getattr(args, 'use_image_conditioned_pooling', True))
    routing_source = str(getattr(args, 'prototype_routing_source', getattr(args, 'routing_source', 'global'))).lower()
    local_routing_temperature = getattr(args, 'prototype_local_routing_temperature', getattr(args, 'local_routing_temperature', None))
    if local_routing_temperature in ('', None):
        local_routing_temperature = None
    else:
        local_routing_temperature = float(local_routing_temperature)
    local_routing_pooling = str(getattr(args, 'prototype_local_routing_pooling', getattr(args, 'local_routing_pooling', 'logsumexp'))).lower()
    local_routing_use_adapter = bool(getattr(args, 'prototype_local_routing_use_adapter', getattr(args, 'local_routing_use_adapter', True)))
    local_routing_adapter_dim = getattr(args, 'prototype_local_routing_adapter_dim', getattr(args, 'local_routing_adapter_dim', None))
    if local_routing_adapter_dim in ('', None, 0):
        local_routing_adapter_dim = None
    else:
        local_routing_adapter_dim = int(local_routing_adapter_dim)
    local_routing_normalize_inputs = bool(
        getattr(args, 'prototype_local_routing_normalize_inputs', getattr(args, 'local_routing_normalize_inputs', True))
    )
    prototype_method_role = str(getattr(args, 'prototype_method_role', 'retrieval_branch')).lower()
    prototype_semantic_enabled = bool(getattr(args, 'prototype_semantic_enabled', prototype_method_role == 'semantic_structure'))
    semantic_structure_enabled = bool(getattr(args, 'semantic_structure_enabled', prototype_semantic_enabled))
    prototype_recompute_enabled = bool(
        getattr(args, 'prototype_recompute_enabled', prototype_semantic_enabled and prototype_method_role == 'semantic_structure')
    )
    prototype_bank_source = str(getattr(args, 'prototype_bank_source', 'learnable_legacy')).lower()
    prototype_use_base_for_semantic_targets = bool(getattr(args, 'prototype_use_base_for_semantic_targets', True))
    semantic_feature_space = str(getattr(args, 'semantic_feature_space', 'prototype_projected')).lower()
    semantic_pbt_enabled = bool(getattr(args, 'semantic_pbt_enabled', True))
    semantic_soft_target_enabled = bool(getattr(args, 'semantic_soft_target_enabled', True))
    semantic_target_temperature = float(getattr(args, 'semantic_target_temperature', 0.01))
    semantic_pred_temperature = float(getattr(args, 'semantic_pred_temperature', 0.07))
    semantic_recompute_schedule = str(getattr(args, 'semantic_recompute_schedule', 'epoch')).lower()
    semantic_recompute_interval = int(getattr(args, 'semantic_recompute_interval', 1))
    semantic_min_cluster_count_for_pbt = float(getattr(args, 'semantic_min_cluster_count_for_pbt', 1.0))
    semantic_empty_cluster_policy = str(getattr(args, 'semantic_empty_cluster_policy', 'skip')).lower()
    semantic_text_teacher_source = str(getattr(args, 'semantic_text_teacher_source', 'exact_diagonal')).lower()
    semantic_text_student_source = str(getattr(args, 'semantic_text_student_source', 'surrogate_diagonal')).lower()
    semantic_image_student_source = str(getattr(args, 'semantic_image_student_source', 'image_semantic_feature')).lower()
    semantic_recompute_start_epoch = max(int(getattr(args, 'semantic_recompute_start_epoch', 0)), 0)
    semantic_recompute_start_step = max(int(getattr(args, 'semantic_recompute_start_step', 0)), 0)
    semantic_loss_ramp_start_epoch = max(int(getattr(args, 'semantic_loss_ramp_start_epoch', 0)), 0)
    semantic_loss_ramp_start_step = max(int(getattr(args, 'semantic_loss_ramp_start_step', 0)), 0)
    semantic_loss_ramp_epochs = max(int(getattr(args, 'semantic_loss_ramp_epochs', 0)), 0)
    semantic_loss_ramp_steps = max(int(getattr(args, 'semantic_loss_ramp_steps', 0)), 0)
    semantic_ramp_loss_diag = bool(getattr(args, 'semantic_ramp_loss_diag', False))
    semantic_ramp_loss_semantic_pbt = bool(getattr(args, 'semantic_ramp_loss_semantic_pbt', True))
    semantic_ramp_loss_semantic_hardneg_margin = bool(
        getattr(args, 'semantic_ramp_loss_semantic_hardneg_margin', True)
    )
    semantic_ramp_loss_semantic_hosthard_weighted = bool(
        getattr(args, 'semantic_ramp_loss_semantic_hosthard_weighted', True)
    )
    semantic_ramp_use_prototype = bool(getattr(args, 'semantic_ramp_use_prototype', False))
    use_loss_semantic_pbt = bool(getattr(args, 'use_loss_semantic_pbt', False))
    lambda_semantic_pbt = float(getattr(args, 'lambda_semantic_pbt', 0.0))
    use_loss_semantic_hardneg_margin = bool(getattr(args, 'use_loss_semantic_hardneg_margin', False))
    lambda_semantic_hardneg_margin = float(getattr(args, 'lambda_semantic_hardneg_margin', 0.0))
    semantic_hardneg_margin = float(getattr(args, 'semantic_hardneg_margin', 0.05))
    semantic_hardneg_eps = float(getattr(args, 'semantic_hardneg_eps', 1e-8))
    use_loss_semantic_hosthard_weighted = bool(getattr(args, 'use_loss_semantic_hosthard_weighted', False))
    lambda_semantic_hosthard_weighted = float(getattr(args, 'lambda_semantic_hosthard_weighted', 0.0))
    semantic_hosthard_margin_ref = float(getattr(args, 'semantic_hosthard_margin_ref', 0.0))
    semantic_hosthard_tau = float(getattr(args, 'semantic_hosthard_tau', 0.1))
    semantic_hosthard_eps = float(getattr(args, 'semantic_hosthard_eps', 1e-8))
    semantic_hosthard_normalize_weights = bool(getattr(args, 'semantic_hosthard_normalize_weights', True))
    use_loss_hbr = bool(getattr(args, 'use_hbr', False))
    lambda_hbr = float(getattr(args, 'lambda_hbr', 0.0))
    hbr_topk_hard_negatives = int(getattr(args, 'hbr_topk_hard_negatives', 5))
    hbr_base_margin = float(getattr(args, 'hbr_base_margin', 0.1))
    hbr_host_gate_margin = float(getattr(args, 'hbr_host_gate_margin', 0.1))
    hbr_host_gate_temperature = float(getattr(args, 'hbr_host_gate_temperature', 0.1))
    hbr_use_global_local_decomposition = bool(getattr(args, 'hbr_use_global_local_decomposition', True))
    hbr_global_gate_margin = float(getattr(args, 'hbr_global_gate_margin', 0.1))
    hbr_local_gate_margin = float(getattr(args, 'hbr_local_gate_margin', 0.1))
    hbr_global_gate_weight = float(getattr(args, 'hbr_global_gate_weight', 1.0))
    hbr_local_gate_weight = float(getattr(args, 'hbr_local_gate_weight', 1.0))
    hbr_use_prototype_pair_signal = bool(getattr(args, 'hbr_use_prototype_pair_signal', True))
    hbr_proto_signal_temperature = float(getattr(args, 'hbr_proto_signal_temperature', 0.1))
    hbr_proto_signal_center = float(getattr(args, 'hbr_proto_signal_center', 0.0))
    hbr_stopgrad_proto_signal = bool(getattr(args, 'hbr_stopgrad_proto_signal', True))
    hbr_control_mode = str(getattr(args, 'hbr_control_mode', 'none')).lower()
    hbr_proto_adaptive_margin_weight = float(getattr(args, 'hbr_proto_adaptive_margin_weight', 0.0))
    hbr_tail_selection_mode = str(getattr(args, 'hbr_tail_selection_mode', 'bottomk')).lower()
    hbr_tail_bottomk = int(getattr(args, 'hbr_tail_bottomk', hbr_topk_hard_negatives))
    hbr_tail_margin_threshold = getattr(args, 'hbr_tail_margin_threshold', None)
    if hbr_tail_margin_threshold in ('', None):
        hbr_tail_margin_threshold = None
    else:
        hbr_tail_margin_threshold = float(hbr_tail_margin_threshold)
    hbr_inner_tail_weight_mode = str(getattr(args, 'hbr_inner_tail_weight_mode', 'uniform')).lower()
    hbr_proto_inner_tau = float(getattr(args, 'hbr_proto_inner_tau', hbr_proto_signal_temperature))
    hbr_proto_inner_center = float(getattr(args, 'hbr_proto_inner_center', hbr_proto_signal_center))
    hbr_adaptive_margin_enabled = bool(getattr(args, 'hbr_adaptive_margin_enabled', False))
    hbr_adaptive_margin_lambda = float(getattr(args, 'hbr_adaptive_margin_lambda', 0.0))

    common_kwargs = dict(
        input_dim=input_dim,
        prototype_dim=getattr(args, 'prototype_dim', input_dim),
        projector_output_dim=projector_output_dim,
        projector_hidden_dim=getattr(args, 'projector_hidden_dim', getattr(args, 'prototype_dim', input_dim)),
        projector_dropout=getattr(args, 'projector_dropout', 0.0),
        projector_type=getattr(args, 'projector_type', 'mlp2'),
        image_adapter=image_adapter,
        text_adapter=text_adapter,
        token_scoring_type=token_scoring_type,
        token_temperature=token_temperature,
        token_policy=getattr(args, 'token_policy', 'content_only'),
        special_token_ids=getattr(args, 'special_token_ids', None),
        error_on_empty_kept_tokens=getattr(args, 'error_on_empty_kept_tokens', True),
        normalize_for_token_scoring=getattr(args, 'normalize_for_token_scoring', True),
        normalize_projector_outputs=getattr(args, 'normalize_projector_outputs', True),
        use_image_conditioned_pooling=use_image_conditioned_pooling,
        num_classes=num_classes,
        proxy_temperature=getattr(args, 'proxy_temperature', 0.07),
        use_loss_diag=getattr(args, 'use_loss_diag', True),
        lambda_diag=getattr(args, 'lambda_diag', 1.0),
        diag_temperature=getattr(args, 'diag_temperature', 0.07),
        use_loss_semantic_pbt=use_loss_semantic_pbt,
        lambda_semantic_pbt=lambda_semantic_pbt,
        use_loss_semantic_hardneg_margin=use_loss_semantic_hardneg_margin,
        lambda_semantic_hardneg_margin=lambda_semantic_hardneg_margin,
        semantic_hardneg_margin=semantic_hardneg_margin,
        semantic_hardneg_eps=semantic_hardneg_eps,
        use_loss_semantic_hosthard_weighted=use_loss_semantic_hosthard_weighted,
        lambda_semantic_hosthard_weighted=lambda_semantic_hosthard_weighted,
        semantic_hosthard_margin_ref=semantic_hosthard_margin_ref,
        semantic_hosthard_tau=semantic_hosthard_tau,
        semantic_hosthard_eps=semantic_hosthard_eps,
        semantic_hosthard_normalize_weights=semantic_hosthard_normalize_weights,
        use_loss_hbr=use_loss_hbr,
        lambda_hbr=lambda_hbr,
        hbr_topk_hard_negatives=hbr_topk_hard_negatives,
        hbr_base_margin=hbr_base_margin,
        hbr_host_gate_margin=hbr_host_gate_margin,
        hbr_host_gate_temperature=hbr_host_gate_temperature,
        hbr_use_global_local_decomposition=hbr_use_global_local_decomposition,
        hbr_global_gate_margin=hbr_global_gate_margin,
        hbr_local_gate_margin=hbr_local_gate_margin,
        hbr_global_gate_weight=hbr_global_gate_weight,
        hbr_local_gate_weight=hbr_local_gate_weight,
        hbr_use_prototype_pair_signal=hbr_use_prototype_pair_signal,
        hbr_proto_signal_temperature=hbr_proto_signal_temperature,
        hbr_proto_signal_center=hbr_proto_signal_center,
        hbr_stopgrad_proto_signal=hbr_stopgrad_proto_signal,
        hbr_control_mode=hbr_control_mode,
        hbr_proto_adaptive_margin_weight=hbr_proto_adaptive_margin_weight,
        hbr_tail_selection_mode=hbr_tail_selection_mode,
        hbr_tail_bottomk=hbr_tail_bottomk,
        hbr_tail_margin_threshold=hbr_tail_margin_threshold,
        hbr_inner_tail_weight_mode=hbr_inner_tail_weight_mode,
        hbr_proto_inner_tau=hbr_proto_inner_tau,
        hbr_proto_inner_center=hbr_proto_inner_center,
        hbr_adaptive_margin_enabled=hbr_adaptive_margin_enabled,
        hbr_adaptive_margin_lambda=hbr_adaptive_margin_lambda,
        prototype_method_role=prototype_method_role,
        prototype_semantic_enabled=prototype_semantic_enabled,
        semantic_structure_enabled=semantic_structure_enabled,
        semantic_feature_space=semantic_feature_space,
        semantic_pbt_enabled=semantic_pbt_enabled,
        semantic_soft_target_enabled=semantic_soft_target_enabled,
        semantic_target_temperature=semantic_target_temperature,
        semantic_pred_temperature=semantic_pred_temperature,
        semantic_min_cluster_count_for_pbt=semantic_min_cluster_count_for_pbt,
        semantic_empty_cluster_policy=semantic_empty_cluster_policy,
        contrastive_temperature_init=getattr(args, 'temperature', 0.07),
    )

    if not bool(getattr(args, 'use_prototype_bank', True)):
        return DirectImageConditionedTextHead(**common_kwargs)

    routing_type = getattr(args, 'routing_similarity', getattr(args, 'prototype_routing_type', 'cosine'))
    routing_temperature = getattr(args, 'tau_p', getattr(args, 'prototype_temperature', 0.07))
    diversity_loss_weight = getattr(args, 'lambda_div', getattr(args, 'diversity_loss_weight', 0.01))
    balance_loss_weight = getattr(args, 'lambda_bal', getattr(args, 'prototype_balance_loss_weight', 0.0))
    prototype_init_seed = getattr(args, 'prototype_init_seed', None)
    if prototype_init_seed is None:
        prototype_init_seed = getattr(args, 'seed', None)
    use_balancing_loss = bool(getattr(args, 'use_balancing_loss', False))
    if use_balancing_loss and balance_loss_weight <= 0.0:
        raise ValueError('use_balancing_loss=true requires lambda_bal / balance_loss_weight to be positive.')
    if not use_balancing_loss and balance_loss_weight != 0.0:
        raise ValueError('lambda_bal / balance_loss_weight must be 0.0 when use_balancing_loss is disabled.')
    return PrototypeConditionedTextHead(
        input_dim=input_dim,
        num_prototypes=getattr(args, 'prototype_num_prototypes', 32),
        prototype_dim=getattr(args, 'prototype_dim', input_dim),
        projector_output_dim=projector_output_dim,
        projector_hidden_dim=getattr(args, 'projector_hidden_dim', getattr(args, 'prototype_dim', input_dim)),
        projector_dropout=getattr(args, 'projector_dropout', 0.0),
        projector_type=getattr(args, 'projector_type', 'mlp2'),
        prototype_init=getattr(args, 'prototype_init', 'normalized_random'),
        prototype_init_path=getattr(args, 'prototype_init_path', None),
        prototype_init_hybrid_ratio=getattr(args, 'prototype_init_hybrid_ratio', 0.5),
        prototype_init_max_iters=getattr(args, 'prototype_init_max_iters', 50),
        prototype_init_tol=getattr(args, 'prototype_init_tol', 1e-4),
        prototype_init_seed=prototype_init_seed,
        prototype_init_features=prototype_init_features,
        image_adapter=image_adapter,
        text_adapter=text_adapter,
        routing_type=routing_type,
        routing_temperature=routing_temperature,
        routing_source=routing_source,
        local_routing_temperature=local_routing_temperature,
        local_routing_pooling=local_routing_pooling,
        local_routing_use_adapter=local_routing_use_adapter,
        local_routing_adapter_dim=local_routing_adapter_dim,
        local_routing_normalize_inputs=local_routing_normalize_inputs,
        prototype_method_role=prototype_method_role,
        prototype_semantic_enabled=prototype_semantic_enabled,
        prototype_recompute_enabled=prototype_recompute_enabled,
        prototype_bank_source=prototype_bank_source,
        prototype_use_base_for_semantic_targets=prototype_use_base_for_semantic_targets,
        semantic_structure_enabled=semantic_structure_enabled,
        semantic_feature_space=semantic_feature_space,
        semantic_pbt_enabled=semantic_pbt_enabled,
        semantic_soft_target_enabled=semantic_soft_target_enabled,
        semantic_target_temperature=semantic_target_temperature,
        semantic_pred_temperature=semantic_pred_temperature,
        semantic_recompute_schedule=semantic_recompute_schedule,
        semantic_recompute_interval=semantic_recompute_interval,
        semantic_min_cluster_count_for_pbt=semantic_min_cluster_count_for_pbt,
        semantic_empty_cluster_policy=semantic_empty_cluster_policy,
        semantic_text_teacher_source=semantic_text_teacher_source,
        semantic_text_student_source=semantic_text_student_source,
        semantic_image_student_source=semantic_image_student_source,
        semantic_recompute_start_epoch=semantic_recompute_start_epoch,
        semantic_recompute_start_step=semantic_recompute_start_step,
        semantic_loss_ramp_start_epoch=semantic_loss_ramp_start_epoch,
        semantic_loss_ramp_start_step=semantic_loss_ramp_start_step,
        semantic_loss_ramp_epochs=semantic_loss_ramp_epochs,
        semantic_loss_ramp_steps=semantic_loss_ramp_steps,
        semantic_ramp_loss_diag=semantic_ramp_loss_diag,
        semantic_ramp_loss_semantic_pbt=semantic_ramp_loss_semantic_pbt,
        semantic_ramp_loss_semantic_hardneg_margin=semantic_ramp_loss_semantic_hardneg_margin,
        semantic_ramp_loss_semantic_hosthard_weighted=semantic_ramp_loss_semantic_hosthard_weighted,
        semantic_ramp_use_prototype=semantic_ramp_use_prototype,
        token_scoring_type=token_scoring_type,
        token_temperature=token_temperature,
        token_policy=getattr(args, 'token_policy', 'content_only'),
        special_token_ids=getattr(args, 'special_token_ids', None),
        error_on_empty_kept_tokens=getattr(args, 'error_on_empty_kept_tokens', True),
        contextualization_enabled=contextualization_enabled,
        contextualization_type=contextualization_type,
        contextualization_residual=contextualization_residual,
        normalize_for_self_interaction=getattr(args, 'normalize_for_self_interaction', True),
        normalize_for_routing=getattr(args, 'normalize_for_routing', True),
        normalize_for_token_scoring=getattr(args, 'normalize_for_token_scoring', True),
        normalize_projector_outputs=getattr(args, 'normalize_projector_outputs', True),
        use_image_conditioned_pooling=use_image_conditioned_pooling,
        num_classes=num_classes,
        proxy_temperature=getattr(args, 'proxy_temperature', 0.07),
        use_loss_diag=getattr(args, 'use_loss_diag', True),
        lambda_diag=getattr(args, 'lambda_diag', 1.0),
        diag_temperature=getattr(args, 'diag_temperature', 0.07),
        use_loss_semantic_pbt=use_loss_semantic_pbt,
        lambda_semantic_pbt=lambda_semantic_pbt,
        use_loss_semantic_hardneg_margin=use_loss_semantic_hardneg_margin,
        lambda_semantic_hardneg_margin=lambda_semantic_hardneg_margin,
        semantic_hardneg_margin=semantic_hardneg_margin,
        semantic_hardneg_eps=semantic_hardneg_eps,
        use_loss_semantic_hosthard_weighted=use_loss_semantic_hosthard_weighted,
        lambda_semantic_hosthard_weighted=lambda_semantic_hosthard_weighted,
        semantic_hosthard_margin_ref=semantic_hosthard_margin_ref,
        semantic_hosthard_tau=semantic_hosthard_tau,
        semantic_hosthard_eps=semantic_hosthard_eps,
        semantic_hosthard_normalize_weights=semantic_hosthard_normalize_weights,
        use_loss_hbr=use_loss_hbr,
        lambda_hbr=lambda_hbr,
        hbr_topk_hard_negatives=hbr_topk_hard_negatives,
        hbr_base_margin=hbr_base_margin,
        hbr_host_gate_margin=hbr_host_gate_margin,
        hbr_host_gate_temperature=hbr_host_gate_temperature,
        hbr_use_global_local_decomposition=hbr_use_global_local_decomposition,
        hbr_global_gate_margin=hbr_global_gate_margin,
        hbr_local_gate_margin=hbr_local_gate_margin,
        hbr_global_gate_weight=hbr_global_gate_weight,
        hbr_local_gate_weight=hbr_local_gate_weight,
        hbr_use_prototype_pair_signal=hbr_use_prototype_pair_signal,
        hbr_proto_signal_temperature=hbr_proto_signal_temperature,
        hbr_proto_signal_center=hbr_proto_signal_center,
        hbr_stopgrad_proto_signal=hbr_stopgrad_proto_signal,
        hbr_control_mode=hbr_control_mode,
        hbr_proto_adaptive_margin_weight=hbr_proto_adaptive_margin_weight,
        hbr_tail_selection_mode=hbr_tail_selection_mode,
        hbr_tail_bottomk=hbr_tail_bottomk,
        hbr_tail_margin_threshold=hbr_tail_margin_threshold,
        hbr_inner_tail_weight_mode=hbr_inner_tail_weight_mode,
        hbr_proto_inner_tau=hbr_proto_inner_tau,
        hbr_proto_inner_center=hbr_proto_inner_center,
        hbr_adaptive_margin_enabled=hbr_adaptive_margin_enabled,
        hbr_adaptive_margin_lambda=hbr_adaptive_margin_lambda,
        use_diversity_loss=getattr(args, 'use_diversity_loss', True),
        diversity_loss_weight=diversity_loss_weight,
        use_balance_loss=use_balancing_loss,
        balance_loss_weight=balance_loss_weight,
        contrastive_temperature_init=getattr(args, 'temperature', 0.07),
        learnable_contrastive_temperature=False,
        dead_prototype_threshold=getattr(args, 'prototype_dead_threshold', 0.005),
        collect_debug_metrics=bool(getattr(args, 'log_debug_metrics', True)),
    )
