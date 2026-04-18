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
    use_loss_dir = getattr(args, 'use_loss_dir', getattr(args, 'use_loss_diag', True))
    lambda_dir = getattr(args, 'lambda_dir', getattr(args, 'lambda_diag', 1.0))
    use_loss_sup = getattr(args, 'use_loss_sup', getattr(args, 'use_loss_support', False))
    lambda_sup = getattr(args, 'lambda_sup', getattr(args, 'lambda_support', 0.0))
    prototype_gap_margin = getattr(args, 'prototype_gap_margin', 0.05)
    prototype_support_target = getattr(args, 'prototype_support_target', getattr(args, 'support_min', 2.0))
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
    use_loss_weight_ret = bool(getattr(args, 'use_loss_weight_ret', False))
    lambda_weight_ret = float(getattr(args, 'lambda_weight_ret', 0.0))
    weight_ret_margin_delta = float(getattr(args, 'weight_ret_margin_delta', 0.0))
    weight_ret_tau = float(getattr(args, 'weight_ret_tau', 0.5))
    weight_ret_detach_host = bool(getattr(args, 'weight_ret_detach_host', True))
    weight_ret_normalize_mean_one = bool(getattr(args, 'weight_ret_normalize_mean_one', True))
    prototype_method_role = str(getattr(args, 'prototype_method_role', 'retrieval_branch')).lower()
    prototype_semantic_enabled = bool(getattr(args, 'prototype_semantic_enabled', prototype_method_role == 'semantic_structure'))
    semantic_structure_enabled = bool(getattr(args, 'semantic_structure_enabled', prototype_semantic_enabled))
    prototype_recompute_enabled = bool(
        getattr(args, 'prototype_recompute_enabled', prototype_semantic_enabled and prototype_method_role == 'semantic_structure')
    )
    prototype_bank_source = str(getattr(args, 'prototype_bank_source', 'learnable_legacy')).lower()
    prototype_use_contextualized_for_routing = bool(getattr(args, 'prototype_use_contextualized_for_routing', True))
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
    use_loss_semantic_pbt = bool(getattr(args, 'use_loss_semantic_pbt', False))
    lambda_semantic_pbt = float(getattr(args, 'lambda_semantic_pbt', 0.0))

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
        lambda_proxy=getattr(args, 'lambda_proxy', 1.0),
        lambda_proxy_image=getattr(args, 'lambda_proxy_image', getattr(args, 'lambda_proxy', 1.0)),
        lambda_proxy_text=getattr(args, 'lambda_proxy_text', getattr(args, 'lambda_proxy', 1.0)),
        lambda_proxy_text_exact=getattr(args, 'lambda_proxy_text_exact', getattr(args, 'lambda_proxy', 1.0)),
        use_loss_proxy_image=getattr(args, 'use_loss_proxy_image', True),
        use_loss_proxy_text=getattr(args, 'use_loss_proxy_text', True),
        use_loss_proxy_text_exact=getattr(args, 'use_loss_proxy_text_exact', True),
        use_loss_align=getattr(args, 'use_loss_align', True),
        lambda_align=getattr(args, 'lambda_align', 1.0),
        use_loss_dir=use_loss_dir,
        lambda_dir=lambda_dir,
        use_loss_gap=getattr(args, 'use_loss_gap', True),
        lambda_gap=getattr(args, 'lambda_gap', 0.5),
        fidelity_gap_margin=prototype_gap_margin,
        use_loss_sup=use_loss_sup,
        lambda_sup=lambda_sup,
        prototype_support_target=prototype_support_target,
        use_loss_diag=getattr(args, 'use_loss_diag', True),
        lambda_diag=getattr(args, 'lambda_diag', 1.0),
        diag_temperature=getattr(args, 'diag_temperature', 0.07),
        use_loss_ret=getattr(args, 'use_loss_ret', True),
        lambda_ret=getattr(args, 'lambda_ret', 1.0),
        use_loss_semantic_pbt=use_loss_semantic_pbt,
        lambda_semantic_pbt=lambda_semantic_pbt,
        use_loss_weight_ret=use_loss_weight_ret,
        lambda_weight_ret=lambda_weight_ret,
        weight_ret_margin_delta=weight_ret_margin_delta,
        weight_ret_tau=weight_ret_tau,
        weight_ret_detach_host=weight_ret_detach_host,
        weight_ret_normalize_mean_one=weight_ret_normalize_mean_one,
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
    support_loss_weight = lambda_sup
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
        prototype_use_contextualized_for_routing=prototype_use_contextualized_for_routing,
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
        lambda_proxy=getattr(args, 'lambda_proxy', 1.0),
        lambda_proxy_image=getattr(args, 'lambda_proxy_image', getattr(args, 'lambda_proxy', 1.0)),
        lambda_proxy_text=getattr(args, 'lambda_proxy_text', getattr(args, 'lambda_proxy', 1.0)),
        lambda_proxy_text_exact=getattr(args, 'lambda_proxy_text_exact', getattr(args, 'lambda_proxy', 1.0)),
        use_loss_proxy_image=getattr(args, 'use_loss_proxy_image', True),
        use_loss_proxy_text=getattr(args, 'use_loss_proxy_text', True),
        use_loss_proxy_text_exact=getattr(args, 'use_loss_proxy_text_exact', True),
        use_loss_align=getattr(args, 'use_loss_align', True),
        lambda_align=getattr(args, 'lambda_align', 1.0),
        use_loss_dir=use_loss_dir,
        lambda_dir=lambda_dir,
        use_loss_gap=getattr(args, 'use_loss_gap', True),
        lambda_gap=getattr(args, 'lambda_gap', 0.5),
        fidelity_gap_margin=prototype_gap_margin,
        use_loss_sup=use_loss_sup,
        lambda_sup=lambda_sup,
        prototype_support_target=prototype_support_target,
        use_loss_diag=getattr(args, 'use_loss_diag', True),
        lambda_diag=getattr(args, 'lambda_diag', 1.0),
        diag_temperature=getattr(args, 'diag_temperature', 0.07),
        use_loss_ret=getattr(args, 'use_loss_ret', True),
        lambda_ret=getattr(args, 'lambda_ret', 1.0),
        use_loss_semantic_pbt=use_loss_semantic_pbt,
        lambda_semantic_pbt=lambda_semantic_pbt,
        use_loss_weight_ret=use_loss_weight_ret,
        lambda_weight_ret=lambda_weight_ret,
        weight_ret_margin_delta=weight_ret_margin_delta,
        weight_ret_tau=weight_ret_tau,
        weight_ret_detach_host=weight_ret_detach_host,
        weight_ret_normalize_mean_one=weight_ret_normalize_mean_one,
        use_loss_support=use_loss_sup,
        support_loss_weight=support_loss_weight,
        support_min=prototype_support_target,
        use_diversity_loss=getattr(args, 'use_diversity_loss', True),
        diversity_loss_weight=diversity_loss_weight,
        use_balance_loss=use_balancing_loss,
        balance_loss_weight=balance_loss_weight,
        contrastive_temperature_init=getattr(args, 'temperature', 0.07),
        learnable_contrastive_temperature=False,
        dead_prototype_threshold=getattr(args, 'prototype_dead_threshold', 0.005),
        collect_debug_metrics=bool(getattr(args, 'log_debug_metrics', True)),
    )
