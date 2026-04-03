from typing import Optional

import torch
import torch.nn as nn

from .head import PrototypeConditionedTextHead


def build_prototype_head(
    args,
    input_dim: int,
    num_classes: int,
    image_adapter: Optional[nn.Module] = None,
    text_adapter: Optional[nn.Module] = None,
    prototype_init_features: Optional[torch.Tensor] = None,
) -> PrototypeConditionedTextHead:
    contextualization_enabled = bool(getattr(args, 'prototype_contextualization_enabled', True))
    projector_output_dim = getattr(args, 'projector_output_dim', getattr(args, 'projection_dim', 256))
    routing_type = getattr(args, 'routing_similarity', getattr(args, 'prototype_routing_type', 'cosine'))
    routing_temperature = getattr(args, 'tau_p', getattr(args, 'prototype_temperature', 0.07))
    token_scoring_type = getattr(args, 'token_similarity', getattr(args, 'token_scoring_type', 'cosine'))
    token_temperature = getattr(args, 'tau_t', getattr(args, 'token_pooling_temperature', 0.07))
    support_loss_weight = getattr(args, 'lambda_support', 0.0)
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
        token_scoring_type=token_scoring_type,
        token_temperature=token_temperature,
        token_policy=getattr(args, 'token_policy', 'content_only'),
        special_token_ids=getattr(args, 'special_token_ids', None),
        error_on_empty_kept_tokens=getattr(args, 'error_on_empty_kept_tokens', True),
        contextualization_enabled=contextualization_enabled,
        contextualization_type=getattr(args, 'prototype_contextualization_type', 'none'),
        contextualization_residual=getattr(args, 'prototype_contextualization_residual', True),
        normalize_for_self_interaction=getattr(args, 'normalize_for_self_interaction', True),
        normalize_for_routing=getattr(args, 'normalize_for_routing', True),
        normalize_for_token_scoring=getattr(args, 'normalize_for_token_scoring', True),
        normalize_projector_outputs=getattr(args, 'normalize_projector_outputs', True),
        num_classes=num_classes,
        proxy_temperature=getattr(args, 'proxy_temperature', 0.07),
        lambda_proxy=getattr(args, 'lambda_proxy', 1.0),
        use_loss_proxy_image=getattr(args, 'use_loss_proxy_image', True),
        use_loss_proxy_text=getattr(args, 'use_loss_proxy_text', True),
        use_loss_proxy_text_exact=getattr(args, 'use_loss_proxy_text_exact', True),
        use_loss_align=getattr(args, 'use_loss_align', True),
        lambda_align=getattr(args, 'lambda_align', 1.0),
        use_loss_diag=getattr(args, 'use_loss_diag', True),
        lambda_diag=getattr(args, 'lambda_diag', 1.0),
        use_loss_ret_exact=getattr(args, 'use_loss_ret_exact', False),
        lambda_ret_exact=getattr(args, 'lambda_ret_exact', 1.0),
        ret_exact_temperature=getattr(args, 'ret_exact_temperature', None),
        use_loss_support=getattr(args, 'use_loss_support', False),
        support_loss_weight=support_loss_weight,
        support_min=getattr(args, 'support_min', 2.0),
        use_diversity_loss=getattr(args, 'use_diversity_loss', True),
        diversity_loss_weight=diversity_loss_weight,
        use_balance_loss=use_balancing_loss,
        balance_loss_weight=balance_loss_weight,
        contrastive_temperature_init=getattr(args, 'temperature', 0.07),
        learnable_contrastive_temperature=False,
        dead_prototype_threshold=getattr(args, 'prototype_dead_threshold', 0.005),
    )
