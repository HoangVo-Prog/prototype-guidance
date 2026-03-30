from .head import PrototypeConditionedTextHead


def should_build_prototype_head(args) -> bool:
    return bool(
        getattr(args, 'use_prototype_bank', False)
        or getattr(args, 'use_image_conditioned_pooling', False)
        or getattr(args, 'use_prototype_contextualization', False)
        or getattr(args, 'prototype_contextualization_enabled', False)
    )


def build_prototype_head(args, input_dim: int) -> PrototypeConditionedTextHead:
    contextualization_enabled = bool(
        getattr(args, 'prototype_contextualization_enabled', False)
        or getattr(args, 'use_prototype_contextualization', False)
    )
    projector_output_dim = getattr(args, 'projector_output_dim', getattr(args, 'projection_dim', 256))
    routing_type = getattr(args, 'routing_similarity', getattr(args, 'prototype_routing_type', 'cosine'))
    routing_temperature = getattr(args, 'tau_p', getattr(args, 'prototype_temperature', 0.07))
    token_scoring_type = getattr(args, 'token_similarity', getattr(args, 'token_scoring_type', 'cosine'))
    token_temperature = getattr(args, 'tau_t', getattr(args, 'token_pooling_temperature', 0.07))
    diversity_loss_weight = getattr(args, 'lambda_div', getattr(args, 'diversity_loss_weight', 0.01))
    balance_loss_weight = getattr(args, 'lambda_bal', getattr(args, 'prototype_balance_loss_weight', 0.0))
    return PrototypeConditionedTextHead(
        input_dim=input_dim,
        num_prototypes=getattr(args, 'prototype_num_prototypes', 32),
        prototype_dim=getattr(args, 'prototype_dim', input_dim),
        projector_output_dim=projector_output_dim,
        projector_hidden_dim=getattr(args, 'projector_hidden_dim', getattr(args, 'prototype_dim', input_dim)),
        projector_dropout=getattr(args, 'projector_dropout', 0.0),
        prototype_init=getattr(args, 'prototype_init', 'normalized_random'),
        prototype_init_path=getattr(args, 'prototype_init_path', None),
        routing_type=routing_type,
        routing_temperature=routing_temperature,
        token_scoring_type=token_scoring_type,
        token_temperature=token_temperature,
        token_policy=getattr(args, 'token_policy', 'content_only'),
        contextualization_enabled=contextualization_enabled,
        contextualization_type=getattr(args, 'prototype_contextualization_type', 'none'),
        contextualization_residual=getattr(args, 'prototype_contextualization_residual', True),
        prototype_normalize=getattr(args, 'prototype_normalize', True),
        use_diversity_loss=getattr(args, 'use_diversity_loss', True),
        diversity_loss_weight=diversity_loss_weight,
        use_balance_loss=balance_loss_weight > 0,
        balance_loss_weight=balance_loss_weight,
        contrastive_temperature_init=getattr(args, 'temperature', 0.07),
        learnable_contrastive_temperature=True,
        dead_prototype_threshold=getattr(args, 'prototype_dead_threshold', 0.005),
    )
