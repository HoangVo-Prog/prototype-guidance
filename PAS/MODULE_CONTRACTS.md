# MODULE_CONTRACTS

## model/prototype/prototype_bank.py
- Class: `PrototypeBank`
- Purpose: owns the learnable prototype table `Theta_v`.
- Outputs: `prototypes` `[N, D]`; optional debug with `raw_prototypes`, norm stats, and init metadata.

## model/prototype/contextualizer.py
- Class: `PrototypeContextualizer`
- Purpose: optional parameter-free prototype self-contextualization.
- Inputs: `prototypes` `[N, D]`.
- Outputs: contextualized prototypes `[N, D]`; optional debug with contextualization weights and entropy.

## model/prototype/router.py
- Class: `Router`
- Purpose: routes each image embedding onto the prototype bank.
- Inputs: `image_embeddings` `[B, D]`, `prototypes` `[N, D]`.
- Outputs: `alpha` `[B, N]`; optional debug includes `routing_effective_support`.

## model/prototype/aggregator.py
- Class: `PrototypeAggregator`
- Purpose: computes the prototype summary `Q = alpha @ Theta_tilde`.
- Inputs: `routing_weights` `[B, N]`, `prototypes` `[N, D]`.
- Outputs: `summary` `[B, D]`.

## model/prototype/token_scorer.py
- Class: `TokenScorer`
- Purpose: scores text tokens against either an image-conditioned summary or a contextualized prototype query.
- Inputs: `query` `[B, D]`, `token_states` `[B, L, D]`.
- Outputs: token scores `[B, L]`.

## model/prototype/token_mask.py
- Class: `TokenMaskBuilder`
- Purpose: builds the valid-token mask for pooling.
- Inputs: `token_ids` `[B, L]`, optional `attention_mask`, optional `special_token_positions`.
- Outputs: keep-mask `[B, L]`; optional debug includes special-token positions.

## model/prototype/token_pooler.py
- Class: `MaskedTokenPooler`
- Purpose: converts token scores into masked softmax weights and pooled text states.
- Inputs: `token_scores` `[B, L]`, `token_states` `[B, L, D]`, `valid_mask` `[B, L]`.
- Outputs: pooled text `[B, D]`, `beta` `[B, L]`; optional debug includes masked logits and entropy.

## model/prototype/projector.py
- Class: `MLPProjector`
- Purpose: projects image features and pooled text states into the retrieval embedding space.
- Inputs: `inputs` with last dimension `input_dim`.
- Outputs: projected features with last dimension `output_dim`; optional debug includes raw and normalized projections.
- Note: the active PAS runtime requires `model.normalize_projector_outputs=true` so proxy supervision and retrieval scoring use the same cosine-normalized embedding family.

## model/prototype/losses.py
- Class: `PrototypeLosses`
- Purpose: implements the amortized surrogate objective plus prototype regularizers.
- Inputs:
  - `image_embeddings` `[B, D_out]`
  - `surrogate_text_embeddings` `[B, D_out]`
  - `exact_text_embeddings` `[B, D_out]`
  - `pids` `[B]` used as class labels for proxy supervision
  - optional `prototypes` `[N, D]`
  - optional `routing_weights` `[B, N]`
- Outputs:
  - `loss_total`
  - `loss_proxy`, `loss_proxy_image`, `loss_proxy_text`
  - `loss_align`
  - `loss_diag`
  - `loss_diversity`
  - `loss_balance`
  - weighted terms, lambda scalars, `proxy_temperature`, and fixed retrieval scaling values
- Additional state: learnable `class_proxies` `[C, D_out]`
- Active behavior: no symmetric InfoNCE / in-batch contrastive objective in the active runtime.

## model/prototype/head.py
- Class: `PrototypeConditionedTextHead`
- Purpose: composes the full PAS branch for amortized training and retrieval evaluation.
- Key sub-interfaces:
  - `encode_image_branch(...)` builds `alpha`, `Q`, and image embeddings.
  - `pool_text_with_summary(...)` computes the exact deployed pooled text object.
  - `build_text_basis_bank(...)` computes the per-caption prototype basis bank `[B, N, D]`.
  - `reconstruct_surrogate_text(...)` combines image routing with the basis bank.
  - `compute_pairwise_similarity(...)` preserves exact deployed inference.
  - `compute_approximate_pairwise_similarity(...)` exposes the optional non-default approximate scorer.
- Training outputs include both surrogate and exact diagonal text embeddings.

## model/prototype/build.py
- Function: `build_prototype_head(args, input_dim, num_classes)`
- Purpose: config-driven construction of `PrototypeConditionedTextHead` for the amortized objective.
- Important args: fixed retrieval temperature, proxy temperature, `lambda_proxy`, `lambda_align`, `lambda_diag`, prototype regularizer weights, and `num_classes > 0`.

## model/build.py
- Class: `PASModel`
- Purpose: primary model wrapper for training, retrieval evaluation, freeze policy, and optimizer grouping.
- Training forward expects `batch['images']`, `batch['caption_ids']`, and `batch['pids']`.
- Retrieval helpers:
  - `encode_image_for_retrieval(...)`
  - `encode_text_for_retrieval(...)`
  - `encode_text_basis_for_retrieval(...)`
  - `compute_retrieval_similarity(...)` for exact deployed scoring
  - `compute_approximate_retrieval_similarity(...)` for the optional approximate scorer
- Optimizer groups include `class_proxies`.

## utils/metric_logging.py
- Purpose: centralized scalar extraction for the amortized loss breakdown and debug metrics.
- Tracked train losses now include `loss_proxy`, `loss_align`, and `loss_diag` instead of `loss_infonce`.

## solver/build.py
- Function: `build_optimizer(args, model)`
- Purpose: constructs optimizer param groups from `named_optimizer_groups()`.
- Relevant group-specific knobs include `optimizer.lr_class_proxies` and `optimizer.weight_decay_class_proxies`.
