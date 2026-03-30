# MODULE_CONTRACTS

## model/prototype/prototype_bank.py
- Class: `PrototypeBank`
- Purpose: owns the learnable prototype table `Theta_v`.
- Inputs: none during `forward`; construction config controls shape and init.
- Outputs: `prototypes` `[N, D]`; optional debug with `raw_prototypes`, `prototype_norm_mean`, `prototype_norm_std`.
- Config dependencies: `prototype.num_prototypes`, `prototype.prototype_dim`, `prototype.prototype_init`, `prototype.prototype_init_path`, `prototype.prototype_normalize`.
- Failure conditions: invalid prototype count/dimension, unsupported init mode, or mismatched checkpoint init tensor.

## model/prototype/contextualizer.py
- Class: `PrototypeContextualizer`
- Purpose: optional parameter-free prototype self-contextualization.
- Inputs: `prototypes` `[N, D]`.
- Outputs: contextualized prototypes `[N, D]`; optional debug `contextualized_prototypes`, `prototype_similarity`, `contextualization_weights`, `prototype_contextualization_entropy`.
- Config dependencies: `prototype.contextualization_enabled`, `prototype.contextualization_type`, `prototype.contextualization_residual`, `prototype.prototype_normalize`.
- Failure conditions: unsupported contextualization type or invalid input rank.

## model/prototype/router.py
- Class: `Router`
- Purpose: routes each image embedding onto the prototype bank.
- Inputs: `image_embeddings` `[B, D]`, `prototypes` `[N, D]`.
- Outputs: `alpha` `[B, N]`; optional debug `routing_logits`, `routing_weights`, `routing_max_prob`, `prototype_assignment_entropy`.
- Config dependencies: `prototype.routing_type`, `prototype.routing_temperature`.
- Failure conditions: invalid rank, mismatched feature dimensions, non-positive temperature, or non-finite outputs.

## model/prototype/aggregator.py
- Class: `PrototypeAggregator`
- Purpose: computes the prototype summary `Q = alpha @ Theta_tilde`.
- Inputs: `routing_weights` `[B, N]`, `prototypes` `[N, D]`.
- Outputs: `summary` `[B, D]`; optional debug `prototype_summary`.
- Config dependencies: none beyond compatible shapes.
- Failure conditions: invalid rank or prototype-count mismatch.

## model/prototype/token_scorer.py
- Class: `TokenScorer`
- Purpose: scores each text token against the image-conditioned summary.
- Inputs: `query` `[B, D]`, `token_states` `[B, L, D]`.
- Outputs: `token_scores` `[B, L]`; optional debug `token_scores`.
- Config dependencies: `text_pooling.scoring_type`, `text_pooling.token_temperature`.
- Failure conditions: invalid ranks, dimension mismatch, non-positive temperature, or non-finite scores.

## model/prototype/token_mask.py
- Class: `TokenMaskBuilder`
- Purpose: builds the valid-token mask for pooling.
- Inputs: `token_ids` `[B, L]`, optional `attention_mask` `[B, L]`, optional `special_token_positions` with `cls` and `eos` tensors `[B]`.
- Outputs: `valid_mask` `[B, L]`; optional debug `valid_mask`, `special_token_positions`.
- Config dependencies: `text_pooling.token_policy`.
- Failure conditions: unsupported policy, missing special-token metadata, invalid EOS recovery, wrong input rank, or rows with zero valid tokens.

## model/prototype/token_pooler.py
- Class: `MaskedTokenPooler`
- Purpose: converts token scores into masked softmax weights and a pooled text representation.
- Inputs: `token_scores` `[B, L]`, `token_states` `[B, L, D]`, `valid_mask` `[B, L]`.
- Outputs: pooled text `[B, D]`, `beta` `[B, L]`; optional debug `masked_logits`, `token_weights`, `pooled_text`, `token_pool_entropy`.
- Config dependencies: upstream token scoring and masking policy.
- Failure conditions: invalid shapes, rows with zero valid tokens, or non-finite outputs.

## model/prototype/projector.py
- Class: `MLPProjector`
- Purpose: projects image and pooled-text features into the contrastive embedding space.
- Inputs: `inputs` with last dimension `input_dim`.
- Outputs: normalized projected features with last dimension `output_dim`; optional debug `projected_features`, `projected_features_pre_norm`.
- Config dependencies: `model.projection_dim`, `model.projector_hidden_dim`, `model.projector_dropout`.
- Failure conditions: non-positive dimensional arguments.

## model/prototype/losses.py
- Class: `PrototypeLosses`
- Purpose: provides symmetric InfoNCE plus optional prototype regularizers.
- Inputs: `image_embeddings` `[B, D]`, `text_embeddings` `[B, D]`, optional `prototypes` `[N, D]`, optional `routing_weights` `[B, N]`.
- Outputs: `loss_total`, `loss_infonce`, `loss_diversity`, `loss_balance`, `logit_scale`, optional `contrastive_logits` `[B, B]`.
- Config dependencies: `model.temperature`, `prototype.use_diversity_loss`, `prototype.diversity_loss_weight`, `prototype.balance_loss_weight`.
- Failure conditions: non-positive temperature, image/text shape mismatch, or invalid input rank.

## model/prototype/head.py
- Class: `PrototypeConditionedTextHead`
- Purpose: composes the full PAS branch for training and retrieval evaluation.
- Inputs: image embeddings `[B, D_img]`, text token states `[B, L, D_txt]`, token ids `[B, L]`, optional attention mask, optional special-token positions, `return_debug`.
- Outputs: image-side outputs, text-side outputs, projected embeddings, loss dict, and optional nested debug dict.
- Config dependencies: all prototype, text-pooling, projector, and loss keys consumed by `model/prototype/build.py`.
- Debug outputs: includes routing, pooling, prototype-usage, geometry, norm, and logit-scale diagnostics.
- Failure conditions: inherits submodule failure conditions.

## model/prototype/build.py
- Functions: `should_build_prototype_head`, `build_prototype_head`
- Purpose: config-driven construction of `PrototypeConditionedTextHead`.
- Inputs: runtime `args`, `input_dim`.
- Outputs: activation boolean or configured prototype head.
- Config dependencies: `model.use_prototype_bank`, `model.use_image_conditioned_pooling`, `model.use_prototype_contextualization`, plus prototype/text-pooling/loss settings.
- Failure conditions: delegated to the constructed modules.

## model/build.py
- Class: `PASModel`
- Purpose: primary model wrapper for training, retrieval evaluation, freeze policy, and optimizer grouping.
- Inputs:
  - `forward(batch, ...)` expects `batch['images']` and `batch['caption_ids']`.
  - `encode_image_for_retrieval(image)` expects `[B, C, H, W]`.
  - `encode_text_for_retrieval(text)` expects `[B, L]`.
- Outputs:
  - `forward(...)` returns `loss_total`, loss breakdown, `temperature`, `logit_scale`, optional `logits`, optional `debug`.
  - `named_optimizer_groups()` returns explicit optimizer-group buckets.
  - `compute_retrieval_similarity(...)` returns `[N_text, N_image]` similarity blocks.
- Config dependencies: model activation flags, prototype settings, freeze policy, evaluation chunk sizes.
- Failure conditions: rejects disabled prototype mode, unsupported sparse routing, unsupported contextualization depth, invalid evaluation chunk sizes, or non-finite loss/similarity outputs.

## utils/metric_logging.py
- Functions: `collect_loss_metrics`, `collect_debug_metrics`, `collect_scalar_metrics`, `build_train_metrics`, `build_validation_metrics`
- Purpose: centralized scalar extraction and logging-name normalization.
- Inputs: forward output dicts, optional evaluator metrics, epoch/step/lr values.
- Outputs: flat metric dicts ready for TensorBoard or Weights & Biases.
- Assumptions: non-scalar tensors stay in the debug dict and are not logged as scalars.

## solver/build.py
- Function: `build_optimizer(args, model)`
- Purpose: constructs optimizer param groups from `named_optimizer_groups()`.
- Inputs: runtime args with per-group LR settings and a model implementing `named_optimizer_groups()`.
- Outputs: configured `torch.optim` optimizer.
- Config dependencies: `optimizer.lr`, `optimizer.lr_prototype_bank`, `optimizer.lr_contextualizer`, `optimizer.lr_projectors`, `optimizer.lr_logit_scale`, `optimizer.lr_image_backbone`, `optimizer.lr_text_backbone`, `optimizer.weight_decay`, optimizer type.
- Failure conditions: model missing `named_optimizer_groups()` or unsupported optimizer type.

