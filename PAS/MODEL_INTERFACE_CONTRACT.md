# Model Interface Contract

## Scope

This document describes the active Phase E model interface for the PAS retrieval system.

## 1. Primary Model Wrapper

- File: `model/build.py`
- Class: `PASModel`
- Builder: `build_model(args, num_classes=0)`

The model owns the CLIP backbone, prototype head, freeze policy, optimizer-group exposure, retrieval encoding helpers, and training forward path.

Precision controls:
- `model.backbone_precision` selects whether CLIP backbone parameters are kept in `fp16` or `fp32`.
- `model.prototype_precision` selects whether prototype-head parameters are kept in `fp16` or `fp32`.
- `training.amp` and `training.amp_dtype` control CUDA autocast/scaler usage during training and retrieval evaluation.
- Unfrozen `fp16` backbone training is only supported when `training.amp=true`.
- `prototype_precision=fp16` training is only supported when `training.amp=true`.

## 2. Image-Side Interface

### Training and feature extraction

- Function: `PASModel.extract_image_features(image)`
- Input: `image` with shape `[B, C, H, W]`
- Return type: `model/interfaces.py:EncoderOutput`

Returned fields:
- `projected_tokens`: `[B, N_img, D]`
- `projected_pooled`: `[B, D]`
  Current v1 path uses the CLS/global image token.
- `pre_projection_tokens`: optional `[B, N_img, D_pre]`
- `pre_projection_pooled`: optional `[B, D_pre]`
- `pooling_mode`: `cls`

### Retrieval interface

- Function: `PASModel.encode_image_for_retrieval(image)`
- Outputs:
  - `image_projected`: `[B, D_out]`
  - `summary`: `[B, D_proto]`

Phase E assumption: routing uses the global image embedding only.

## 3. Text-Side Interface

### Training and feature extraction

- Function: `PASModel.extract_text_features(text)`
- Input: `text` with shape `[B, L]`
- Return type: `model/interfaces.py:EncoderOutput`

Returned fields:
- `projected_tokens`: `[B, L, D]`
- `projected_pooled`: `[B, D]`
  EOS pooled for the raw encoder path.
- `pre_projection_tokens`: optional `[B, L, D_pre]`
- `pre_projection_pooled`: optional `[B, D_pre]`
- `token_mask`: `[B, L]` boolean valid-token mask derived from explicit token metadata
- `special_token_positions`:
  - `cls`: optional `[B]` leading special-token positions when configured
  - `eos`: `[B]` EOS positions derived from explicit token ids or validated metadata
- `pooling_mode`: `image_conditioned`

### Retrieval interface

- Function: `PASModel.encode_text_for_retrieval(text)`
- Outputs:
  - `text_token_states`: `[B, L, D_proto]`
  - `token_ids`: `[B, L]`
  - `attention_mask`: `[B, L]`
  - `special_token_positions`: dict with explicit special-token positions used for deterministic masking

## 4. Prototype Boundary

- File: `model/prototype/head.py`
- Class: `PrototypeConditionedTextHead`

### Image branch
- Function: `encode_image_branch(image_embeddings, ...)`
- Inputs:
  - `image_embeddings`: `[B, D_proto]`
- Outputs:
  - `prototypes`: `[N, D_proto]`
  - `contextualized_prototypes`: `[N, D_proto]`
  - `routing_weights`: `[B, N]`
  - `summary`: `[B, D_proto]`
  - `image_projected`: `[B, D_out]`

### Text branch
- Function: `pool_text_with_summary(summary, text_token_states, token_ids, ...)`
- Inputs:
  - `summary`: `[B, D_proto]`
  - `text_token_states`: `[B, L, D_proto]`
  - `token_ids`: `[B, L]`
- Outputs:
  - `token_scores`: `[B, L]`
  - `valid_mask`: `[B, L]`
  - `token_weights`: `[B, L]`
  - `pooled_text`: `[B, D_proto]`
  - `text_projected`: `[B, D_out]`

## 5. Similarity and Loss Boundary

- File: `model/prototype/losses.py`
- Class: `PrototypeLosses`

### Training loss entrypoint
- Function: `PrototypeLosses.forward(image_embeddings, text_embeddings, prototypes=None, routing_weights=None, return_debug=False)`
- Outputs:
  - `loss_total`
  - `loss_infonce`
  - `loss_diversity`
  - `loss_balance`
  - `logit_scale`
  - optional `contrastive_logits` with shape `[B, B]`

### Retrieval similarity entrypoint
- Function: `PASModel.compute_retrieval_similarity(image_features, text_features)`
- Delegates to: `PrototypeConditionedTextHead.compute_pairwise_similarity(...)`
- Output: similarity matrix `[N_text, N_image]`

## 6. Optimizer Boundary

- Function: `PASModel.named_optimizer_groups()`
- Returned groups:
  - `prototype_bank`
  - `projectors`
  - `logit_scale`
  - `image_backbone`
  - `text_backbone`
  - `other`

`solver/build.py` is the only active optimizer-construction surface.

## 7. Debug Output Contract

### Training forward
- Function: `PASModel.forward(batch, ..., return_debug=None)`
- Always returns:
  - `loss_total`
  - `loss_infonce`
  - `loss_diversity`
  - `loss_balance`
  - `temperature`
  - `logit_scale`
- Optional when enabled:
  - `logits`
  - `debug`

### Canonical debug keys

The `debug` dict may include:
- `image_global`
- `text_tokens`
- `token_mask`
- `special_token_positions`
- `alpha`
- `beta`
- `Q`
- `Theta_v`
- `Theta_tilde`
- `routing_max_prob`
- `routing_entropy`
- `prototype_usage_entropy`
- `prototype_dead_count`
- `token_pool_entropy`
- `token_special_mass`
- `valid_token_fraction`
- `prototype_pairwise_cosine_mean`
- `prototype_pairwise_cosine_max`
- `q_norm`
- `t_pool_norm`
- `image_embed_norm`
- `text_embed_norm`

## 8. Phase E Constraints

Future extensions may assume:
- Global image embeddings are available for routing.
- Token-level text states, masks, and special-token positions are available.
- Retrieval evaluation uses projected image embeddings plus image-conditioned pooled text embeddings.
- Loss bookkeeping is structured around `loss_total` and named sub-losses.

Future extensions must not assume:
- Patch-token image routing exists in v1.
- Sparse prototype assignment or deeper contextualization require explicit hyperparameter configuration.
- Legacy ITSELF or GRAB branches are not part of the active model path.



