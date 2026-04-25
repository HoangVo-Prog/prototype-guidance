# Model Interface Contract

## Scope

This document describes the active PAS model interface after the host-plus-prototype integration update.

## 1. Primary Model Wrapper

- File: `model/build.py`
- Class: `PASModel`
- Builder: `build_model(args, num_classes)`

The model owns the CLIP backbone, a preserved host retrieval head, an optional prototype enhancement head, residual score fusion, freeze policy, optimizer-group exposure, retrieval encoding helpers, exact deployed retrieval scoring, and the combined host-plus-prototype training forward path.

Backbone support:
- PAS currently supports only `ViT-B/16`, `ViT-B/32`, and `ViT-L/14`.
- The runtime requires a ViT image token interface and `transformer_width == embed_dim` because the prototype path consumes text pre-projection token states.

Precision controls:
- `model.backbone_precision` controls host-side parameter precision (`host_backbone` + `host_retrieval`).
- `model.prototype_precision` controls prototype-side parameter precision (`prototype_bank`, `prototype_projector`, `routing`, `fusion`).
- `training.amp` and `training.amp_dtype` control CUDA autocast/scaler usage during training and retrieval evaluation.
- Unfrozen `fp16` backbone training is only supported when `training.amp=true`.
- `prototype_precision=fp16` training is only supported when `training.amp=true`.
- `model.normalize_projector_outputs` must remain `true` in the active runtime so training and retrieval stay on the same cosine-normalized embedding family.
- Exact retrieval scoring keeps a fixed temperature; there is no learnable logit-scale surface in the active runtime.
- `build_model(...)` requires `num_classes > 0` so class proxies are instantiated consistently for both training and evaluation builds.

## 2. Image-Side Interface

### Training and feature extraction

- Function: `PASModel.extract_image_features(image)`
- Input: `image` with shape `[B, C, H, W]`
- Return type: `model/interfaces.py:EncoderOutput`

Returned fields:
- `projected_tokens`: `[B, N_img, D]`
- `projected_pooled`: `[B, D]`
- `pre_projection_tokens`: optional `[B, N_img, D_pre]`
- `pre_projection_pooled`: optional `[B, D_pre]`
- `pooling_mode`: `cls`

### Retrieval interface

- Function: `PASModel.encode_image_for_retrieval(image)`
- Outputs:
  - `image_projected`: `[B, D_out]`
  - `summary`: `[B, D_proto]`
  - `routing_weights`: `[B, N_proto]`

## 3. Text-Side Interface

### Training and feature extraction

- Function: `PASModel.extract_text_features(text)`
- Input: `text` with shape `[B, L]`
- Return type: `model/interfaces.py:EncoderOutput`

Returned fields:
- `projected_tokens`: `[B, L, D]`
- `projected_pooled`: `[B, D]`
- `pre_projection_tokens`: optional `[B, L, D_pre]`
- `pre_projection_pooled`: optional `[B, D_pre]`
- `token_mask`: `[B, L]`
- `special_token_positions`: explicit EOS / CLS metadata
- `pooling_mode`: `image_conditioned`

### Exact deployed retrieval interface

- Function: `PASModel.encode_text_for_retrieval(text)`
- Outputs:
  - `text_token_states`: `[B, L, D_proto]`
  - `token_ids`: `[B, L]`
  - `attention_mask`: `[B, L]`
  - `special_token_positions`: dict used by exact pairwise pooling

### Optional approximate retrieval interface

- Function: `PASModel.encode_text_basis_for_retrieval(text)`
- Outputs:
  - `basis_bank`: `[B, N_proto, D_proto]`

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

### Exact deployed text branch
- Function: `pool_text_with_summary(summary, text_token_states, token_ids, ...)`
- Outputs:
  - `token_scores`: `[B, L]`
  - `token_weights`: `[B, L]`
  - `pooled_text`: `[B, D_proto]`
  - `text_projected`: `[B, D_out]`

### Amortized text basis branch
- Function: `build_text_basis_bank(text_token_states, token_ids, contextualized_prototypes, ...)`
- Outputs:
  - `basis_bank`: `[B, N, D_proto]`
  - optional `basis_token_scores`: `[B, N, L]`
  - optional `basis_token_weights`: `[B, N, L]`

### Surrogate reconstruction
- Function: `reconstruct_surrogate_text(routing_weights, basis_bank)`
- Inputs:
  - `routing_weights`: `[B, N]`
  - `basis_bank`: `[B, N, D_proto]`
- Output:
  - surrogate pooled text `[B, D_proto]`

## 5. Similarity and Loss Boundary

- File: `model/prototype/losses.py`
- Class: `PrototypeLosses`

### Training loss entrypoint
- Function: `PrototypeLosses.forward(image_embeddings, surrogate_text_embeddings, exact_text_embeddings, pids, prototypes=None, routing_weights=None, return_debug=False)`
- Outputs:
  - `loss_total`
  - `loss_proxy`
  - `loss_proxy_image`
  - `loss_proxy_text`
  - `loss_align`
  - `loss_diag`
  - `loss_support`
  - `loss_diversity`
  - `loss_balance`
  - weighted terms and lambda scalars
  - `proxy_temperature`
  - `retrieval_temperature`
  - `logit_scale`

The training objective is amortized surrogate training:
- main train-time text representation is the surrogate diagonal object
- exact diagonal deployed pooling is used only as a fidelity anchor
- no in-batch contrastive loss is used in the active runtime

### Retrieval similarity entrypoints
- Function: `PASModel.compute_retrieval_similarity(image_features, text_features)`
  - exact deployed scorer, default evaluator path
- Function: `PASModel.compute_approximate_retrieval_similarity(image_features, text_basis_features)`
  - optional approximate scorer, non-default

## 6. Optimizer Boundary

- Function: `PASModel.named_optimizer_groups()`
- Returned groups:
  - `prototype_bank`
  - `projectors`
  - `class_proxies`
  - `image_backbone`
  - `text_backbone`
  - `other`

## 7. Debug Output Contract

### Training forward
- Function: `PASModel.forward(batch, ..., return_debug=None)`
- Requires `batch['pids']` as class labels for the amortized proxy objective.
- Always returns:
  - `loss_total`
  - `loss_proxy`
  - `loss_align`
  - `loss_diag`
  - `loss_support`
  - `loss_diversity`
  - `loss_balance`
  - `proxy_temperature`
  - `retrieval_temperature`
  - `logit_scale`
  - `alpha`
  - `z_v`
  - `z_t_hat_diag`
  - `z_t_exact_diag`
- Optional when enabled:
  - `debug`

### Canonical debug keys

The `debug` dict may include:
- `alpha`
- `beta`
- `Q`
- `Theta_v`
- `Theta_tilde`
- `basis_bank`
- `T_hat_pool`
- `T_exact_pool`
- `Z_v`
- `Z_t`
- `Z_t_exact`
- routing / prototype-usage / geometry / norm diagnostics, including both entropy-based and IPR-based effective support metrics

## 8. Phase E Constraints

Future extensions may assume:
- Global image embeddings are available for routing.
- Token-level text states, masks, and special-token positions are available.
- Exact retrieval evaluation still uses deployed pairwise pooling `T(c_j, q_i)` by default.
- Training uses the amortized surrogate operator plus diagonal fidelity anchoring.

Future extensions must not assume:
- Exact deployed inference scoring was replaced by the surrogate scorer.
- Full pairwise `[B, B, ...]` training enumeration is part of the active runtime.
- Patch-token image routing exists in v1.
