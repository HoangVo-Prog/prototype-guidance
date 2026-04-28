# Prototype Flow In This Codebase

This document explains the prototype path end-to-end: how it is initialized, how batch inputs flow through prototype modules, and how prototype signals contribute to final retrieval scores.

## 1) Entry Point and Model Construction

### 1.1 Config to runtime args

- YAML keys are flattened into runtime args in `utils/config.py` (`PRIMARY_CONFIG_KEY_MAP`), including:
- `model.runtime_mode -> args.runtime_mode`
- `model.use_prototype_branch -> args.use_prototype_branch`
- `model.use_prototype_bank -> args.use_prototype_bank`
- `prototype.* -> args.prototype_*`
- `fusion.* -> args.fusion_*`
- `objectives.* -> loss flags/weights (`use_loss_ret`, `lambda_ret`, etc.)

So prototype behavior is mostly controlled by `model`, `prototype`, `fusion`, and `objectives` sections in config.

### 1.2 Builder routing

`model/build.py` chooses runtime implementation:

- If prototype branch is active and runtime mode is explicit (`prototype_only`, `joint_training`, etc.), it builds the structural split model via `build_structural_split_model(...)`.
- That returns `PASRuntimeModel` from `model/plug_and_play.py`, which is the main runtime path for prototype-enabled modes.

### 1.3 What `PASRuntimeModel` wires

`PASRuntimeModel` composes three parts:

- `HostCore`: host encoder + host head path.
- `PrototypePlugin`: adapter that calls `prototype_head`.
- `Composer`: fusion module for host/prototype similarity composition.

Prototype module internals are created inside `prototype_head` (`model/prototype/head.py`), including:

- `PrototypeBank` (learnable prototypes)
- `PrototypeContextualizer`
- `Router`
- `PrototypeAggregator`
- Token modules (`TokenMaskBuilder`, `TokenScorer`, `MaskedTokenPooler`)
- Projectors (`image_projector`, `text_projector`)
- `PrototypeLosses`

## 2) Training-Time Data Flow (Batch -> Prototype Loss)

This is the core path in `PASRuntimeModel.forward(...)`.

### 2.1 Host forward runs first

Given batch inputs (`images`, `caption_ids`, `pids`):

1. `HostCore.forward_host(...)` extracts image/text features from the CLIP backbone.
2. Host head computes host outputs and host losses (`host_outputs['losses']`).

### 2.2 Runtime mode decides prototype activation

- In `host_only`: prototype plugin is bypassed and replaced with `_empty_prototype_outputs(...)`.
- In `prototype_only`, `fused_external`, `joint_training`, `calibration_only`: host exports an interface and prototype plugin executes.

### 2.3 Host -> prototype interface handoff

`HostCore.build_plugin_interface(...)` exports:

- image embeddings
- text token states
- token ids / masks / special-token positions
- optional host pairwise logits (mode-dependent)

`PrototypePlugin.forward_from_interface(...)` validates interface version and calls `prototype_head.forward(...)`.

### 2.4 Inside `prototype_head.forward(...)`

Prototype forward in `model/prototype/head.py` is:

1. Build prototype context
- `get_prototype_context()`
- `prototypes = PrototypeBank(...)`
- `contextualized_prototypes = Contextualizer(prototypes)`

2. Encode image branch
- `image_features = image_adapter(image_embeddings)`
- Compute routing weights `alpha` via router:
- global routing or local-evidence routing (depends on `prototype.routing_source`)
- Aggregate prototype summary:
- `summary = alpha @ contextualized_prototypes`
- Build prototype image representation:
- `image_proxy_features = image_features + proto_query_proj(summary)`
- `image_projected = image_projector(image_proxy_features)`

3. Build text basis bank per prototype
- For each text sample and each prototype query, pool token states.
- Produces `basis_bank` with shape roughly `[B_text, N_proto, D]`.

4. Reconstruct surrogate text for each image
- `surrogate_pooled_text = einsum(alpha, basis_bank)` (image-conditioned)
- `surrogate_text_projected = text_projector(surrogate_pooled_text)`

5. Build exact text projection for diagnostics/comparison
- `exact_outputs = pool_text_with_summary(summary, text_token_states, token_ids, ...)`
- Produces `exact_text_projected`.

6. Build surrogate pairwise logits (if retrieval loss enabled)
- If `use_loss_ret`, compute `surrogate_pairwise_logits`.

7. Compute prototype losses
- `PrototypeLosses.forward(...)` consumes projected image/text features, routing, optional surrogate logits, optional host logits.
- Returns `loss_total` plus components (`loss_ret`, `loss_diag`, `loss_support`, etc. depending on config).

### 2.5 Loss used for optimization by runtime mode

Back in `PASRuntimeModel.forward(...)`:

- `joint_training`: `loss_total = lambda_host * host_loss_total + prototype_loss_total`
- `prototype_only` and `fused_external`: `loss_total = prototype_loss_total`
- `host_only`: `loss_total = lambda_host * host_loss_total`
- `calibration_only`: `loss_total = composer calibration loss`

So in `prototype_only`, optimization is fully carried by prototype loss.

## 3) Evaluation-Time Flow (Features -> Final Score)

Evaluation lives in `utils/metrics.py` (`Evaluator`).

### 3.1 Feature extraction

- Text side:
- `encode_text_for_retrieval(...)` for exact scoring
- `encode_text_basis_for_retrieval(...)` for approximate scoring
- Image side:
- `encode_image_for_retrieval(...)`

When prototype plugin is active, retrieval features include prototype-specific tensors:

- image: `prototype_image_projected`, `prototype_summary`, `routing_weights`
- text exact: token states + ids (for exact prototype similarity)
- text approximate: `basis_bank`

### 3.2 Similarity components

`PASRuntimeModel` computes:

- `host_similarity`: from host head similarity path.
- `prototype_similarity`:
- exact mode: prototype head exact pairwise similarity from token states and summaries
- approximate mode: prototype head approximate similarity from routing weights + basis bank

### 3.3 Final score composition

`fuse_retrieval_similarity(...)` calls `Composer.fuse(...)`, which applies:

- schema checks
- optional calibration scaling
- weighted fusion through `ResidualScoreFusion`

Final formula is:

`S_final = lambda_host * S_host + lambda_prototype * S_proto`

If prototype similarity is missing, prototype weight must be zero.

### 3.4 Metric rows and subset sweeps

Evaluator can emit:

- default `pas-t2i` row (default fusion weights)
- explicit `host-t2i`, `prototype-t2i`
- optional `fusion.eval_subsets` sweep rows

Each row is computed from the same host/prototype component similarities with different lambda pairs.

## 4) Tensor-Level Prototype Contribution Summary

Prototype contribution enters the final score through this chain:

1. `PrototypeBank -> contextualized prototypes`
2. `Router -> routing weights alpha`
3. `alpha + basis_bank -> surrogate text`
4. `image_projected` and prototype text projection -> `S_proto`
5. `S_proto` fused with `S_host` by composer/fusion weights

In short:

- During prototype training (`prototype_only`), prototype branch controls optimization via `prototype_loss_total`.
- During retrieval/eval, prototype branch contributes via `S_proto` in weighted fusion.

## 5) Fast File Map

- Config flattening: `utils/config.py`
- Runtime build routing: `model/build.py`
- Structural split runtime model: `model/plug_and_play.py`
- Prototype core logic: `model/prototype/head.py`
- Prototype loss definitions: `model/prototype/losses.py`
- Score fusion: `model/fusion.py`
- Retrieval evaluation and fusion row generation: `utils/metrics.py`
