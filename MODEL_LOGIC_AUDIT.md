# MODEL_LOGIC_AUDIT

## 1. Executive summary

The implementation is runnable on its default `ViT-B/16` path, but it is not logically consistent end to end.

Major logic risks exist. The most serious one is a core train/inference mismatch: training forms each text embedding under only its paired image summary, while inference recomputes the same text under every candidate image summary. The model is therefore not trained on the scoring function it uses at evaluation time.

There is a second major supervision issue: the training sampler intentionally packs multiple same-identity samples into a batch, but the InfoNCE loss treats only the diagonal pair as positive. Off-diagonal same-identity image/text pairs become false negatives.

The pipeline is mathematically valid in the narrow sense that tensor shapes and reductions mostly line up, but several parts are semantically wrong or misleading:

- training score semantics do not match inference score semantics
- batch labels do not match retrieval identity semantics
- some config switches are not real switches
- some advertised backbone choices are not actually supported by the executed PAS path

Bottom line: the code should not be trusted for serious experiments until the train/infer scoring mismatch and batch-positive semantics are fixed.

## 2. Real implemented pipeline

Actual entry points:

- training script: `PAS/train.py`
- inference script: `PAS/test.py`
- train loop: `PAS/processor/processor.py::do_train`
- eval loop: `PAS/utils/metrics.py::Evaluator.eval`
- model build: `PAS/model/build.py::build_model`
- runtime model forward: `PAS/model/build.py::PASModel.forward`

Stepwise executed pipeline:

1. Input batch
- Location: `PAS/datasets/bases.py::ImageTextDataset.__getitem__`, `PAS/datasets/build.py::collate`
- Inputs: image tensor `[B, 3, H, W]`, token ids `caption_ids` `[B, L]`, plus `pids`, `image_ids`
- Outputs used by model: `batch['images']`, `batch['caption_ids']`
- Always active in training

2. Image backbone encoding
- Location: `PAS/model/build.py::extract_image_features`, calling `PAS/model/clip_model.py::CLIP.encode_image_intermediates`
- Input: images `[B, 3, H, W]`
- Output on ViT path:
  - `projected_tokens` `[B, T_img, D_embed]`
  - `image_global = projected_tokens[:, 0, :]` `[B, D_embed]`
  - optional `pre_projection_tokens` `[B, T_img, D_backbone]`
- Always active
- Important semantic fact: PAS uses the projected CLS image token as the image branch input to the prototype head

3. Text backbone encoding
- Location: `PAS/model/build.py::extract_text_features`, calling `PAS/model/clip_model.py::CLIP.encode_text_intermediates`
- Input: token ids `[B, L]`
- Outputs:
  - `projected_tokens` `[B, L, D_embed]`
  - `pre_projection_tokens` `[B, L, D_text_backbone]`
  - `token_mask` `[B, L]` from `PAS/model/prototype/token_mask.py::TokenMaskBuilder.build_valid_mask`
  - `special_token_positions` such as EOS from `TokenMaskBuilder.get_special_token_positions`
- Always active
- Important semantic fact: PAS does not pool or score text with CLIP projected tokens. It uses `pre_projection_tokens` via `PASModel._resolve_text_states`

4. Prototype bank retrieval
- Location: `PAS/model/prototype/head.py::get_prototype_context`
- Input: none other than learned parameters
- Outputs:
  - `Theta_v = prototypes` `[N_proto, D_proto]`
  - `Theta_tilde = contextualized_prototypes` `[N_proto, D_proto]`
- Config dependent:
  - bank always active in runtime
  - contextualization active only if enabled and type is not `none`

5. Image-to-prototype routing
- Location: `PAS/model/prototype/head.py::encode_image_branch`
- Input: image embedding `[B, D_embed]`
- Internal steps:
  - optional `image_adapter`: `[B, D_embed] -> [B, D_proto]`
  - `alpha = Router(image_features, Theta_tilde)` in `PAS/model/prototype/router.py`
  - `Q = alpha @ Theta_tilde` in `PAS/model/prototype/aggregator.py`
  - `Z_v = image_projector(image_features)` in `PAS/model/prototype/projector.py`
- Outputs:
  - `routing_weights = alpha` `[B, N_proto]`
  - `summary = Q` `[B, D_proto]`
  - `image_projected = Z_v` `[B, D_out]`
- Always active
- Important semantic fact: `Z_v` is projected directly from the image feature, not from `Q`

6. Image-conditioned text scoring and pooling
- Location: `PAS/model/prototype/head.py::pool_text_with_summary`
- Inputs:
  - summary `Q` `[B, D_proto]`
  - text token states `[B, L, D_text_backbone]`
  - token ids `[B, L]`
- Internal steps:
  - optional `text_adapter`: `[B, L, D_text_backbone] -> [B, L, D_proto]`
  - token scores `S_t` from `PAS/model/prototype/token_scorer.py::TokenScorer`
  - keep-mask from `TokenMaskBuilder.build`
  - token weights `beta` from `PAS/model/prototype/token_pooler.py::MaskedTokenPooler`
  - pooled text `T_pool = sum_l beta_l h_l`
  - `Z_t = text_projector(T_pool)`
- Outputs:
  - `token_scores` `[B, L]`
  - `token_keep_mask` `[B, L]`
  - `token_weights = beta` `[B, L]`
  - `pooled_text = T_pool` `[B, D_proto]`
  - `text_projected = Z_t` `[B, D_out]`
- Always active

7. Training loss
- Location: `PAS/model/prototype/head.py::forward` then `PAS/model/prototype/losses.py::PrototypeLosses.forward`
- Inputs:
  - `image_projected = Z_v` `[B, D_out]`
  - `text_projected = Z_t` `[B, D_out]`
  - optionally `prototypes`, `routing_weights`
- Loss terms:
  - symmetric InfoNCE over batch rows
  - optional diversity loss on prototype cosine matrix
  - optional balance loss on mean routing usage
- Effective optimization targets:
  - diagonal row pairing only for InfoNCE
  - prototype separation via diversity loss
  - uniform average prototype usage via balance loss

8. Training loop integration
- Location: `PAS/processor/processor.py::do_train`
- Behavior:
  - `outputs = model(batch)`
  - backprop only through `outputs['loss_total']`
  - no separate eval-time objective

9. Inference / evaluation path
- Location: `PAS/utils/metrics.py::Evaluator._compute_similarity`
- Steps:
  - encode all texts independently with `PASModel.encode_text_for_retrieval`
  - encode all images independently with `PASModel.encode_image_for_retrieval`
  - compute full pairwise similarity with `PASModel.compute_retrieval_similarity`
- Important semantic fact:
  - inference recomputes text pooling for every image-text pair by expanding `summary` and `text_token_states` in `PAS/model/prototype/head.py::compute_pairwise_similarity`
  - similarity matrix shape is `[N_text, N_image]`

## 3. Module by module audit

### PASModel
- File: `PAS/model/build.py`
- Claimed role by structure: runtime wrapper around CLIP plus prototype-guided retrieval head
- Actual computation: extracts image CLS projected token, extracts text pre-projection token states, delegates prototype logic to `PrototypeConditionedTextHead`
- Used: yes, this is the only active model class
- Match to role: partially
- Possible issues:
  - assumes ViT-style image token tensor and CLIP text hidden width compatible with `embed_dim`
  - forwards training and inference through different effective scoring semantics

### PrototypeBank
- File: `PAS/model/prototype/prototype_bank.py`
- Purpose implied by name: learned prototype parameter bank
- Actual computation: plain learnable matrix `prototypes [N, D]`, optionally initialized from file, optionally row-normalized at init only
- Used: yes
- Match to role: yes
- Possible issues:
  - normalization is only an initialization choice, not an invariant during training

### PrototypeContextualizer
- File: `PAS/model/prototype/contextualizer.py`
- Purpose implied by name: prototype self-contextualization
- Actual computation: single parameter-free self-attention-like update `softmax(sim(Theta, Theta)) @ Theta`, optionally residual
- Used: yes when enabled and type != `none`
- Match to role: mostly yes
- Possible issues:
  - not learnable; if users expect a trainable contextualizer, that is not what the code does

### Router
- File: `PAS/model/prototype/router.py`
- Purpose implied by name: route image evidence into prototypes
- Actual computation: dense softmax over image-prototype similarities
- Used: yes
- Match to role: yes for dense soft routing
- Possible issues:
  - no sparse or top-k routing exists
  - reported `routing_active_count` is not meaningful because dense softmax is positive everywhere

### PrototypeAggregator
- File: `PAS/model/prototype/aggregator.py`
- Purpose implied by name: aggregate routed prototypes into an image summary
- Actual computation: `Q = alpha @ prototypes`
- Used: yes
- Match to role: yes

### TokenScorer
- File: `PAS/model/prototype/token_scorer.py`
- Purpose implied by name: score text tokens against an image-conditioned query
- Actual computation: per-token dot or cosine similarity with `Q`, divided by `tau_t`
- Used: yes
- Match to role: yes

### TokenMaskBuilder
- File: `PAS/model/prototype/token_mask.py`
- Purpose implied by name: keep only valid text tokens for pooling
- Actual computation: derives valid mask from attention mask, pad ids, or EOS; then applies one of `content_only`, `content_plus_special`, or `eos_only`
- Used: yes
- Match to role: yes
- Possible issues:
  - behavior depends entirely on correct `special_token_ids`; code correctly refuses to hardcode them

### MaskedTokenPooler
- File: `PAS/model/prototype/token_pooler.py`
- Purpose implied by name: masked softmax pooling over token states
- Actual computation: masked softmax over token scores, then weighted sum of token states
- Used: yes
- Match to role: yes

### Projectors
- File: `PAS/model/prototype/projector.py`
- Purpose implied by name: map image/text branch outputs into shared retrieval space
- Actual computation: either linear or 2-layer MLP, optionally followed by L2 normalization
- Used: yes
- Match to role: yes
- Possible issues:
  - `normalize_projector_outputs` is semantically inconsistent with training because InfoNCE renormalizes regardless

### PrototypeLosses
- File: `PAS/model/prototype/losses.py`
- Purpose implied by structure: contrastive loss plus prototype regularizers
- Actual computation:
  - symmetric diagonal InfoNCE over batch row alignment
  - diversity loss on prototype cosine matrix
  - balance loss on average routing usage
- Used: yes
- Match to role: only partially
- Possible issues:
  - InfoNCE positive structure is row-only, not identity-aware
  - training loss uses a different effective text-conditioning regime than inference

### Evaluator
- File: `PAS/utils/metrics.py`
- Purpose implied by structure: compute retrieval similarity and ranking metrics
- Actual computation: caches encoded images/texts, then computes full pairwise similarity by reconditioning text on every image summary
- Used: yes
- Match to role: yes
- Possible issues:
  - the evaluation score family is not the same function optimized during training

## 4. Tensor semantics audit

### Issue A: training text embedding and inference text embedding are not the same object
- Where: training in `PAS/model/prototype/head.py::forward`; inference in `PAS/model/prototype/head.py::compute_pairwise_similarity`
- Current code:
  - training: `text_outputs = pool_text_with_summary(image_outputs['summary'], text_token_states, ...)`
  - inference: expands every image summary against every text and recomputes pooled text per pair
- Why shape-wise it works:
  - both paths produce `[*, D_out]` text embeddings and compatible dot products
- Why it is semantically wrong:
  - during training, `Z_t_i` is conditioned only on image `i`
  - during inference, `Z_t(i,j)` is conditioned on image `j`
  - the optimized scoring function and deployed scoring function are different

### Issue B: identity semantics are lost inside InfoNCE labels
- Where: `PAS/datasets/*::_process_anno`, `PAS/datasets/sampler.py::RandomIdentitySampler`, `PAS/model/prototype/losses.py::symmetric_infonce`
- Current code:
  - training dataset creates one sample per caption, preserving repeated `pid`
  - sampler explicitly selects multiple instances per identity in one batch
  - InfoNCE labels are `torch.arange(B)`
- Why shape-wise it works:
  - logits are `[B, B]`, labels are `[B]`
- Why it is semantically wrong:
  - off-diagonal same-identity pairs are real positives under retrieval semantics
  - the loss still penalizes them as negatives

### Issue C: projector normalization flag does not preserve meaning across train and infer
- Where: `PAS/model/prototype/projector.py`, `PAS/model/prototype/losses.py`, `PAS/model/prototype/head.py::compute_pairwise_similarity`
- Current code:
  - training loss always L2-normalizes both embeddings inside `symmetric_infonce`
  - inference uses the raw projector outputs as returned
- Why shape-wise it works:
  - both normalized and unnormalized vectors support dot products
- Why it is semantically wrong:
  - `normalize_projector_outputs=false` barely changes training InfoNCE
  - the same flag changes inference score magnitude and direction family
  - the config name suggests a shared train/infer behavior knob, but it is not one

### Issue D: backbone support surface is wider than the actually valid tensor assumptions
- Where: `PAS/model/clip_model.py::build_CLIP_from_openai_pretrained`, `PAS/model/build.py::extract_image_features`, `PAS/model/build.py::_resolve_text_states`
- Current code:
  - CLIP loader exposes both ViT and ResNet models
  - PAS assumes image outputs are token tensors with a CLS slot and that text pre-projection width matches `embed_dim`
- Why shape-wise it works on default ViT:
  - ViT image outputs are `[B, T, D]`
  - for ViT CLIP variants, text hidden width and `embed_dim` coincide
- Why it is semantically wrong as a public surface:
  - ResNet path returns a global image embedding, not token sequence
  - some CLIP variants have `transformer_width != embed_dim`, but PAS builds prototype adapters using `embed_dim` while consuming `pre_projection_tokens`

## 5. Mathematical correctness audit

Core equations actually implemented:

1. Prototype routing
- `alpha = softmax((sim(v, Theta_tilde)) / tau_p)`
- with cosine mode: `sim(a, b) = <a/||a||, b/||b||>`
- Algebraic status: correct and numerically stabilized by max subtraction
- Semantic status: correct for dense soft routing

2. Prototype aggregation
- `Q = alpha * Theta_tilde = sum_k alpha_k Theta_tilde_k`
- Algebraic status: correct
- Semantic status: correct for convex prototype summary because `alpha` is softmax-normalized

3. Prototype contextualization
- `W = softmax((normalize(Theta) normalize(Theta)^T) / sqrt(D))`
- `Theta_tilde = Theta + W Theta` if residual else `W Theta`
- Algebraic status: correct
- Numerical status: stable enough for small prototype counts
- Semantic status: acceptable as one-step parameter-free self-interaction

4. Token scoring
- `s_l = <q_l, h_l> / tau_t` where `q_l` is really shared summary `Q` broadcast across tokens
- Algebraic status: correct
- Semantic status: correct for image-conditioned token salience

5. Masked token pooling
- `beta = softmax(masked_logits)`
- `T_pool = sum_l beta_l h_l`
- Algebraic status: correct
- Numerical status: good; masking happens before softmax and rows are renormalized
- Semantic status: correct

6. Training contrastive loss
- `logits_ij = gamma * <normalize(Z_t_i), normalize(Z_v_j)>`
- `loss = 0.5 * [CE(logits, arange(B)) + CE(logits^T, arange(B))]`
- Algebraic status: correct
- Numerical status: reasonable; `gamma = exp(logit_scale)` clamped to max 100
- Semantic status: wrong for this training setup because:
  - positives are forced to be row-diagonal only
  - `Z_t_i` is conditioned on only `Q_i`, not on `Q_j`

7. Inference similarity
- `sim(text_i, image_j) = gamma * <Z_t(text_i ; Q_j), Z_v(image_j)>`
- Algebraic status: correct
- Semantic status: internally consistent for pairwise image-conditioned retrieval
- But it does not match the trained scoring function above

8. Diversity loss
- `L_div = sum_{a,b} (cos(Theta_a, Theta_b) - I[a=b])^2`
- Algebraic status: correct
- Numerical status: fine
- Semantic status: acceptable, but scale grows with `N_proto^2` because it is not averaged

9. Balance loss
- `L_bal = sum_k (mean_i alpha_{ik} - 1/N_proto)^2`
- Algebraic status: correct
- Semantic status: acceptable as a usage regularizer

## 6. Training versus inference consistency

Confirmed mismatch:

- Training uses one summary per row and pools each text once with its own row summary.
- Inference recomputes text pooling for every image-text pair.
- Therefore training optimizes `Z_t(text_i ; Q_i)` but inference ranks with `Z_t(text_i ; Q_j)`.

Consistent parts:

- inference reuses the same prototype bank parameters as training
- inference reuses the same router, aggregator, token scorer, token mask, token pooler, and projector families
- inference reuses the same learned `logit_scale`

Inconsistent parts:

- image-conditioned text representation is pairwise at inference but row-wise at training
- optional projector normalization is enforced in training loss even if disabled in the projector, but not enforced in inference
- standalone evaluation and training-time validation both use the evaluator path, so the mismatch affects both reported validation and final test

## 7. Config surface audit

Material config keys and actual behavior:

| Config key | Affected code | Actual behavior | Risk |
|---|---|---|---|
| `model.use_prototype_bank` | `PAS/model/build.py::_validate_configuration` | must be `true`, otherwise runtime raises | misleading as a switch; not a real branch |
| `model.use_image_conditioned_pooling` | `PAS/model/build.py::_validate_configuration` | must be `true`, otherwise runtime raises | misleading as a switch; not a real branch |
| `model.use_prototype_contextualization` | `PAS/utils/options.py`, `PAS/model/prototype/build.py` | OR-ed with `prototype.contextualization_enabled` | disabling one flag alone may do nothing |
| `prototype.contextualization_enabled` | same as above | OR-ed with `model.use_prototype_contextualization` | duplicate surface |
| `prototype.contextualization_type` | `PAS/model/prototype/contextualizer.py` | `self_attention` or `none` | real behavior switch |
| `prototype.contextualization_residual` | contextualizer | toggles residual add | real behavior switch |
| `prototype.routing_type` / `prototype.routing_temperature` | router | cosine or dot routing; softmax temperature | real behavior switch |
| `text_pooling.token_policy` | token mask builder | controls kept tokens | real behavior switch |
| `text_pooling.scoring_type` / `text_pooling.token_temperature` | token scorer | cosine or dot token scoring | real behavior switch |
| `model.normalize_projector_outputs` | projectors | changes projector outputs, but training loss renormalizes anyway | misleading train/infer semantics |
| `model.learn_logit_scale` / `model.temperature` | losses | learnable or fixed contrastive scale | real behavior switch |
| `prototype.use_diversity_loss` / `prototype.diversity_loss_weight` | losses | toggles prototype diversity penalty | real behavior switch |
| `prototype.use_balancing_loss` / `prototype.balance_loss_weight` | losses | toggles routing usage regularizer | real behavior switch |
| `model.pretrain_choice` | CLIP loader | exposes ViT and ResNet CLIP names | wider than PAS-valid runtime surface |
| `evaluation.prototype_image_chunk_size` / `prototype_text_chunk_size` | pairwise similarity | chunking only, not formula | execution-only knob |

Flags that appear to exist but do not provide independent control:

- `model.use_prototype_bank`
- `model.use_image_conditioned_pooling`
- `model.use_prototype_contextualization` and `prototype.contextualization_enabled` as separate knobs

## 8. Silent failure / suspicious patterns

- `PAS/model/prototype/router.py`: `routing_active_count` is computed as `alpha.gt(0).sum(...)`; with dense softmax this is effectively always all prototypes.
- `PAS/model/prototype/head.py::get_prototype_context`: calls bank/contextualizer with `return_debug=True` even when the caller asked for no debug.
- `PAS/datasets/build.py`: distributed identity-sampler branch creates `batch_sampler` but never constructs `train_loader`.
- `PAS/datasets/build.py`: branch for `Flickr` / `MSCOCO` exists even though those datasets are not in `__factory`.
- `PAS/model/build.py`: `num_classes` is threaded through the build path but not used by PASModel.

## 9. Confirmed issues

| ID | Severity | Location | Issue | Why it is wrong | Impact on training/inference | Suggested fix direction |
|---|---|---|---|---|---|---|
| C1 | critical | `PAS/model/prototype/head.py::forward`, `PAS/model/prototype/head.py::compute_pairwise_similarity` | Training and inference use different scoring functions | train pools each text once with its matched image summary; infer repools text for every candidate image summary | validation/test metrics do not measure the function the optimizer trained | either train with the same pairwise-conditioned scoring used at inference, or simplify inference to the row-wise text representation actually trained |
| C2 | major | `PAS/datasets/*::_process_anno`, `PAS/datasets/sampler.py::RandomIdentitySampler`, `PAS/model/prototype/losses.py::symmetric_infonce` | Same-identity positives are treated as negatives | batch sampler groups multiple instances per identity, but InfoNCE labels only diagonal rows as positive | representation learning is pushed to separate true retrieval positives inside the batch | use identity-aware multi-positive contrastive targets or a sampler/loss combination that preserves only one positive per row |
| C3 | major | `PAS/model/prototype/projector.py`, `PAS/model/prototype/losses.py`, `PAS/model/prototype/head.py::compute_pairwise_similarity` | `normalize_projector_outputs` is inconsistent between train and infer | training renormalizes embeddings regardless of projector setting; inference does not | toggling the flag changes inference semantics without equivalently changing training | either always normalize in both places or never renormalize inside the loss when the projector flag is off |
| C4 | major | `PAS/model/clip_model.py::build_CLIP_from_openai_pretrained`, `PAS/model/build.py::extract_image_features`, `PAS/model/build.py::_resolve_text_states` | Advertised CLIP backbone surface is broader than the valid PAS path | PAS assumes ViT image token tensors and text hidden width compatible with `embed_dim`; that is not true for all OpenAI CLIP backbones, especially ResNet variants | non-default backbones can crash or silently miswire dimensions | explicitly restrict PAS to supported ViT backbones or add proper handling for global-image ResNet outputs and `transformer_width != embed_dim` |
| C5 | moderate | `PAS/utils/options.py`, `PAS/model/prototype/build.py` | Contextualization disable flags are duplicated and OR-coupled | setting only one of the two flags to `false` does not actually disable contextualization if the other remains `true` | ablations and CLI overrides can silently do the wrong thing | collapse to one authoritative flag or define precedence explicitly |
| C6 | moderate | `PAS/datasets/build.py` | Distributed identity-sampler training path does not return a `train_loader` | `batch_sampler` is created but no `DataLoader` is built in the distributed branch | multi-GPU training path is broken | construct `train_loader` from `batch_sampler` before returning |
| C7 | minor | `PAS/model/prototype/router.py` | `routing_active_count` metric is meaningless | dense softmax produces positive mass for nearly every prototype | debug metric can mislead monitoring and ablation analysis | replace with entropy, top-k support count, or thresholded usage mass |

## 10. Things that look correct

- Token masking is driven by configured special-token ids rather than brittle argmax or hardcoded EOS assumptions.
- Masked token pooling applies masking before softmax and renormalizes over the kept tokens, which is the right order.
- Router softmax is stabilized by subtracting the per-row max.
- Pairwise retrieval reuses the trained prototype bank and learned logit scale rather than introducing a second inference-only bank.
- The precision guardrails around fp16 backbone/prototype training are explicit and reject unsupported combinations.
- The contextualizer is consistently treated as parameter-free and excluded from optimizer groups.
- The evaluator computes text-to-image ranking from a full similarity matrix with correct `[N_text, N_image]` orientation.

## 11. Final verdict

The implementation does not fully preserve a coherent intended model logic.

It does implement a real prototype-guided, image-conditioned text pooling architecture, and most local tensor operations are mathematically sound. But the end-to-end learning objective is not aligned with the inference-time scoring function, and the batch supervision semantics conflict with identity-based retrieval labels.

It is not safe to treat current experiment results as trustworthy for serious conclusions.

Must be fixed before serious experiments:

- train/inference scoring mismatch
- diagonal-only InfoNCE under multi-instance same-identity batches
- unsupported backbone/config surface ambiguity

## Recommended next actions

Must fix now:

- make the training score exactly match the inference score family
- stop treating same-identity cross-pairs as negatives
- narrow or repair the supported backbone/config surface

Should verify with tests:

- train/infer parity test that compares training logits against eval logits on the same batch
- multi-positive batch test with repeated `pid`
- backbone compatibility tests for every allowed `pretrain_choice`
- config override tests proving contextualization can be disabled with one authoritative flag

Safe as is:

- token masking and masked pooling mechanics
- dense routing plus weighted prototype aggregation
- basic prototype diversity and usage regularizer formulas
