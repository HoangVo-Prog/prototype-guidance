# IMPLEMENTATION_SPEC

## 1. Purpose of this file

This file is the implementation contract for the method. It is written for an autonomous coding agent such as Codex.

This file defines:

- what must be implemented
- what is fixed in minimal v1
- what is configurable
- exact tensor contracts
- exact forward behavior
- exact masking and normalization rules
- required tests, logging, and debugging checks

This file is **not** a paper Method section. It must prioritize coding clarity, reproducibility, and architectural fidelity over paper-style exposition.

---

## 2. Scope and non-goals

### 2.1 Scope

This specification covers:

- model architecture and module boundaries
- tensor shapes and tensor semantics
- forward pass contract
- masking policy
- normalization and similarity policy
- loss contract
- configuration surface
- required unit tests
- required diagnostics and debugging hooks
- ablation switches that must be implemented cleanly

### 2.2 Non-goals

This specification does **not** cover:

- paper writing
- related work
- benchmark storytelling
- submission strategy
- experiment timeline
- result interpretation in prose
- large-scale training orchestration beyond the interfaces needed for reproducible training

### 2.3 Implementation objective

The first implementation must optimize for:

1. stability
2. reproducibility
3. debuggability
4. clean ablation support
5. strict preservation of the intended research logic

It must **not** optimize for maximal novelty engineering, maximal abstraction, or premature complexity.

---

## 3. Fixed design decisions

This section defines what Codex must treat as locked in minimal v1.

### 3.1 FIXED NOW

These are fixed for minimal v1 and must not be silently changed.

| Item | Fixed value |
|---|---|
| Vision backbone | CLIP ViT-B/16 image tower |
| Text backbone | Paired CLIP text encoder |
| Visual representation used by method | Global image embedding only |
| Text representation used before pooling | Token-level hidden states from last text layer |
| Backbone regime | Frozen by default |
| Prototype bank type | One global shared trainable bank |
| Number of prototypes | 32 |
| Prototype dimension | Same as backbone feature dimension `D` |
| Prototype contextualization | On, normalized self-interaction + residual |
| Routing similarity | Cosine |
| Token scoring similarity | Cosine |
| Default token policy | Content tokens only, exclude special tokens and padding |
| Projectors | Symmetric 2-layer MLP on image and text branches |
| Projector output dimension | 256 |
| Main loss | Symmetric InfoNCE |
| Default regularizer | Prototype diversity regularization |
| Balancing loss | Off by default |
| Query representation for token scoring | Single image-conditioned prototype summary vector `Q` |
| Retrieval-time interaction claim | Lightweight, structured interaction only |
| Patch-token image interaction | Not part of v1 |

### 3.2 ABLATION ONLY

These must not replace the default path unless explicitly enabled by config.

| Item | Allowed ablation values |
|---|---|
| `num_prototypes` | 16, 32, 64 |
| `use_contextualization` | `true`, `false` |
| `contextualization_mode` | `residual`, `overwrite` |
| `prototype_init` | `normalized_random`, `sampled_image_embeddings`, `kmeans_centroids` |
| `routing_similarity` | `cosine`, `dot` |
| `token_similarity` | `cosine`, `dot` |
| `token_policy` | `content_only`, `content_plus_special`, `eos_only` |
| `pooler_variant` | `weighted_pool`, `mean_pool`, `text_only_attention`, `direct_image_conditioned_pool` |
| `use_diversity_loss` | `true`, `false` |
| `use_balancing_loss` | `true`, `false` |
| `freeze_vision_backbone` | `true`, `false` |
| `freeze_text_backbone` | `true`, `false` |
| `projector_type` | `mlp2`, `linear` |
| `tau_p`, `tau_t` | sensitivity sweeps only |

### 3.3 OPTIONAL LATER

These are explicitly outside minimal v1. Do not implement unless requested later.

- multi-head prototype summaries
- per-head or hierarchical prototype banks
- sparsemax, entmax, or top-k routing
- learned bilinear routing or token scorers
- balancing loss as part of the default recipe
- patch-token visual evidence modeling
- heavy cross-attention modules
- automatic prototype reinitialization during training
- extra semantic reasoning modules
- retrieval-time pairwise fusion

---

## 4. System-level module graph

Minimal v1 module graph:

```text
ImageEncoderWrapper
    -> V

TextEncoderWrapper
    -> H
    -> token_valid_mask
    -> token_keep_mask

PrototypeBank
    -> Theta_v

PrototypeContextualizer(Theta_v)
    -> Theta_tilde

Router(V, Theta_tilde)
    -> alpha

PrototypeAggregator(alpha, Theta_tilde)
    -> Q

TokenScorer(Q, H)
    -> S_t

MaskedTokenPooler(S_t, H, token_keep_mask)
    -> beta
    -> T_pool

ImageProjector(V)
    -> Z_v

TextProjector(T_pool)
    -> Z_t

LossComputer(Z_v, Z_t, Theta_v, alpha)
    -> logits
    -> L_infonce
    -> L_div
    -> L_bal(optional)
    -> L_total
```

### 4.1 Dependency rules

- `Theta_tilde` depends only on `Theta_v`.
- `alpha` depends on `V` and `Theta_tilde`.
- `Q` depends on `alpha` and `Theta_tilde`.
- `S_t` depends on `Q` and `H`.
- `beta` depends on `S_t` and token masks.
- `T_pool` depends on `beta` and `H`.
- `Z_v` depends only on `V`.
- `Z_t` depends only on `T_pool`.
- `L_div` depends on `Theta_v`.
- `L_bal` depends on `alpha` and is off by default.
- `L_infonce` depends on normalized `Z_v` and `Z_t`.

### 4.2 Architectural intent

The image affects the final text representation through:

`V -> alpha -> Q -> token scores -> beta -> T_pool`

This is the main mechanism. Codex must preserve this causal path.

---

## 5. Tensor contract

All tensors are batch-first. All softmax dimensions must be explicit in code.

### 5.1 Core tensors

| Name | Meaning | Shape | Dtype | Trainable or derived | Normalization expectation | Masking applies |
|---|---|---:|---|---|---|---|
| `X` | Input image batch | `[B, C, H_img, W_img]` | float | input | backbone preprocessing only | no |
| `T` | Token ids | `[B, L]` | int64 | input | none | yes, via masks |
| `M` | Input attention mask from tokenizer | `[B, L]` | bool or int64 | input | values in `{0,1}` | yes |
| `V` | Global image embedding from image encoder | `[B, D]` | float32/float16/bfloat16 | derived | normalize only at similarity sites | no |
| `H` | Token-level text hidden states | `[B, L, D]` | float32/float16/bfloat16 | derived | normalize only at similarity sites | yes |
| `Theta_v` | Global trainable prototype bank | `[N, D]` | float32 | trainable parameter | normalize only at similarity sites and init if configured | no |
| `Theta_norm` | Row-wise normalized prototype bank for self-interaction | `[N, D]` | float | derived | L2 over `D` | no |
| `S_p` | Prototype self-similarity matrix | `[N, N]` | float | derived | from normalized prototypes | no |
| `P_p` | Prototype self-attention weights | `[N, N]` | float | derived | row-wise softmax over prototypes | no |
| `Theta_tilde` | Contextualized prototype bank | `[N, D]` | float | derived | raw tensor, not permanently normalized | no |
| `V_norm` | Normalized image embeddings for routing | `[B, D]` | float | derived | L2 over `D` | no |
| `Theta_tilde_norm` | Normalized contextualized prototypes for routing | `[N, D]` | float | derived | L2 over `D` | no |
| `R` | Routing similarity logits before temperature | `[B, N]` | float | derived | cosine by default | no |
| `alpha_logits` | Temperature-scaled routing logits | `[B, N]` | float | derived | `R / tau_p` | no |
| `alpha` | Routing weights over prototypes | `[B, N]` | float | derived | softmax over `N` | no |
| `Q` | Image-conditioned prototype summary vector | `[B, D]` | float | derived | raw tensor, normalize only for token scoring | no |
| `Q_norm` | Normalized summary vector for token scoring | `[B, D]` | float | derived | L2 over `D` | no |
| `H_norm` | Normalized token features for token scoring | `[B, L, D]` | float | derived | L2 over `D` | yes |
| `S_t` | Token similarity logits before temperature | `[B, L]` | float | derived | cosine by default | yes |
| `S_t_scaled` | Temperature-scaled token logits | `[B, L]` | float | derived | `S_t / tau_t` | yes |
| `token_valid_mask` | True for non-padding tokens | `[B, L]` | bool | derived from `M` | none | yes |
| `token_keep_mask` | True for tokens allowed by current token policy | `[B, L]` | bool | derived | none | yes |
| `beta_logits_masked` | Token logits after invalid positions set to `-inf` | `[B, L]` | float | derived | explicit masking before softmax | yes |
| `beta` | Token pooling weights | `[B, L]` | float | derived | softmax over valid kept tokens only | yes |
| `T_pool` | Weighted pooled text representation | `[B, D]` | float | derived | raw tensor, projector input | no |
| `Z_t_raw` | Text projector output before normalization | `[B, d_out]` | float | derived | none | no |
| `Z_v_raw` | Image projector output before normalization | `[B, d_out]` | float | derived | none | no |
| `Z_t` | Normalized text projector output | `[B, d_out]` | float | derived | L2 over feature dim | no |
| `Z_v` | Normalized image projector output | `[B, d_out]` | float | derived | L2 over feature dim | no |
| `logit_scale` | Learnable contrastive logit scale parameter | `[]` or `[1]` | float32 | trainable parameter | exponentiated at use time | no |
| `similarity_logits` | Contrastive similarity matrix | `[B, B]` | float | derived | `exp(logit_scale) * Z_t @ Z_v.T` | no |
| `L_infonce` | Symmetric InfoNCE loss | `[]` | float | derived | finite scalar | no |
| `L_div` | Diversity regularization | `[]` | float | derived | finite scalar | no |
| `L_bal` | Optional balancing loss | `[]` | float | derived | finite scalar | no |
| `L_total` | Total loss | `[]` | float | derived | finite scalar | no |

### 5.2 Derived constants

| Name | Meaning |
|---|---|
| `B` | Batch size |
| `L` | Padded text length |
| `D` | Backbone feature dimension, e.g. CLIP hidden width / embedding width |
| `N` | Number of prototypes |
| `d_out` | Projector output dimension, default `256` |

### 5.3 Dtype rules

- `Theta_v` must be stored in parameter dtype, usually `float32`.
- Similarity computations may happen under AMP, but masking and normalization must remain numerically safe.
- Softmax inputs must be floating point.
- Token ids must be `int64`.
- Boolean masks should use `bool` where practical.

### 5.4 Mask contract

- `token_valid_mask` excludes padding only.
- `token_keep_mask` excludes padding plus disallowed token categories under current token policy.
- `beta` must be exactly zero on invalid positions after masking and softmax.
- No masked position may contribute to `T_pool`.

---

## 6. Forward pass contract

The forward pass must follow the sequence below.

### 6.1 Required inputs

- `pixel_values`: preprocessed images `[B, C, H_img, W_img]`
- `input_ids`: token ids `[B, L]`
- `attention_mask`: tokenizer attention mask `[B, L]`
- optional metadata for diagnostics:
  - raw tokens or decoded text
  - token type indicators if precomputed

### 6.2 Forward pass pseudocode

```python
def forward(pixel_values, input_ids, attention_mask, token_metadata=None):
    # 1. Encode image batch into a single global vector per image
    V = image_encoder(pixel_values)                  # [B, D]

    # 2. Encode text batch into token-level hidden states
    H = text_encoder(input_ids, attention_mask)      # [B, L, D]

    # 3. Build token masks from padding mask and token policy
    token_valid_mask = build_valid_mask(input_ids, attention_mask)     # [B, L], bool
    token_keep_mask = build_keep_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_policy=config.token_policy,
        tokenizer_info=tokenizer_info
    )                                                # [B, L], bool

    # 4. Contextualize the shared prototype bank
    Theta_v = prototype_bank()                       # [N, D]
    if config.use_contextualization:
        Theta_tilde = prototype_contextualizer(Theta_v)   # [N, D]
    else:
        Theta_tilde = Theta_v

    # 5. Route global image embeddings over contextualized prototypes
    alpha = router(V, Theta_tilde, tau_p=config.tau_p)    # [B, N]

    # 6. Aggregate contextualized prototypes into one summary vector per image
    Q = prototype_aggregator(alpha, Theta_tilde)     # [B, D]

    # 7. Score each token against the image-conditioned summary
    S_t = token_scorer(Q, H, tau_t=config.tau_t)     # [B, L]

    # 8. Mask invalid token logits with -inf before softmax
    beta, beta_logits_masked = masked_token_pooler.compute_weights(
        logits=S_t,
        keep_mask=token_keep_mask
    )                                                # [B, L], [B, L]

    # 9. Weighted token pooling
    T_pool = masked_token_pooler.pool(H, beta)       # [B, D]

    # 10. Projection heads
    Z_v_raw = image_projector(V)                     # [B, d_out]
    Z_t_raw = text_projector(T_pool)                 # [B, d_out]

    # 11. L2 normalize projector outputs
    Z_v = l2_normalize(Z_v_raw, dim=-1)              # [B, d_out]
    Z_t = l2_normalize(Z_t_raw, dim=-1)              # [B, d_out]

    # 12. Contrastive logits
    similarity_logits = contrastive_logits(
        Z_t=Z_t,
        Z_v=Z_v,
        logit_scale=logit_scale
    )                                                # [B, B]

    # 13. Losses
    L_infonce = symmetric_infonce(similarity_logits)
    L_div = diversity_loss(Theta_v) if config.use_diversity_loss else zero_scalar()
    L_bal = balancing_loss(alpha) if config.use_balancing_loss else zero_scalar()

    L_total = L_infonce \
              + config.lambda_div * L_div \
              + config.lambda_bal * L_bal

    # 14. Return loss and diagnostics
    return {
        "loss": L_total,
        "loss_infonce": L_infonce,
        "loss_div": L_div,
        "loss_bal": L_bal,
        "V": V,
        "H": H,
        "Theta_v": Theta_v,
        "Theta_tilde": Theta_tilde,
        "alpha": alpha,
        "Q": Q,
        "S_t": S_t,
        "beta": beta,
        "T_pool": T_pool,
        "Z_t": Z_t,
        "Z_v": Z_v,
        "similarity_logits": similarity_logits,
        "token_valid_mask": token_valid_mask,
        "token_keep_mask": token_keep_mask,
        "beta_logits_masked": beta_logits_masked,
    }
```

### 6.3 Critical constraints

1. The image encoder must output one vector per image. No patch-token interaction in v1.
2. The text encoder must output token-level hidden states, not only pooled CLIP text output.
3. Token masking must happen **before** token softmax.
4. Masked token logits must be set to `-inf`, not zero, before softmax.
5. `beta` must sum to 1 over valid kept tokens.
6. `alpha` must sum to 1 over prototypes.
7. Normalization must happen at similarity computation sites, not by permanently overwriting raw feature tensors.
8. Codex must not replace cosine with raw dot product unless the config explicitly requests it.
9. Gradient must reach `Theta_v` through the default path.

---

## 7. Per-module specification

### 7.1 Image encoder wrapper

#### Purpose
Wrap the CLIP image tower and expose a single global image embedding per image.

#### Inputs
- `pixel_values`: `[B, C, H_img, W_img]`

#### Outputs
- `V`: `[B, D]`

#### Exact behavior
- Use the pretrained CLIP ViT-B/16 image encoder.
- Return the global image embedding used by CLIP for image-text alignment.
- Do not return patch tokens in minimal v1.
- Support frozen default behavior.
- If partial unfreezing is later enabled, do not change the interface.

#### Must not change
- Do not replace CLIP ViT-B/16 in default path.
- Do not add patch-token branches in v1.
- Do not add extra MLPs before the prototype module.

#### Implementation notes
- Prefer using a wrapper around a stable library implementation, such as OpenCLIP or Hugging Face, but keep the returned interface simple.
- If the library returns projected and unprojected features, document and standardize which one is used. Default: use the backbone-side global feature aligned with the CLIP embedding space.
- Expose `feature_dim`.

#### Likely failure cases
- Wrong feature dimension because of mixing hidden width and projected width
- Using patch tokens by accident
- Accidentally detaching `V`
- Incorrect freeze behavior

---

### 7.2 Text encoder wrapper

#### Purpose
Wrap the paired CLIP text encoder and expose token-level hidden states.

#### Inputs
- `input_ids`: `[B, L]`
- `attention_mask`: `[B, L]`

#### Outputs
- `H`: `[B, L, D]`

#### Exact behavior
- Use the paired CLIP text encoder.
- Return last-layer token hidden states for all positions.
- Do not pool internally for the method path.
- Respect tokenizer padding and truncation behavior.

#### Must not change
- Do not return only EOS or CLS in the default path.
- Do not apply query conditioning inside the text encoder.
- Do not add cross-attention.

#### Implementation notes
- The wrapper must expose enough tokenizer metadata to construct special-token masks.
- If CLIP tokenizer does not define `CLS`, keep the token policy implementation generic and backbone-aware.
- Token policy must be handled outside this encoder wrapper.

#### Likely failure cases
- Returning only pooled text output
- Misaligned hidden dimension relative to image encoder
- Missing special-token ids
- Wrong padding behavior

---

### 7.3 Prototype bank

#### Purpose
Store a single global trainable bank of reusable visual prototypes.

#### Inputs
- none at forward time beyond module state

#### Outputs
- `Theta_v`: `[N, D]`

#### Exact behavior
- Implement as a trainable parameter of shape `[num_prototypes, feature_dim]`.
- Default initialization: row-normalized random initialization.
- Optional ablations: sampled image embeddings, k-means centroids.
- Bank is shared across all batches and all samples.
- No per-sample bank.
- No multi-bank logic.

#### Must not change
- No semantic labeling of prototypes in code.
- No dynamic creation or deletion of prototypes during standard training.
- No hidden projection inside the bank.

#### Implementation notes
- Provide initialization utilities separately from the module itself.
- Keep bank parameter easy to inspect and log.
- A dedicated `reset_parameters(init_mode=...)` method is recommended.

#### Likely failure cases
- Wrong dimension
- Bank accidentally re-created per batch
- Non-normalized init causing unstable early routing
- Silent use of sample-specific prototypes

---

### 7.4 Prototype contextualizer

#### Purpose
Allow lightweight interaction among prototypes before routing.

#### Inputs
- `Theta_v`: `[N, D]`

#### Outputs
- `Theta_tilde`: `[N, D]`

#### Exact behavior
Default residual contextualization:

1. `Theta_norm = l2_normalize(Theta_v, dim=-1)`
2. `S_p = Theta_norm @ Theta_norm.T` -> `[N, N]`
3. `A_p = S_p / sqrt(D)`
4. `P_p = softmax(A_p, dim=-1)` -> row-wise over prototypes
5. If mode is `residual`:
   - `Theta_tilde = Theta_v + P_p @ Theta_v`
6. If mode is `overwrite`:
   - `Theta_tilde = P_p @ Theta_v`
7. If contextualization is disabled:
   - `Theta_tilde = Theta_v`

#### Must not change
- Default mode is `residual`.
- Default similarity basis is normalized dot product, equivalent to cosine similarity after L2 normalization.
- Default softmax dimension is `-1`, over prototypes.
- No learned attention matrices in v1.

#### Implementation notes
- Keep self-interaction parameter-free in spirit.
- Use `sqrt(D)` scaling explicitly.
- Do not normalize `Theta_tilde` in-place for future stages. Normalize on demand.

#### Likely failure cases
- Wrong softmax axis
- Using raw dot product when cosine is expected
- Forgetting residual in default mode
- Numerical issues from mixed precision if not careful

---

### 7.5 Router

#### Purpose
Compute image-conditioned soft routing over contextualized prototypes.

#### Inputs
- `V`: `[B, D]`
- `Theta_tilde`: `[N, D]`
- `tau_p`: scalar

#### Outputs
- `alpha`: `[B, N]`

#### Exact behavior
Default routing:

1. `V_norm = l2_normalize(V, dim=-1)`
2. `Theta_tilde_norm = l2_normalize(Theta_tilde, dim=-1)`
3. `R = V_norm @ Theta_tilde_norm.T` -> `[B, N]`
4. `alpha_logits = R / tau_p`
5. `alpha = softmax(alpha_logits, dim=-1)`

If ablation `routing_similarity == "dot"` is enabled:

- use `R = V @ Theta_tilde.T`
- still apply `softmax(R / tau_p, dim=-1)`

#### Must not change
- Default routing similarity is cosine.
- Routing is dense soft routing.
- No sparse selection, top-k, thresholding, or hard assignment in v1.

#### Implementation notes
- `tau_p` should be fixed by default, not learnable.
- Expose entropy diagnostics.
- Use explicit matrix multiply.

#### Likely failure cases
- Wrong temperature direction
- Wrong softmax axis
- Accidental use of `Theta_v` instead of `Theta_tilde`
- Uniform routing from too large a temperature
- Near one-hot routing from too small a temperature

---

### 7.6 Prototype aggregator

#### Purpose
Aggregate contextualized prototypes into one summary vector per image.

#### Inputs
- `alpha`: `[B, N]`
- `Theta_tilde`: `[N, D]`

#### Outputs
- `Q`: `[B, D]`

#### Exact behavior
- Compute `Q = alpha @ Theta_tilde`
- This is a weighted sum over prototypes for each sample.

#### Must not change
- Default summary is one vector per image.
- No multi-head summary.
- No concatenation with image embedding in v1.

#### Implementation notes
- Keep this as a separate module or function to isolate testing.
- Return `Q` only, not extra hidden states.

#### Likely failure cases
- Shape mismatch
- Using `Theta_v` instead of `Theta_tilde`
- Wrong batch broadcasting

---

### 7.7 Token scorer

#### Purpose
Score each text token using the image-conditioned summary vector.

#### Inputs
- `Q`: `[B, D]`
- `H`: `[B, L, D]`
- `tau_t`: scalar

#### Outputs
- `S_t`: `[B, L]`

#### Exact behavior
Default scoring:

1. `Q_norm = l2_normalize(Q, dim=-1)` -> `[B, D]`
2. `H_norm = l2_normalize(H, dim=-1)` -> `[B, L, D]`
3. Compute cosine scores:
   - `S_t = einsum("bd,bld->bl", Q_norm, H_norm)`
4. Temperature-scale:
   - `S_t = S_t / tau_t`

If ablation `token_similarity == "dot"` is enabled:

- use `S_t = einsum("bd,bld->bl", Q, H) / tau_t`

#### Must not change
- Default similarity is cosine.
- Output shape must remain `[B, L]`.
- Token scoring is one score per token, not per token-prototype pair.

#### Implementation notes
- Use `einsum` or equivalent batched dot product for clarity.
- Keep temperature scaling outside similarity if that makes logging easier.

#### Likely failure cases
- Wrong dimension reduction
- Missing normalization
- Mixing up `[B, L, D]` and `[B, D, L]`
- Special tokens dominating due to missing mask or raw dot product

---

### 7.8 Masked token pooler

#### Purpose
Apply token policy, compute token weights, and pool text features.

#### Inputs
- `logits`: `[B, L]`
- `H`: `[B, L, D]`
- `keep_mask`: `[B, L]`

#### Outputs
- `beta`: `[B, L]`
- `T_pool`: `[B, D]`

#### Exact behavior
1. Start from token logits `[B, L]`.
2. Set all positions where `keep_mask == False` to `-inf`.
3. Apply `softmax(masked_logits, dim=-1)` to get `beta`.
4. Replace any numerically problematic rows only if needed with a safe fallback.
   - Minimal v1 assumption: each row has at least one valid kept token.
5. Force invalid positions in `beta` to zero after softmax if needed for safety.
6. Weighted pooling:
   - `T_pool = einsum("bl,bld->bd", beta, H)`

#### Must not change
- Masked logits must use `-inf` before softmax.
- Invalid positions must get zero final beta.
- Default token policy excludes padding and special tokens.
- No averaging over invalid positions.

#### Implementation notes
- Keep mask building separate from pooling.
- Validate that every sample has at least one kept token under current token policy.
- If a dataset example becomes empty under a pathological token policy, raise a clear error.

#### Likely failure cases
- Masked logits set to zero instead of `-inf`
- Wrong softmax axis
- Non-zero beta on invalid tokens
- All-invalid row causing NaNs
- Accidentally using `token_valid_mask` instead of `token_keep_mask`

---

### 7.9 Projection heads

#### Purpose
Project image and pooled text into the contrastive space.

#### Inputs
- `V`: `[B, D]`
- `T_pool`: `[B, D]`

#### Outputs
- `Z_v_raw`: `[B, d_out]`
- `Z_t_raw`: `[B, d_out]`
- normalized outputs handled outside or inside consistently

#### Exact behavior
Default projector type `mlp2`:

For each branch:
- Linear(`D -> D`)
- GELU
- Dropout(optional, default `0.0`)
- Linear(`D -> d_out`)

Default projector type is symmetric:
- image and text branches use the same architecture, but separate parameters

Normalization:
- final projector outputs must be L2 normalized before InfoNCE

#### Must not change
- Default is a 2-layer MLP, not linear.
- Output dimension is fixed to `256` in minimal v1.
- No hidden asymmetry between image and text projectors in default path.

#### Implementation notes
- Keep projectors independently testable.
- Name modules clearly, e.g. `image_projector` and `text_projector`.

#### Likely failure cases
- Forgetting normalization before InfoNCE
- Sharing projector weights unintentionally
- Wrong output dimension
- Adding unnecessary extra layers

---

### 7.10 Loss module

#### Purpose
Compute InfoNCE and regularization losses.

#### Inputs
- `Z_t`: `[B, d_out]`
- `Z_v`: `[B, d_out]`
- `Theta_v`: `[N, D]`
- `alpha`: `[B, N]`
- config coefficients

#### Outputs
- `similarity_logits`: `[B, B]`
- `L_infonce`
- `L_div`
- `L_bal`
- `L_total`

#### Exact behavior

##### InfoNCE
1. Compute `scale = exp(logit_scale)`
2. Compute `similarity_logits = scale * (Z_t @ Z_v.T)` -> `[B, B]`
3. Use matched diagonal as positives.
4. Compute text-to-image cross-entropy.
5. Compute image-to-text cross-entropy.
6. Return average.

##### Diversity loss
Default form:
1. `Theta_norm = l2_normalize(Theta_v, dim=-1)`
2. `G = Theta_norm @ Theta_norm.T`
3. `I = identity_matrix(N)`
4. `L_div = ||G - I||_F^2`

##### Balancing loss
Optional only:
1. `usage = alpha.mean(dim=0)` -> `[N]`
2. `target = full_like(usage, 1 / N)`
3. `L_bal = ((usage - target) ** 2).sum()`

##### Total loss
- `L_total = L_infonce + lambda_div * L_div + lambda_bal * L_bal`

#### Must not change
- InfoNCE is the main objective.
- Diversity loss is on by default in minimal v1.
- Balancing loss is off by default.
- Contrastive similarity uses normalized projector outputs.

#### Implementation notes
- Clamp or bound `logit_scale` if needed for stability, but do so explicitly and document the bound.
- Return separate loss components for logging.

#### Likely failure cases
- Wrong label construction for InfoNCE
- Using unnormalized projector outputs
- Forgetting to multiply regularizers by their coefficients outside or inside consistently
- Exploding logit scale
- Diversity loss dominating training

---

## 8. Masking and token policy

This section is strict. Codex must implement it exactly.

### 8.1 Base masking rules

- Padded tokens must always be masked out.
- Special tokens are excluded by default.
- Token policy must be config-driven.
- EOS-only and include-special variants are ablations only.
- Masked logits must use `-inf` before softmax.
- Invalid tokens must receive zero final `beta`.

### 8.2 Required masks

Implement two explicit masks:

1. `token_valid_mask`
   - true for non-padding tokens only
   - derived from tokenizer attention mask

2. `token_keep_mask`
   - true for positions allowed under current token policy
   - must always be a subset of `token_valid_mask`

### 8.3 Token policy options

#### `content_only` (default)
Keep:
- non-padding tokens
- non-special tokens

Exclude:
- padding
- start token
- end token
- any tokenizer-defined special token

#### `content_plus_special`
Keep:
- all valid tokens, including special tokens

Exclude:
- padding only

#### `eos_only`
Keep:
- EOS token position only for each sample

Exclude:
- all other positions

Rules for `eos_only`:
- implementation must identify the actual EOS position robustly
- if multiple EOS-like markers appear, use the final valid EOS under tokenizer semantics
- if EOS cannot be found, raise a clear error rather than silently falling back

### 8.4 Softmax masking rule

For token weights:
- `masked_logits = logits.masked_fill(~token_keep_mask, -inf)`
- `beta = softmax(masked_logits, dim=-1)`

### 8.5 Post-softmax safety rule

After softmax:
- optionally zero invalid positions again:
  - `beta = beta * token_keep_mask.float()`
- optionally renormalize over valid positions only if required for numerical safety

Minimal v1 preferred behavior:
- use correct `-inf` masking so re-normalization is normally unnecessary
- still validate that row sums are 1 over kept tokens up to tolerance

### 8.6 Empty-row policy

The implementation may assume that the default tokenizer and `content_only` policy leave at least one valid token per sample.

Still, implement a defensive check:
- if any sample has zero kept tokens, raise a `ValueError` with sample indices and token policy name

---

## 9. Similarity and normalization policy

This section is critical. Codex must follow it exactly.

### 9.1 General rule

Default similarity everywhere is cosine-like similarity implemented as normalized dot product, except where an ablation explicitly switches to raw dot product.

Codex must not silently swap cosine for dot product.

### 9.2 Prototype self-interaction

Default:
- row-normalize `Theta_v`
- compute `S_p = Theta_norm @ Theta_norm.T`
- scale by `1 / sqrt(D)`
- softmax row-wise over prototypes

This is the default prototype contextualization rule.

### 9.3 Routing similarity

Default:
- normalize `V`
- normalize `Theta_tilde`
- compute `R = V_norm @ Theta_tilde_norm.T`

This is cosine similarity.

### 9.4 Token scoring similarity

Default:
- normalize `Q`
- normalize `H`
- compute per-token scores by batched dot product

This is cosine similarity.

### 9.5 InfoNCE similarity

Default:
- normalize projector outputs `Z_t_raw`, `Z_v_raw`
- compute `similarity_logits = exp(logit_scale) * (Z_t @ Z_v.T)`

### 9.6 Where normalization must happen

| Stage | Normalize what | Default |
|---|---|---|
| Prototype self-interaction | `Theta_v` rows | yes |
| Routing | `V`, `Theta_tilde` | yes |
| Token scoring | `Q`, `H` | yes |
| InfoNCE | `Z_t_raw`, `Z_v_raw` | yes |

### 9.7 Where normalization must not be silently baked in

- Do not permanently overwrite `Theta_v` with normalized values.
- Do not permanently overwrite `V`, `H`, or `Q`.
- Normalize only for the relevant similarity computation, unless a later explicit refactor is requested.

### 9.8 Temperature policy

Default:
- `tau_p`: fixed scalar, default `0.07`
- `tau_t`: fixed scalar, default `0.07`
- `logit_scale`: learnable CLIP-style logit scale

### 9.9 Logit scale policy

Recommended:
- store `logit_scale` as a trainable scalar parameter
- initialize so `exp(logit_scale)` approximates `1 / 0.07`
- clamp if necessary, but do so explicitly

---

## 10. Loss contract

### 10.1 Default-on losses

#### InfoNCE
Main loss. Always on.

Expected inputs:
- `Z_t`, `Z_v`

Expected output:
- scalar `L_infonce`

#### Diversity loss
Default regularizer. On by default in minimal v1.

Expected inputs:
- `Theta_v`

Expected output:
- scalar `L_div`

### 10.2 Default-off losses

#### Balancing loss
Off by default. Ablation only unless explicitly enabled.

Expected inputs:
- `alpha`

Expected output:
- scalar `L_bal`

### 10.3 Total loss form

Default:
```python
L_total = L_infonce + lambda_div * L_div
```

If balancing loss is enabled:
```python
L_total = L_infonce + lambda_div * L_div + lambda_bal * L_bal
```

### 10.4 Required defaults

| Loss component | Default | Notes |
|---|---|---|
| `L_infonce` | on | main objective |
| `L_div` | on | default regularizer |
| `L_bal` | off | ablation only |

### 10.5 Default coefficients

| Key | Default |
|---|---:|
| `lambda_div` | `1e-2` |
| `lambda_bal` | `0.0` |

### 10.6 Loss implementation rules

- Return all loss components separately.
- Keep loss computation deterministic given seed and batch.
- No hidden detach in the default path.
- All losses must remain differentiable with respect to intended trainable parameters.

---

## 11. Config contract

The implementation must expose a clear config surface. Config-dependent branches must be explicit and readable.

### 11.1 Model config

| Key | Type | Allowed values | Default | Research impact |
|---|---|---|---|---|
| `vision_backbone` | str | `{"clip_vit_b16"}` in minimal v1 | `"clip_vit_b16"` | fixed claim |
| `text_backbone` | str | `{"paired_clip_text"}` in minimal v1 | `"paired_clip_text"` | fixed claim |
| `feature_dim` | int | backbone-derived | derived | fixed by backbone |
| `freeze_vision_backbone` | bool | `true`, `false` | `true` | ablation |
| `freeze_text_backbone` | bool | `true`, `false` | `true` | ablation |
| `use_global_image_embedding_only` | bool | `true` | `true` | fixed claim |

### 11.2 Prototype bank config

| Key | Type | Allowed values | Default | Research impact |
|---|---|---|---|---|
| `num_prototypes` | int | positive int | `32` | ablation if changed |
| `prototype_dim` | int | must equal `feature_dim` in v1 | derived | fixed claim |
| `prototype_init` | str | `{"normalized_random", "sampled_image_embeddings", "kmeans_centroids"}` | `"normalized_random"` | ablation |
| `prototype_init_scale` | float | positive float | `0.02` | safe engineering |
| `use_contextualization` | bool | `true`, `false` | `true` | ablation |
| `contextualization_mode` | str | `{"residual", "overwrite"}` | `"residual"` | ablation |
| `contextualization_scale_by_sqrt_d` | bool | `true`, `false` | `true` | fixed default |

### 11.3 Similarity config

| Key | Type | Allowed values | Default | Research impact |
|---|---|---|---|---|
| `routing_similarity` | str | `{"cosine", "dot"}` | `"cosine"` | ablation |
| `token_similarity` | str | `{"cosine", "dot"}` | `"cosine"` | ablation |
| `infonce_similarity` | str | `{"cosine"}` in v1 | `"cosine"` | fixed claim |
| `normalize_for_self_interaction` | bool | `true`, `false` | `true` | fixed default |
| `normalize_for_routing` | bool | `true`, `false` | `true` | fixed default |
| `normalize_for_token_scoring` | bool | `true`, `false` | `true` | fixed default |
| `normalize_projector_outputs` | bool | `true`, `false` | `true` | fixed claim |
| `tau_p` | float | positive float | `0.07` | sensitivity ablation |
| `tau_t` | float | positive float | `0.07` | sensitivity ablation |
| `learn_logit_scale` | bool | `true`, `false` | `true` | fixed default |
| `logit_scale_init` | float | positive float | `log(1/0.07)` if stored in log-space | safe engineering |
| `logit_scale_max` | float | positive float | implementation-defined explicit bound | safe engineering |

### 11.4 Token policy config

| Key | Type | Allowed values | Default | Research impact |
|---|---|---|---|---|
| `token_policy` | str | `{"content_only", "content_plus_special", "eos_only"}` | `"content_only"` | high |
| `special_token_ids` | dict[str, int or list[int]] | tokenizer-dependent | derived | required |
| `error_on_empty_kept_tokens` | bool | `true`, `false` | `true` | safety |

### 11.5 Projector config

| Key | Type | Allowed values | Default | Research impact |
|---|---|---|---|---|
| `projector_type` | str | `{"mlp2", "linear"}` | `"mlp2"` | ablation |
| `projector_hidden_dim` | int | positive int | `feature_dim` | moderate |
| `projector_output_dim` | int | positive int | `256` | fixed now |
| `projector_dropout` | float | `0.0` to `<1.0` | `0.0` | safe engineering |
| `projector_activation` | str | `{"gelu"}` in minimal v1 | `"gelu"` | fixed now |

### 11.6 Loss config

| Key | Type | Allowed values | Default | Research impact |
|---|---|---|---|---|
| `use_diversity_loss` | bool | `true`, `false` | `true` | ablation |
| `lambda_div` | float | non-negative | `1e-2` | sensitivity |
| `use_balancing_loss` | bool | `true`, `false` | `false` | ablation |
| `lambda_bal` | float | non-negative | `0.0` | ablation |
| `balance_target` | str | `{"uniform"}` in v1 | `"uniform"` | fixed if enabled |

### 11.7 Training config

| Key | Type | Allowed values | Default | Research impact |
|---|---|---|---|---|
| `optimizer` | str | `{"adamw"}` in v1 | `"adamw"` | safe engineering |
| `lr_backbone` | float | non-negative | `0.0` when frozen | ablation if unfrozen |
| `lr_prototypes` | float | positive | `1e-3` | important |
| `lr_projectors` | float | positive | `1e-3` | important |
| `lr_logit_scale` | float | positive | `1e-4` | moderate |
| `weight_decay_backbone` | float | non-negative | `0.0` if frozen else explicit | engineering |
| `weight_decay_prototypes` | float | non-negative | `1e-2` | engineering |
| `weight_decay_projectors` | float | non-negative | `5e-2` | engineering |
| `grad_clip_norm` | float | positive | `1.0` | engineering |
| `amp_dtype` | str | `{"fp16", "bf16", "fp32"}` | environment-dependent | engineering |
| `seed` | int | integer | user-specified | reproducibility |

### 11.8 Debugging and logging config

| Key | Type | Allowed values | Default | Research impact |
|---|---|---|---|---|
| `log_alpha_entropy` | bool | `true`, `false` | `true` | required |
| `log_beta_entropy` | bool | `true`, `false` | `true` | required |
| `log_prototype_usage` | bool | `true`, `false` | `true` | required |
| `log_pairwise_prototype_cosine` | bool | `true`, `false` | `true` | required |
| `log_top_tokens` | bool | `true`, `false` | `true` | required |
| `log_tensor_norms` | bool | `true`, `false` | `true` | required |
| `detect_dead_prototypes` | bool | `true`, `false` | `true` | required |
| `dead_prototype_threshold` | float | non-negative | `0.005` | engineering |

### 11.9 Config implementation rules

- Every config field must have a documented default.
- Config branches must be localized. Avoid spread-out if-statements that make behavior hard to track.
- Config values that alter the research claim must be clearly named and easy to diff in experiment logs.

---

## 12. Implementation constraints for Codex

This section contains hard engineering rules.

1. No hidden architectural changes.
   - Do not add modules, skip connections, or projections that are not specified.

2. No in-place ops on tensors that require grad.
   - Especially avoid unsafe in-place masking or normalization on tensors used later.

3. All softmax dimensions must be explicit.
   - Never rely on implicit defaults.

4. All masking must be explicit.
   - Use named masks and named masked logits.

5. All config-dependent branches must be isolated and readable.
   - Default path must remain easy to inspect.

6. Avoid unnecessary abstraction.
   - Prefer a small number of clearly named modules over over-generalized meta-framework code.

7. Prioritize debuggability over cleverness.
   - Intermediate tensors should be easy to return and inspect.

8. Keep each module independently testable.
   - Shape tests and behavior tests must be possible per module.

9. Preserve differentiability.
   - No hard routing, no hard token selection, no hidden detach in default path.

10. Do not mix ablation logic into the default path messily.
    - Ablations should be switchable without rewriting core code.

11. Use numerically safe normalization.
    - Add epsilon where needed and keep it explicit.

12. Keep tokenizer-dependent logic centralized.
    - Special-token handling must not be scattered across the codebase.

13. Keep initialization utilities separate.
    - Prototype initialization from sampled embeddings or k-means should be utility functions, not hidden inside training code.

14. Keep the forward return rich during development.
    - Minimal training may consume only loss, but debug mode must expose intermediate tensors.

15. Do not silently fall back on alternative behavior when an assumption breaks.
    - Raise clear errors for missing EOS, empty kept-token rows, wrong backbone dims, or invalid config.

---

## 13. Required unit tests

These tests must pass before training.

### 13.1 Shape tests

- image encoder returns `[B, D]`
- text encoder returns `[B, L, D]`
- prototype bank returns `[N, D]`
- contextualizer returns `[N, D]`
- router returns `[B, N]`
- aggregator returns `[B, D]`
- token scorer returns `[B, L]`
- pooler returns `beta` `[B, L]` and `T_pool` `[B, D]`
- projector returns `[B, 256]`
- contrastive logits return `[B, B]`

### 13.2 Masking tests

- padded tokens are masked out under every token policy
- excluded special tokens are masked out under `content_only`
- `content_plus_special` keeps special tokens
- `eos_only` keeps exactly one EOS position per sample when valid
- invalid tokens receive zero final beta

### 13.3 Probability normalization tests

- `alpha.sum(dim=-1)` is `1` within tolerance
- `beta.sum(dim=-1)` is `1` over valid kept tokens within tolerance
- `P_p.sum(dim=-1)` is `1` within tolerance

### 13.4 Gradient tests

- gradient reaches `Theta_v`
- gradient reaches projector parameters
- if backbones are unfrozen in an ablation test, gradient reaches allowed backbone parameters
- no gradient on frozen backbone parameters

### 13.5 Numerical stability tests

- forward pass with dummy batch produces finite loss
- no NaN in `alpha`
- no NaN in `beta`
- no NaN in `Q`
- no NaN in contrastive logits
- diversity loss finite

### 13.6 Behavioral sanity tests

- disabling contextualization makes `Theta_tilde == Theta_v`
- residual contextualization differs from overwrite when prototypes are non-trivial
- `Q` changes when `alpha` changes
- token scores change when `Q` changes
- switching token policy changes `token_keep_mask` as expected

### 13.7 Config tests

- every documented config field is accepted
- invalid config values raise clear errors
- ablation configs do not break shape contracts

### 13.8 Integration smoke test

Run one dummy batch end-to-end with:
- `B=4`
- `L` from tokenizer
- default config

Required assertions:
- no exception
- finite total loss
- non-empty diagnostics
- backward pass succeeds

---

## 14. Required logging and diagnostics

The implementation must log the following during training or debug evaluation.

### 14.1 Routing diagnostics

- mean alpha entropy
- alpha max probability mean
- prototype usage histogram `alpha.mean(dim=0)`
- dead prototype count under threshold
- routing temperature `tau_p`

### 14.2 Token pooling diagnostics

- mean beta entropy
- beta max probability mean
- top-weighted token identity if token strings are available
- fraction of beta mass on special tokens if token policy allows them
- token scoring temperature `tau_t`

### 14.3 Prototype geometry diagnostics

- pairwise prototype cosine matrix statistics:
  - mean
  - std
  - max off-diagonal cosine
- pairwise contextualized prototype cosine statistics
- diversity loss value

### 14.4 Representation norm diagnostics

- `||Q||`
- `||T_pool||`
- `||V||`
- `||Z_t_raw||`
- `||Z_v_raw||`

### 14.5 Contrastive diagnostics

- logit scale value
- positive logits mean
- negative logits mean
- `L_infonce`
- total loss

### 14.6 Optional structured debug payload

In debug mode, expose:
- decoded tokens per sample
- top-k beta-weighted tokens
- top-k alpha prototypes by weight
- mask summaries per sample

### 14.7 Logging implementation rule

Keep logging cheap in normal training and richer in debug mode. Expensive diagnostics such as full heatmaps may run only periodically.

---

## 15. Failure modes and debugging checklist

### 15.1 Prototype collapse

**Symptom**
- pairwise prototype cosine off-diagonal values are too high
- diversity loss remains large or decreases while prototypes remain near-duplicate
- routing concentrates on a few near-identical prototypes

**Likely cause**
- weak initialization
- diversity loss disabled or too weak
- wrong normalization in self-interaction or routing

**Inspect**
- prototype cosine matrix
- `L_div`
- prototype usage histogram
- initialization mode

---

### 15.2 Dead prototypes

**Symptom**
- some prototypes receive near-zero average routing mass for long periods

**Likely cause**
- poor initialization
- routing too sharp
- bank redundancy
- too many prototypes

**Inspect**
- `alpha.mean(dim=0)`
- dead prototype threshold counts
- `tau_p`
- initialization mode
- `num_prototypes`

---

### 15.3 Uniform routing

**Symptom**
- alpha entropy close to `log(N)` for most samples
- prototype usage nearly uniform and uninformative
- `Q` varies too little across images

**Likely cause**
- `tau_p` too high
- poor prototype separation
- routing similarity implementation bug

**Inspect**
- alpha entropy
- `tau_p`
- prototype cosine stats
- raw routing logits distribution

---

### 15.4 Overly sharp routing

**Symptom**
- alpha nearly one-hot from very early training
- unstable gradients
- dead prototypes emerge quickly

**Likely cause**
- `tau_p` too low
- raw dot product used accidentally
- norm explosion

**Inspect**
- alpha max probability
- routing similarity mode
- feature norms
- `tau_p`

---

### 15.5 Token pooling dominated by special tokens

**Symptom**
- EOS or other special tokens get most beta mass

**Likely cause**
- token policy bug
- special-token masking not applied
- raw dot-product scorer inflates norm effects

**Inspect**
- `token_keep_mask`
- top-weighted tokens
- token similarity mode
- beta mass by token type

---

### 15.6 Shape bugs

**Symptom**
- runtime shape mismatch
- silent broadcasting gives wrong results

**Likely cause**
- wrong use of `einsum`
- wrong transpose
- confusion between `[B, L, D]` and `[B, D, L]`

**Inspect**
- assert shapes at each module boundary
- unit tests for each module

---

### 15.7 Wrong softmax axis

**Symptom**
- alpha or beta do not sum to 1 along intended dimension
- strange routing or token weights

**Likely cause**
- omitted explicit `dim`
- incorrect dimension value

**Inspect**
- row sums of `P_p`, `alpha`, `beta`
- source code of all softmax calls

---

### 15.8 Masking bugs

**Symptom**
- padded or excluded tokens receive non-zero beta
- NaNs after token softmax

**Likely cause**
- masked logits set to zero instead of `-inf`
- empty keep rows
- wrong mask type or shape

**Inspect**
- `token_keep_mask`
- `beta_logits_masked`
- final `beta` on invalid positions

---

### 15.9 Exploding norms

**Symptom**
- `Q`, `T_pool`, or projector outputs have unstable norms
- contrastive logits saturate

**Likely cause**
- bad learning rate
- missing normalization before similarity
- projector instability

**Inspect**
- norm logs
- learning rates
- `normalize_projector_outputs`
- `logit_scale`

---

### 15.10 Temperature instability

**Symptom**
- contrastive logits become too large or too flat
- training stalls or diverges

**Likely cause**
- unstable learnable logit scale
- bad initialization
- no clamp when needed

**Inspect**
- `logit_scale`
- positive and negative logit stats
- optimizer group for logit scale

---

## 16. Minimal v1 implementation target

This is the first version to build before any optimization or extra ablation.

### 16.1 Exact minimal build

Implement exactly the following:

- CLIP ViT-B/16 image encoder wrapper that returns a global image embedding
- paired CLIP text encoder wrapper that returns token-level hidden states
- one shared trainable prototype bank with `N=32`
- default prototype initialization: row-normalized random
- prototype contextualization using normalized self-interaction plus residual
- cosine routing from global image embedding to contextualized prototypes
- single summary vector `Q = alpha @ Theta_tilde`
- cosine token scoring between `Q` and token states
- default `content_only` token policy
- masked softmax token pooling
- symmetric 2-layer MLP projectors with output dim `256`
- normalized projector outputs
- symmetric InfoNCE loss
- prototype diversity loss
- rich intermediate outputs for debugging
- required unit tests and required logging hooks

### 16.2 What not to add in minimal v1

Do not add:

- patch-token image interaction
- cross-attention
- balancing loss in default path
- top-k routing
- multi-head summaries
- multiple banks
- semantic concept labeling of prototypes
- extra losses beyond what is specified
- hidden optimizer tricks inside modules

### 16.3 Success criterion for minimal v1

Minimal v1 is complete only when:

1. the default config runs end-to-end on a dummy batch
2. all required unit tests pass
3. total loss is finite
4. gradients reach the prototype bank
5. alpha and beta behave like valid probability distributions
6. masking is correct
7. diagnostics are available for debugging and ablation

This is the implementation target Codex should build first.
