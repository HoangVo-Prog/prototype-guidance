# INTEGRATION_PLAN.md

## 0. Objective

Integrate the new **image-side prototype-guided interaction mechanism** into the existing codebase (`WACV2026-Oral-ITSELF-custom`) while:

* preserving all existing training, evaluation, and baseline pipelines
* avoiding destructive refactors
* introducing the new method as a **configurable extension**
* ensuring backward compatibility and reproducibility

---

## 1. Core Integration Principles

### 1.1 Non-destructive integration

* DO NOT rewrite existing modules unless strictly necessary
* DO NOT remove or rename existing public APIs
* DO NOT break existing training scripts or evaluation flows

### 1.2 Config-driven activation

All new functionality must be gated behind config flags:

```yaml
model:
  use_prototype_bank: false
  use_image_conditioned_pooling: false
  use_prototype_contextualization: false
```

Default = **all OFF → exact old behavior**

---

### 1.3 Backward compatibility

* Existing baselines (CLIP-style, mean pooling, etc.) must:

  * run without modification
  * produce identical outputs (within numerical tolerance)

---

### 1.4 Additive design

* New components must be **plug-in modules**, not invasive rewrites
* Prefer:

  * wrapper modules
  * optional branches
  * feature flags

---

## 2. High-Level Mapping

### 2.1 Existing pipeline (assumed)

```
image -> vision encoder -> image embedding
text  -> text encoder   -> text embedding
→ similarity (cosine / InfoNCE)
```

---

### 2.2 New pipeline (when enabled)

```
image -> vision encoder -> visual tokens + global embedding
                               ↓
                     prototype bank (new)
                               ↓
               image-conditioned prototype summary
                               ↓
text tokens -> text encoder -> token features
                               ↓
             image-conditioned token pooling (new)
                               ↓
                    enhanced text embedding

→ similarity (same head as before)
```

---

## 3. Module-Level Integration

### 3.1 Reuse AS-IS (must not change)

* Vision backbone (ViT / ResNet / etc.)
* Text backbone (Transformer / BERT / CLIP text encoder)
* Existing projection heads
* Loss functions (InfoNCE, etc.)
* Training loop
* Evaluation scripts
* Dataset loaders

---

### 3.2 Refactor IN-PLACE (minimal, controlled)

#### (A) Model forward()

Add optional branches:

```python
if not use_prototype_bank:
    return original_forward(...)
else:
    return new_forward(...)
```

Constraints:

* original path must remain untouched
* new path must reuse as much intermediate outputs as possible

---

#### (B) Text pooling module

Replace hard-coded pooling with:

```python
if use_image_conditioned_pooling:
    pooled = new_pooling(...)
else:
    pooled = original_pooling(...)
```

---

### 3.3 Add NEW modules

Create a new directory:

```
models/prototype/
```

#### Required modules:

---

### (1) prototype_bank.py

Responsibilities:

* maintain learnable prototype matrix Θ_v ∈ ℝ^{N×D}
* support:

  * initialization (random / k-means / pretrained)
  * forward assignment

Outputs:

* assignment weights
* prototype representations

---

### (2) prototype_assignment.py

Responsibilities:

* compute similarity between visual tokens and prototypes
* produce soft / sparse assignment

Constraints:

* no heavy cross-attention
* must be efficient (O(N × #tokens))

---

### (3) prototype_contextualization.py (optional flag)

Responsibilities:

* allow interaction between prototypes
* controlled by:

```yaml
use_prototype_contextualization: true/false
```

---

### (4) image_summary.py

Responsibilities:

* aggregate token-level info into prototype-level summary
* output:

```
prototype_summary: [N, D]
```

---

### (5) token_pooling.py

Responsibilities:

* perform image-conditioned pooling over text tokens

Inputs:

* text token embeddings [L, D]
* prototype summary [N, D]

Output:

* pooled text embedding [D]

---

## 4. Data Flow Integration

### 4.1 Vision side

```python
v_tokens, v_global = vision_encoder(image)

if use_prototype_bank:
    proto_assign = prototype_assignment(v_tokens)
    proto_summary = aggregate(v_tokens, proto_assign)
```

---

### 4.2 Text side

```python
t_tokens = text_encoder(text)

if use_image_conditioned_pooling:
    t_final = conditioned_pooling(t_tokens, proto_summary)
else:
    t_final = original_pooling(t_tokens)
```

---

### 4.3 Final similarity

UNCHANGED:

```python
sim = cosine(image_embedding, text_embedding)
```

---

## 5. Config System

### 5.1 Add new config group

```yaml
prototype:
  num_prototypes: 32
  init: "kmeans"  # or "random"
  temperature: 0.07
  sparse_assignment: true
```

---

### 5.2 Feature toggles

```yaml
model:
  use_prototype_bank: true
  use_image_conditioned_pooling: true
  use_prototype_contextualization: false
```

---

## 6. Training Compatibility

### 6.1 Losses

Keep existing losses unchanged.

Optional additions (config-controlled):

```yaml
loss:
  use_diversity_loss: false
  use_balance_loss: false
```

---

### 6.2 Logging

Add:

* prototype usage statistics
* assignment entropy
* pooling weights distribution

---

## 7. Baseline Preservation

Must support:

* CLIP-style EOS / CLS pooling
* mean pooling
* text-only attention pooling

Implementation rule:

* baseline = config switch
* no duplicate code paths

---

## 8. Testing Requirements

### 8.1 Unit tests

* prototype assignment shape correctness
* pooling output shape
* no NaN / Inf

---

### 8.2 Regression tests

* run baseline before integration
* run baseline after integration
* ensure same metrics

---

### 8.3 Smoke test

* 1 batch forward
* 1 epoch mini training
* confirm loss decreases

---

## 9. Risks and Constraints

### 9.1 Architectural risks

* accidental overwrite of baseline pooling
* prototype bank dominating training
* shape mismatch in token aggregation

---

### 9.2 Performance risks

* prototype assignment too slow
* memory overhead from token-level ops

---

### 9.3 Compatibility risks

* breaking checkpoint loading
* config mismatch

---

## 10. Strict DO / DO NOT

### DO

* reuse existing encoders
* isolate new logic into modules
* keep everything config-driven

---

### DO NOT

* rewrite training loop
* introduce cross-attention
* change embedding dimension
* break existing evaluation scripts

---

## 11. Expected Outcome

After integration:

* repo supports BOTH:

  * original baseline
  * new prototype-based method
* switching is done via config only
* Codex can implement incrementally without ambiguity

