# Prototype Metrics Taxonomy and W&B Logging Guide

This document reorganizes the current metric surface into a cleaner taxonomy for prototype debugging and validation analysis. The goal is to make W&B easier to navigate, reduce the overload caused by a single `debug/*` bucket, and make validation diagnostics readable without hiding everything under `val/debug/*`.

The current repository already exposes a rich metric surface under `train/*`, `debug/*`, and `val/*`, including loss terms, routing metrics, prototype usage metrics, gradient norms, geometry statistics, token pooling diagnostics, retrieval metrics, and validation diagnostics. This guide keeps those metrics, but groups them under more specific namespaces and gives each group a clear purpose. fileciteturn0file0

---

## 1. Main problems with the current layout

### 1.1 `debug/*` is too broad

Right now, `debug/*` contains many different kinds of signals at once: routing behavior, prototype collapse indicators, embedding norms, surrogate and exact geometry, proxy alignment, gradients, token pooling, and prototype-bank structure. That makes it hard to answer a simple question such as: *is the prototype bank alive?* or *is the routing meaningful?* because the relevant metrics are scattered inside one very large namespace. fileciteturn0file0

### 1.2 `val/debug/*` mixes unrelated validation signals

The current validation debug metrics include dataset structure, model state, geometry, and norm statistics in the same namespace. For example, positive gallery counts, logit scale, exact positive cosine, and image projected norms are all logged under `val/debug/*`. This is difficult to scan in W&B and makes validation dashboards less interpretable. fileciteturn0file0

### 1.3 The metric surface is rich, but not query-friendly

The repo already logs enough metrics to debug prototype learning well. The issue is not missing metrics. The issue is missing semantic grouping. The same metric surface becomes much easier to use if it is grouped by the question it answers.

---

## 2. Recommended top-level taxonomy

### 2.1 Training namespaces

```text
train/epoch
train/step
train/lr

train/loss/*
train/model/*
train/prototype_usage/*
train/routing/*
train/fidelity/*
train/geometry/*
train/proxy/*
train/prototype_geometry/*
train/token_pool/*
train/norm/*
train/grad/*
```

### 2.2 Validation namespaces

```text
val/loss/*
val/retrieval/*
val/model/*
val/data/*
val/geometry/*
val/norm/*
```

### 2.3 Principle

A namespace should answer one question only.

- `prototype_usage/*` answers whether prototypes are alive, dead, or collapsing.
- `routing/*` answers how selective or diffuse the routing is.
- `fidelity/*` answers whether the surrogate is learning the exact target.
- `geometry/*` answers whether embedding space margins are improving.
- `grad/*` answers whether the relevant modules are receiving meaningful gradients.
- `prototype_geometry/*` answers whether the bank itself is diverse or redundant.

This is much easier to search and filter in W&B than a single `debug/*` bucket.

---

## 3. Proposed mapping from current metrics to cleaner groups

The following sections map the current metric surface into semantic groups. These groups can be used both as documentation and as the future W&B namespace layout.

---

## 4. Group A. Optimization and schedule state

### Purpose
Track global optimization state and scalar controls that affect the whole training process.

### Recommended namespace
```text
train/model/*
```

### Current metrics
- `debug/logit_scale`
- `debug/host_logit_scale`
- `debug/host_retrieval_temperature`
- `debug/fusion_coefficient`
- `debug/proxy_temperature`
- `debug/retrieval_temperature`
- `debug/host_loss_total`
- `debug/host_loss_ret`

### Description
These metrics describe global scaling or mixing parameters that shape the loss surface and similarity scores. They are not prototype-specific by themselves, but they strongly influence how easy it is for the prototype branch to learn.

### How to read them
- If temperatures or logit scales drift unexpectedly, many geometry metrics may change even when the underlying representation quality did not change much.
- `fusion_coefficient` is important if host and prototype branches are combined. It determines how much the prototype path can influence the final score.

---

## 5. Group B. Prototype usage and collapse monitoring

### Purpose
Answer the question: **Are the prototypes alive, being used broadly enough, and avoiding collapse into a few hubs?**

### Recommended namespace
```text
train/prototype_usage/*
```

### Current metrics
- `debug/prototype_usage_entropy`
- `debug/prototype_usage_max`
- `debug/prototype_dead_count`
- `debug/prototype_active_count_eps_1e-3`
- `debug/prototype_active_count_eps_1e-2`
- `debug/prototype_active_count_eps_1e-3_window_100`
- `debug/prototype_active_count_eps_1e-3_window_500`
- `debug/prototype_active_count_eps_1e-2_window_100`
- `debug/prototype_active_count_eps_1e-2_window_500`
- `debug/prototype_usage_entropy_window_100`
- `debug/prototype_usage_entropy_window_500`
- `debug/prototype_usage_max_window_100`
- `debug/prototype_usage_max_window_500`
- `debug/routing_top1_usage_entropy`
- `debug/routing_top1_usage_max`
- `debug/routing_top1_dead_count`
- `debug/routing_top1_active_count_window_100`
- `debug/routing_top1_active_count_window_500`
- `debug/routing_top1_dead_count_window_100`
- `debug/routing_top1_dead_count_window_500`
- `debug/routing_top1_usage_entropy_window_100`
- `debug/routing_top1_usage_entropy_window_500`
- `debug/routing_top1_usage_max_window_100`
- `debug/routing_top1_usage_max_window_500`
- `debug/prototype_assignment_entropy`

### Description of the subgroup
This group measures **dataset-level prototype utilization**. It tells you whether the bank is meaningfully shared across the dataset or whether a few prototypes dominate all samples.

### Metric descriptions
- **usage entropy**: measures how evenly prototype assignments are distributed. Higher means more distributed usage.
- **usage max**: maximum share captured by a single prototype. Higher means more dominance by one prototype.
- **dead count**: number of prototypes effectively unused.
- **active count**: number of prototypes used above a fixed threshold.
- **windowed active count / entropy / max**: short-horizon usage statistics over recent batches. These are crucial for distinguishing healthy sparse routing from true hub collapse.
- **top1 usage metrics**: same logic as above, but computed only from each sample's winning prototype.
- **prototype assignment entropy**: entropy of the assignment distribution, useful as a complementary view of usage concentration.

### Interpretation
Good signs:
- dead count stays low
- usage max is not extreme
- active count remains reasonably high over windows
- windowed entropy remains healthy even if per-batch routing is sparse

Bad signs:
- one or two prototypes dominate top1 usage for long windows
- dead count steadily rises
- active count collapses and does not recover

---

## 6. Group C. Routing sharpness and support

### Purpose
Answer the question: **Is routing meaningful, selective, and structurally sensible?**

### Recommended namespace
```text
train/routing/*
```

### Current metrics
- `debug/routing_max_prob`
- `debug/routing_entropy`
- `debug/routing_effective_support`
- `debug/routing_effective_support_ipr`
- `debug/routing_effective_support_ipr_p10`
- `debug/routing_effective_support_ipr_p50`
- `debug/routing_effective_support_ipr_p90`
- `debug/routing_support_below_2_frac`
- `debug/routing_support_below_3_frac`
- `debug/routing_support_below_min_frac`
- `debug/routing_top1_minus_top2`
- `debug/routing_top2_mass`
- `debug/routing_top4_mass`

### Description of the subgroup
This group measures the **shape of the per-sample routing distribution** over prototypes.

### Metric descriptions
- **routing max probability**: how much weight goes to the top prototype.
- **routing entropy**: how diffuse the routing is. High means broad or flat routing. Low means sharper routing.
- **effective support**: effective number of prototypes used by a sample.
- **IPR variants**: inverse participation ratio views of effective support, including percentiles for distributional analysis.
- **support below 2 / 3 / min fractions**: fraction of samples that are too sparse.
- **top1 minus top2**: confidence gap between the first and second prototype.
- **top2 mass / top4 mass**: how much routing mass lives in the strongest few prototypes.

### Interpretation
Good signs:
- routing is not fully uniform
- routing is not one-hot for nearly every sample
- effective support is low to moderate, not near 1 for everything and not near `num_prototypes` either
- top1 minus top2 is meaningful, showing real preference

Bad signs:
- routing entropy is too high for too long, suggesting indecision or averaging
- routing entropy is too low while usage also collapses, suggesting hard hub routing
- support-below-k fractions are too high and never recover

---

## 7. Group D. Surrogate versus exact fidelity

### Purpose
Answer the question: **Is the prototype-mediated surrogate actually learning the intended exact target?**

### Recommended namespace
```text
train/fidelity/*
```

### Current metrics
- `debug/diag_cos_full`
- `debug/diag_cos_top1`
- `debug/diag_cos_top2`
- `debug/diag_cos_top4`
- `debug/loss_diag_full`
- `debug/loss_diag_top1`
- `debug/loss_diag_top2`
- `debug/loss_diag_top4`
- `train/loss_diag`
- `val/loss_diag`

### Description of the subgroup
This group measures **how well the amortized or surrogate path matches the exact path**. It is the most direct evidence that the prototype branch is learning the intended behavior.

### Metric descriptions
- **diag cosine metrics**: cosine agreement between surrogate and exact representations.
- **top1 / top2 / top4 variants**: fidelity under restricted prototype support. Useful for checking whether the model only learns a dominant prototype or truly learns mixtures.
- **diag loss metrics**: direct fidelity loss values. Lower is better.

### Interpretation
Good signs:
- `train/loss_diag` falls steadily
- `val/loss_diag` also improves
- `diag_cos_full` rises over time
- `diag_cos_top4` and `diag_cos_full` improve together if mixture learning matters

Bad signs:
- only `top1` improves but `full` stays weak
- train fidelity improves while validation fidelity is flat or deteriorates
- diagonal cosine remains stagnant even though routing sharpness changes a lot

---

## 8. Group E. Retrieval geometry and margin quality

### Purpose
Answer the question: **Are the learned embeddings becoming more retrieval-friendly?**

### Recommended namespace
```text
train/geometry/*
```

### Current metrics
- `debug/image_surrogate_positive_cosine_mean`
- `debug/image_surrogate_positive_cosine_std`
- `debug/image_surrogate_hardest_negative_cosine_mean`
- `debug/image_surrogate_hardest_negative_cosine_std`
- `debug/image_surrogate_margin_mean`
- `debug/image_surrogate_margin_min`
- `debug/image_surrogate_positive_logit_mean`
- `debug/image_surrogate_hardest_negative_logit_mean`
- `debug/image_exact_positive_cosine_mean`
- `debug/image_exact_positive_cosine_std`
- `debug/image_exact_hardest_negative_cosine_mean`
- `debug/image_exact_hardest_negative_cosine_std`
- `debug/image_exact_margin_mean`
- `debug/image_exact_margin_min`
- `debug/image_exact_positive_logit_mean`
- `debug/image_exact_hardest_negative_logit_mean`
- `debug/surrogate_pairwise_positive_cosine_mean`
- `debug/surrogate_pairwise_positive_cosine_std`
- `debug/surrogate_pairwise_hardest_negative_cosine_mean`
- `debug/surrogate_pairwise_hardest_negative_cosine_std`
- `debug/surrogate_pairwise_margin_mean`
- `debug/surrogate_pairwise_margin_min`
- `debug/surrogate_pairwise_positive_logit_mean`
- `debug/surrogate_pairwise_hardest_negative_logit_mean`
- `debug/surrogate_pairwise_logit_mean`
- `debug/surrogate_pairwise_logit_std`

### Description of the subgroup
This group measures **positive versus hardest-negative separation** for surrogate and exact representations.

### Metric descriptions
- **positive cosine mean/std**: similarity to the correct partner.
- **hardest negative cosine mean/std**: similarity to the most confusable negative.
- **margin mean/min**: positive similarity minus hardest negative similarity. One of the most useful geometry metrics.
- **positive/hardest-negative logit means**: same story in logit space.
- **pairwise statistics**: broader pairwise scoring statistics for the surrogate branch.

### Interpretation
Good signs:
- positive cosine increases
- hardest negative cosine does not rise in parallel
- margin mean becomes more positive
- exact and surrogate geometry improve in the same direction

Bad signs:
- fidelity improves but margin stays weak, meaning the surrogate may match the exact target without becoming useful for retrieval
- positive and hardest negative similarities rise together, leaving margin unchanged

---

## 9. Group F. Proxy alignment metrics

### Purpose
Answer the question: **Are class proxies and proxy-based training signals aligned with the learned representation?**

### Recommended namespace
```text
train/proxy/*
```

### Current metrics
- `debug/image_proxy_logit_mean`
- `debug/image_proxy_logit_std`
- `debug/image_proxy_logit_min`
- `debug/image_proxy_logit_max`
- `debug/text_proxy_logit_mean`
- `debug/text_proxy_logit_std`
- `debug/text_proxy_logit_min`
- `debug/text_proxy_logit_max`
- `debug/image_positive_proxy_cosine_mean`
- `debug/image_positive_proxy_cosine_std`
- `debug/image_hardest_negative_proxy_cosine_mean`
- `debug/image_hardest_negative_proxy_cosine_std`
- `debug/image_proxy_margin_mean`
- `debug/image_proxy_margin_min`
- `debug/text_positive_proxy_cosine_mean`
- `debug/text_positive_proxy_cosine_std`
- `debug/text_hardest_negative_proxy_cosine_mean`
- `debug/text_hardest_negative_proxy_cosine_std`
- `debug/text_proxy_margin_mean`
- `debug/text_proxy_margin_min`
- `debug/text_exact_positive_proxy_cosine_mean`
- `debug/text_exact_positive_proxy_cosine_std`
- `debug/text_exact_hardest_negative_proxy_cosine_mean`
- `debug/text_exact_hardest_negative_proxy_cosine_std`
- `debug/text_exact_proxy_margin_mean`
- `debug/text_exact_proxy_margin_min`
- `debug/class_proxy_norm_mean`
- `debug/class_proxy_norm_std`
- `debug/class_proxy_norm_min`
- `debug/class_proxy_norm_max`
- `debug/class_proxy_norm_normalized_mean`
- `debug/class_proxy_norm_normalized_std`
- `debug/class_proxy_norm_normalized_min`
- `debug/class_proxy_norm_normalized_max`

### Description of the subgroup
These metrics are relevant when proxy losses are enabled. They diagnose whether the representation is aligned with class proxies, whether positive proxy similarity is separated from hard negatives, and whether proxy norms are well behaved.

### Interpretation
Good signs:
- positive proxy cosine exceeds hardest-negative proxy cosine
- proxy margin becomes positive
- proxy norms stay stable instead of exploding or collapsing

Bad signs:
- proxy margin remains negative for many steps
- proxy logit range is extremely narrow or unstable
- proxy norms drift abnormally

---

## 10. Group G. Prototype bank geometry and redundancy

### Purpose
Answer the question: **Is the prototype bank diverse, or are many prototypes nearly duplicates?**

### Recommended namespace
```text
train/prototype_geometry/*
```

### Current metrics
- `debug/prototype_pairwise_cosine_mean`
- `debug/prototype_pairwise_cosine_std`
- `debug/prototype_pairwise_cosine_max`
- `debug/contextualized_prototype_pairwise_cosine_mean`
- `debug/contextualized_prototype_pairwise_cosine_std`
- `debug/contextualized_prototype_pairwise_cosine_max`
- `debug/prototype_contextualization_entropy`

### Description of the subgroup
This group measures the **internal geometry of the prototype bank**, before and after contextualization.

### Metric descriptions
- **pairwise cosine mean/std/max**: how similar prototypes are to each other overall and at the most similar pair.
- **contextualized pairwise cosine**: same statistics after contextualization.
- **prototype contextualization entropy**: how selective the contextualization process is.

### Interpretation
Good signs:
- pairwise cosine mean is moderate rather than very high
- maximum pairwise cosine is not close to full duplication
- contextualization does not collapse prototypes into near-identical vectors

Bad signs:
- contextualized prototypes become much more similar than raw prototypes
- contextualization entropy is very high and stays saturated, suggesting uniform averaging

---

## 11. Group H. Gradient flow and optimization health

### Purpose
Answer the question: **Is the prototype branch actually being trained?**

### Recommended namespace
```text
train/grad/*
```

### Current metrics
- `debug/grad_norm_class_proxies`
- `debug/grad_norm_image_projector`
- `debug/grad_norm_text_projector`
- `debug/grad_norm_prototype_bank`
- `debug/grad_norm_image_backbone`
- `debug/grad_norm_text_backbone`
- `debug/grad_norm_image_projected_output`
- `debug/grad_norm_surrogate_text_projected_output`
- `debug/grad_norm_exact_text_projected_output`
- `debug/surrogate_retrieval_grad_norm`
- `debug/grad_norm_total`

### Description of the subgroup
This group checks whether gradients reach the intended modules and whether learning is being routed into the prototype bank instead of being absorbed entirely by easier components.

### Interpretation
Good signs:
- prototype bank gradient norm is consistently non-zero
- projected outputs also receive meaningful gradients
- total gradient norm is stable

Bad signs:
- prototype bank gradient norm stays near zero for long periods
- projectors learn while the bank remains almost frozen
- gradient spikes appear without corresponding gain in fidelity or margin metrics

---

## 12. Group I. Token pooling and text-side conditioning

### Purpose
Answer the question: **Is the token-pooling mechanism behaving sensibly?**

### Recommended namespace
```text
train/token_pool/*
```

### Current metrics
- `debug/token_pool_entropy`
- `debug/beta_max_prob`
- `debug/token_special_mass`
- `debug/token_valid_fraction`
- `debug/valid_token_fraction`

### Description of the subgroup
These metrics describe how the model distributes attention or pooling mass across tokens.

### Metric descriptions
- **token pool entropy**: how diffuse the token-pooling distribution is.
- **beta max probability**: dominance of the strongest token.
- **token special mass**: fraction of mass placed on special tokens.
- **valid token fraction**: fraction of usable tokens after masking.

### Interpretation
Good signs:
- pooling is neither uniform over all tokens nor fully collapsed to a single token for every sample
- special-token mass is not excessively dominant unless that is intended by design

Bad signs:
- entropy saturates high, suggesting averaging over nearly all tokens
- almost all mass goes to special tokens or a degenerate token position

---

## 13. Group J. Norm and scale stability

### Purpose
Answer the question: **Are representation norms stable and consistent?**

### Recommended namespace
```text
train/norm/*
```

### Current metrics
- `debug/image_embed_norm_std`
- `debug/image_embed_norm_min`
- `debug/image_embed_norm_max`
- `debug/text_embed_norm_std`
- `debug/text_embed_norm_min`
- `debug/text_embed_norm_max`
- `debug/q_norm`
- `debug/surrogate_t_pool_norm`
- `debug/exact_t_pool_norm`
- `debug/image_feature_norm`
- `debug/image_embed_norm_raw`
- `debug/image_embed_unit_norm`
- `debug/surrogate_text_embed_norm_raw`
- `debug/surrogate_text_embed_unit_norm`
- `debug/exact_text_embed_norm_raw`
- `debug/exact_text_embed_unit_norm`

### Description of the subgroup
These metrics monitor norm stability for image, surrogate-text, exact-text, and query-like intermediate representations.

### Interpretation
Good signs:
- norms are stable across training
- raw norms do not explode or collapse
- unit-normalized norms remain close to expected values

Bad signs:
- large drifts in raw norms
- surrogate and exact norms behave very differently without a good reason

---

## 14. Loss namespaces

The repository already exposes a rich loss surface for training and validation, including total loss, host losses, proxy losses, retrieval loss, alignment loss, diagonal fidelity loss, support loss, diversity loss, and balance loss. These are already clean enough to keep under `train/loss/*` and `val/loss/*`. fileciteturn0file0

### Recommended layout
```text
train/loss/total
train/loss/host
train/loss/host_ret
train/loss/host_cid
train/loss/proto_total
train/loss/proxy
train/loss/ret
train/loss/align
train/loss/diag
train/loss/support
train/loss/diversity
train/loss/balance

val/loss/total
val/loss/host
val/loss/host_ret
val/loss/host_cid
val/loss/proto_total
val/loss/proxy
val/loss/ret
val/loss/align
val/loss/diag
val/loss/support
val/loss/diversity
val/loss/balance
```

### Note
Weighted variants can be kept either as:
- `train/loss_weighted/*`, `val/loss_weighted/*`

or as:
- `train/loss/*_weighted`, `val/loss/*_weighted`

The first option is usually cleaner in W&B.

---

## 15. Validation cleanup proposal

The current validation surface includes retrieval metrics under `val/pas/*` and several diagnostics under `val/debug/*`, such as positive gallery counts, logit scale, retrieval temperature, exact cosine, exact margin, and norm statistics. These should be split by purpose. fileciteturn0file0

### 15.1 Keep retrieval metrics separate

```text
val/retrieval/R1
val/retrieval/R5
val/retrieval/R10
val/retrieval/mAP
val/retrieval/mINP
val/retrieval/rSum
```

### 15.2 Replace `val/debug/*` with cleaner namespaces

#### Current to proposed mapping

```text
val/debug/eval_positive_gallery_count_min
-> val/data/positive_gallery_count_min

val/debug/eval_positive_gallery_count_mean
-> val/data/positive_gallery_count_mean

val/debug/eval_logit_scale
-> val/model/logit_scale

val/debug/eval_retrieval_temperature
-> val/model/retrieval_temperature

val/debug/eval_positive_exact_cosine_mean
-> val/geometry/exact_positive_cosine_mean

val/debug/eval_hardest_negative_exact_cosine_mean
-> val/geometry/exact_hardest_negative_cosine_mean

val/debug/eval_exact_margin_mean
-> val/geometry/exact_margin_mean

val/debug/eval_positive_exact_pair_cosine_mean
-> val/geometry/exact_pair_cosine_mean

val/debug/eval_image_projected_norm_mean
-> val/norm/image_projected_norm_mean

val/debug/eval_image_projected_norm_std
-> val/norm/image_projected_norm_std

val/debug/eval_positive_exact_text_embed_norm_mean
-> val/norm/exact_text_embed_norm_mean

val/debug/eval_positive_exact_text_embed_norm_std
-> val/norm/exact_text_embed_norm_std

val/debug/eval_positive_exact_text_embed_unit_norm_mean
-> val/norm/exact_text_embed_unit_norm_mean
```

### 15.3 Why this is better

- Validation retrieval lives in one obvious location: `val/retrieval/*`
- Validation geometry lives in one obvious location: `val/geometry/*`
- Validation norm checks live in one obvious location: `val/norm/*`
- Model state diagnostics live in `val/model/*`
- Dataset or evaluation-structure diagnostics live in `val/data/*`

This makes validation dashboards much easier to read and makes W&B search significantly more useful.

---

## 16. Minimal prototype debug board

If you want a compact board that is still strong enough to judge whether prototype learning is healthy, the following set is usually enough.

### A. Prototype alive or collapsed
- `train/prototype_usage/prototype_usage_entropy`
- `train/prototype_usage/prototype_usage_max`
- `train/prototype_usage/prototype_dead_count`
- `train/prototype_usage/routing_top1_active_count_window_100`

### B. Routing meaningful or not
- `train/routing/routing_entropy`
- `train/routing/routing_max_prob`
- `train/routing/routing_effective_support`
- `train/routing/routing_top1_minus_top2`

### C. Surrogate learning the exact target
- `train/loss/diag`
- `val/loss/diag`
- `train/fidelity/diag_cos_full`

### D. Geometry improving or not
- `train/geometry/image_surrogate_margin_mean`
- `train/geometry/image_exact_margin_mean`

### E. Prototype bank structurally healthy
- `train/prototype_geometry/prototype_pairwise_cosine_mean`
- `train/grad/grad_norm_prototype_bank`

This board is small, but it still answers the main debugging questions.

---

## 17. Suggested W&B section descriptions

These can be copied into dashboard panel descriptions or README documentation.

### `prototype_usage/*`
Measures dataset-level prototype utilization, dead prototypes, active prototype count, and collapse patterns over recent windows.

### `routing/*`
Measures how selective, diffuse, or sparse the per-sample routing distribution is over prototypes.

### `fidelity/*`
Measures how well surrogate representations match exact representations, including restricted-support variants such as top1, top2, and top4.

### `geometry/*`
Measures positive-versus-negative separation in embedding space for surrogate and exact retrieval paths.

### `proxy/*`
Measures alignment to class proxies, proxy margins, proxy logit statistics, and proxy norm stability.

### `prototype_geometry/*`
Measures pairwise similarity and redundancy within the prototype bank, before and after contextualization.

### `grad/*`
Measures whether gradients reach the prototype bank, projectors, backbones, and key projected outputs.

### `token_pool/*`
Measures token pooling sharpness, token-mass concentration, and special-token dominance.

### `norm/*`
Measures norm stability for image, query, surrogate-text, and exact-text representations.

### `val/retrieval/*`
Measures task-level retrieval quality on the validation set.

### `val/geometry/*`
Measures exact-path validation geometry, especially positive cosine, hardest-negative cosine, and margin.

### `val/norm/*`
Measures validation-time norm stability for image and exact text embeddings.

### `val/model/*`
Measures model-state scalars that affect validation scoring, such as logit scale and retrieval temperature.

### `val/data/*`
Measures validation-set or evaluator structure, such as positive gallery counts.

---

## 18. Practical implementation advice

1. Keep loss terms under `train/loss/*` and `val/loss/*`.
2. Move all current `debug/*` metrics into more specific semantic groups.
3. Remove `val/debug/*` entirely as a long-term namespace.
4. Use windowed usage metrics by default on the main prototype dashboard.
5. Use fidelity plus geometry together. Fidelity alone is not enough.
6. Keep proxy metrics visible only when proxy losses are enabled.
7. Build validation dashboards around `val/retrieval/*`, `val/geometry/*`, and `val/loss/*`.

---

## 19. Recommended final namespace sketch

```text
train/epoch
train/step
train/lr

train/loss/*
train/loss_weighted/*
train/model/*
train/prototype_usage/*
train/routing/*
train/fidelity/*
train/geometry/*
train/proxy/*
train/prototype_geometry/*
train/token_pool/*
train/norm/*
train/grad/*

val/loss/*
val/loss_weighted/*
val/retrieval/*
val/model/*
val/data/*
val/geometry/*
val/norm/*
```

This layout is much easier to search, group, and plot in W&B, while still preserving the richness of the current metric surface. It also cleanly separates training-time prototype debugging from validation-time retrieval analysis.
