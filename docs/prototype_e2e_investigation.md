# Prototype E2E Investigation (Code-Grounded)

## 1. Executive Summary

- I traced the full path from YAML to train/eval scoring for the implemented prototype branch.
- The repo does not define explicit labels `version1/version2/version3` in code. The clean implementation-level reconstruction is by the two switches:
  - `prototype.routing_source`
  - `prototype.use_host_deflated_input`
- Across `configs/head_type/itself/*.yaml`, there are exactly 3 effective switch combinations:
  - `global + false`
  - `global + true`
  - `local_evidence + true`
- Critical finding: `use_host_deflated_input` is only applied inside the `local_evidence` routing path. So `global + true` is behaviorally the same as `global + false` in current code.
- `s_host`, `s_proto`, and `s_total` are not explicit variable names in training code, but their realized implementations are:
  - `s_host`: host-head similarity matrix (ITSELF: cosine global + optional GRAB mix).
  - `s_proto`: prototype pairwise similarity produced from projected prototype image/text embeddings via `compute_paired_similarity`.
  - `s_total`: only used in retrieval/eval fusion (`lambda_host * host_similarity + lambda_prototype * prototype_similarity`), not as the train objective.
- Stage behavior is primarily controlled by `training.freeze_schedule` (trainability + optional loss/lr overrides), not by `training.stage` alone.
- Most suspicious anchor-like source in current implementation is not the v1/v2 input toggle; it is the shared scoring interface and stage/loss coupling (especially shared image backbone path + detached teacher losses + host/proto fusion/weighting choices).

---

## 2. How the 3 Versions Are Actually Implemented

### Observed in code

- No hardcoded `version1/2/3` identifiers were found.
- Unique combos present in `configs/head_type/itself/*.yaml` are:
  - `routing_source=global`, `use_host_deflated_input=false`
  - `routing_source=global`, `use_host_deflated_input=true`
  - `routing_source=local_evidence`, `use_host_deflated_input=true`
- Evidence:
  - `configs/head_type/itself/direction1_optionB_locked_prototype.yaml:67,73`
  - `configs/head_type/itself/direction1_optionA_adaptive_prototype.yaml:67,73`
  - `configs/head_type/itself/direction2_stage2_optionA_frozen_transfer.yaml:67,73`
  - `configs/head_type/itself/direction2_stage2_optionB_lowlr_adaptation.yaml:67,73`

### Reconstructed version mapping

| Reconstructed version | Effective config | Prototype input used for routing | Actual behavior difference |
|---|---|---|---|
| V1 | `routing_source=global`, `use_host_deflated_input=false` | Global image embedding after adapter (`image_features`) | Baseline global routing |
| V2 | `routing_source=global`, `use_host_deflated_input=true` | Same as V1 (global path) | No effective difference vs V1 |
| V3 | `routing_source=local_evidence`, `use_host_deflated_input=true` | Local/patch tokens (optionally host-deflated residualized) | Different routing weights and downstream surrogate/scoring |

### Why V1 and V2 collapse

- `use_host_deflated_input` is only checked in `_prepare_local_routing_tokens(...)`:
  - `model/prototype/head.py:493-529`
- `_prepare_local_routing_tokens(...)` is only called when `routing_source == 'local_evidence'`:
  - `model/prototype/head.py:538-550`
- `routing_source == 'global'` path bypasses host-deflated logic:
  - `model/prototype/head.py:558-559`

### Interpretation

- End-to-end, the meaningful split is effectively:
  - `{global}` vs `{local_evidence (+ optional deflation)}`
- If your "3 versions" hypothesis is tied only to these two switches, current code yields 2 behavior classes, not 3, unless you include configs that set `local_evidence + false` (not present in inspected YAMLs).

---

## 3. Config Trace: `prototype.routing_source` and `prototype.use_host_deflated_input`

### 3.1 YAML -> args namespace

- Canonical mapping entries:
  - `('prototype','routing_source') -> prototype_routing_source` (`utils/config.py:70`)
  - `('prototype','use_host_deflated_input') -> prototype_use_host_deflated_input` (`utils/config.py:76`)
- Backward-compatible alias map also includes `routing_source` (`utils/config.py:237`).
- Parsing/merge flow:
  - CLI parse -> load base+override YAML -> apply config maps -> finalize args (`utils/options.py:423-443`)
  - Config application precedence: CLI > explicit override file > merged config (`utils/config.py:1047-1061`)

### 3.2 Defaults if omitted

- Parser defaults:
  - `prototype_routing_source='global'` (`utils/options.py:159`)
  - `prototype_use_host_deflated_input=False` (`utils/options.py:166-173`)
- Base YAML also sets:
  - `routing_source: global` (`configs/base.yaml:67`)
  - `use_host_deflated_input: false` (`configs/base.yaml:73`)
- Final normalization:
  - `args.prototype_routing_source = ...lower()` (`utils/options.py:397`)
  - `args.prototype_use_host_deflated_input = bool(...)` (`utils/options.py:411`)

### 3.3 args -> model construction

- Read in prototype builder:
  - routing source read (`model/prototype/build.py:31`)
  - deflated-input flag read (`model/prototype/build.py:47-49`)
- Passed into head ctor:
  - `routing_source=...` (`model/prototype/build.py:142`)
  - `use_host_deflated_input=...` (`model/prototype/build.py:148`)

### 3.4 Runtime control points

- `routing_source` controls branch in `_compute_routing_weights`:
  - local evidence: `route_from_local_evidence` (`model/prototype/head.py:538-550`)
  - global routing: `self.router(image_features, ...)` (`model/prototype/head.py:558`)
- `use_host_deflated_input` controls only residualization in local token prep:
  - condition (`model/prototype/head.py:515`)
  - transform function (`model/prototype/head.py:448-475`)

### 3.5 Functionally coupled keys

- Only active when `routing_source=local_evidence`:
  - `prototype.local_routing_temperature` (`build.py:32-36`, `head.py:119-123`, router call `head.py:547`)
  - `prototype.local_routing_pooling` (`build.py:37`, `head.py:123-127`, router call `head.py:546`)
  - `prototype.local_routing_use_adapter` / `prototype.local_routing_adapter_dim` (`build.py:38-43`, `head.py:128-171`)
  - `prototype.local_routing_normalize_inputs` (`build.py:44-46`, `head.py:129`, router call `head.py:548`)
- Routing/scoring normalization knobs:
  - `prototype.routing_type`, `prototype.routing_temperature`, `prototype.normalize_for_routing` (`build.py:110-112`, `head.py:156-160`, `router.py:40-45,127-130`)
  - projector output normalization (`model/prototype/projector.py:48`)
- Fusion knobs (eval path):
  - `fusion.lambda_host`, `fusion.lambda_prototype` (`model/fusion.py:11-13,54-56`)
- Training host weight (train objective path):
  - `objectives.lambda.host -> self.lambda_host` (`model/pas_model.py:69,1135`)
- Stage control:
  - `training.freeze_schedule` controls trainability/lr/loss overrides (`processor/processor.py:267-345`, `utils/freeze_schedule.py:301-391`)

---

## 4. Forward Path Trace: image -> prototype input -> routing -> surrogate -> scores

### 4.1 Image features entering prototype branch

- `image_output.projected_tokens` is produced by `_encode_image_intermediates(...)` (`model/pas_model.py:731-737`).
- Global image vector used by prototype call:
  - `image_global = projected_tokens[:, 0, :]` (`model/pas_model.py:740`)
  - passed as `image_embeddings` to prototype head (`model/pas_model.py:1121`)
- Local tokens passed for local routing path:
  - `image_local_tokens=image_output.projected_tokens` (`model/pas_model.py:1122`)

### 4.2 Prototype image branch

- In `encode_image_branch(...)`:
  - `image_features = image_adapter(image_embeddings)` (`model/prototype/head.py:576`)
  - routing weights computed (`model/prototype/head.py:588-595`)
  - summary `Q = alpha @ contextualized_prototypes` (`model/prototype/head.py:596`, `model/prototype/aggregator.py:15`)
  - image proxy features: `image_features + proto_query_proj(Q)` (`model/prototype/head.py:598`)
  - image projection to prototype score space (`model/prototype/head.py:599`, `model/prototype/projector.py:46-49`)

### 4.3 Routing branch split

- Global routing:
  - `alpha = Router(image_features, contextualized_prototypes)` (`model/prototype/head.py:558`)
- Local evidence routing:
  - local tokens prepared from patch tokens (`model/prototype/head.py:511-523`)
  - optional host-deflated residualization (`model/prototype/head.py:515-521`)
  - routed by local pooling mode (`model/prototype/head.py:543-549`, `model/prototype/router.py:90-99`)

### 4.4 Host-deflated local tokens (when active)

- `host_unit = normalize(img_global.detach())` (`model/prototype/head.py:463`)
- remove projection onto host direction and normalize residual (`model/prototype/head.py:464-467`)
- Returns residualized local tokens with diagnostics (`model/prototype/head.py:475`)

### 4.5 Text basis and surrogate construction

- Build per-text per-prototype basis bank (`model/prototype/head.py:712-754`)
- Reconstruct diagonal surrogate text:
  - `T_hat_i = sum_n alpha_i[n] * basis_bank_i[n]` via `einsum('bn,bnd->bd',...)` (`model/prototype/head.py:767-775`)
- Project surrogate text with `text_projector` (`model/prototype/head.py:973`)

### 4.6 Pairwise surrogate logits for retrieval loss

- For each image i and text t, combine image-i routing with text-t basis bank:
  - `einsum('in,tnd->tid', routing_chunk, basis_chunk)` (`model/prototype/head.py:813`)
- Project pairwise surrogate text and score via `compute_paired_similarity` (`model/prototype/head.py:814-817`)
- Return `[B_image, B_text]` logits (`model/prototype/head.py:902-910`)

### Interpretation

- The "version" switches affect routing weights `alpha`, which then affect both diagonal surrogate `T_hat` and pairwise surrogate logits.
- Downstream scorer/loss code is shared; differences mainly enter through `alpha` and any upstream feature path affected by local routing.

---

## 5. Exact Score Definitions (`s_host`, `s_proto`, `s_total`)

### 5.1 `s_host`

### Observed in code

- Host similarity comes from host head `compute_similarity_matrix(...)` (`model/pas_model.py:865-869`).
- ITSELF host implementation:
  - global cosine matrix (`model/host_heads.py:481-484` with `cosine_similarity_matrix` at `model/host_heads.py:179-182`)
  - optional GRAB mix: `w_global * sim_global + (1-w_global) * sim_grab` (`model/host_heads.py:485-486`)

### Interpretation

- For ITSELF, `s_host` is cosine-style similarity in host space; it is not fused with prototype during host-loss computation.

### 5.2 `s_proto`

### Observed in code

- Core pair scorer in prototype loss stack:
  - `compute_paired_similarity(image,text)` -> optional normalize + dot + `logit_scale` (`model/prototype/losses.py:152-156`)
- Training retrieval logits (`loss_ret`) use `surrogate_pairwise_logits` from `compute_surrogate_pairwise_logits(...)` (`model/prototype/head.py:887-910`, called at `1015-1021`).
- Eval exact prototype similarity uses `compute_pairwise_similarity(...)` (`model/prototype/head.py:912-934`, called via `model/pas_model.py:896-905`).

### Interpretation

- `s_proto = cos(z_global_image, Z_text)` is not exact in implementation.
- Actual image tensor is `image_projected` from `image_proxy_features` (`image_adapter(image_global)` plus `proto_query_proj(summary)`), then projected (`model/prototype/head.py:576,598-599`).
- So prototype image score tensor is transformed/global-plus-summary, not raw `image_global` directly.

### 5.3 `s_total`

### Observed in code

- Eval/retrieval fusion is explicit in `ResidualScoreFusion`:
  - `host_weight*host_similarity + prototype_weight*prototype_similarity` (`model/fusion.py:54-56`)
  - invoked by `PASModel.fuse_retrieval_similarity` (`model/pas_model.py:871-886`)
- Evaluator also calls the same fusion interface (`utils/metrics.py:370-385,443-449`).

### Important distinction

- Training objective uses:
  - `loss_total = lambda_host * host_losses['loss_total'] + prototype_losses['loss_total']` (`model/pas_model.py:1135`)
- This is not the same as fusing `s_host` and `s_proto` into a single train-time score matrix.

---

## 6. Loss-by-Loss Investigation

## `loss_host`

- Defined in host head forward (ITSELF path): `loss_total = tal_loss + cid_loss` (`model/host_heads.py:797,829-836`).
- Called from PAS forward via host head (`model/pas_model.py:1083-1091`), then added with weight `lambda_host` (`model/pas_model.py:1135`).
- Depends on:
  - host global embeddings from image/text encoder outputs (`model/host_heads.py:418,452`)
  - optional GRAB embeddings (`model/host_heads.py:431-443,460-472`)
- Gradient targets:
  - host retrieval modules (`host_head.*`) when trainable
  - base encoder outputs upstream when `host_backbone` trainable
- Stage usage:
  - Stage1 direction2 config sets `use_host_loss=false`, `lambda_host=0.0` (`direction2_stage1...yaml:99,118`)
  - Stage2 direction2 configs set `use_host_loss=true`, `lambda_host=0.1` (`direction2_stage2...yaml:99,118`)
- Weight source:
  - `objectives.lambda.host` -> `args.lambda_host` -> `self.lambda_host` (`utils/config.py:110`, `model/pas_model.py:69`)
  - phase overrides can mutate it (`utils/freeze_schedule.py:319-337`)

## `loss_ret` (prototype retrieval)

- Defined: `surrogate_retrieval_loss(...)` row-wise CE on `[B,B]` (`model/prototype/losses.py:419-429`).
- Called in `PrototypeLosses.forward(...)` when `use_loss_ret` (`model/prototype/losses.py:555,585-587`).
- Input tensor origin:
  - `surrogate_pairwise_logits` from `PrototypeConditionedTextHead.compute_surrogate_pairwise_logits` (`model/prototype/head.py:1015-1021`, `887-910`).
- Formula:
  - `loss_ret = CE(surrogate_pairwise_logits, arange(B))`
  - weighted as `lambda_ret * loss_ret` (`model/prototype/losses.py:586`)
- Depends on:
  - surrogate text construction path + routing + prototype image projected path
  - not on host score by default
- Host coupling:
  - only optional through `loss_weight_ret` if enabled (`model/prototype/losses.py:523-530,556-558`)
- Stage usage:
  - enabled in inspected stage1/stage2 direction2 configs (`direction2_stage1...yaml:109`, `direction2_stage2...yaml:109`)
- Detach/freeze notes:
  - no detach in `loss_ret` itself; detach appears only in optional weighted host-guided variant (`model/prototype/losses.py:467`)

## `loss_support`

- Defined: `support_loss(routing_weights)` (`model/prototype/losses.py:394-401`).
- Called in forward unconditionally, active if `use_loss_sup` true (`model/prototype/losses.py:589,395`).
- Formula:
  - effective support: `1 / sum_n alpha_n^2` (`model/prototype/losses.py:391-392`)
  - penalty: `mean(max((support-target)/target,0)^2)` (`model/prototype/losses.py:398-400`)
- Depends only on routing weights `alpha`.
- Gradients:
  - to routing outputs and upstream image/prototype path that produces `alpha`.
- Stage usage:
  - enabled in inspected direction2 stage configs (`...yaml:106`, weight at `...yaml:126`)
- Weight source:
  - `objectives.lambda.support -> lambda_sup` (`utils/config.py:118`) with alias support handling (`utils/options.py:386-389`)

## `loss_gap`

- Defined as `fidelity_gap_loss(...)` (`model/prototype/losses.py:371-389`).
- Called in forward (`model/prototype/losses.py:583-584`).
- Formula:
  - `pos = diag(cos(surrogate, exact_detached))`
  - `hardneg = max offdiag`
  - `loss = mean(max(margin - pos + hardneg, 0))`
- Depends on surrogate text embedding and detached exact text embedding.
- Gradient targets:
  - surrogate branch only (exact side detached).
- Stage usage:
  - disabled in inspected stage1/stage2 direction2 configs (`use_loss_gap:false`, `lambda gap:0.0`: line `105,125`)

## `loss_diag` (included because pipeline-critical)

- Implemented as alias to directional/relative diagonal loss (`model/prototype/losses.py:317,320,652`).
- Core function: `symmetric_relative_diagonal_loss` (`model/prototype/losses.py:322-369`).
- Formula:
  - student = normalize(surrogate)
  - teacher = normalize(exact.detach())
  - similarity = `student @ teacher^T`
  - CE row + CE col over diagonal targets (`model/prototype/losses.py:355-359`)
- Gradient targets:
  - surrogate path only (teacher exact detached at `line 338`).
- Stage usage:
  - enabled in inspected stage1/stage2 direction2 configs (`...yaml:104`, nonzero diag lambda `...yaml:124`)

### `loss_ret` special clarification (requested)

- In current implementation, `loss_ret` is computed from surrogate pairwise logits only.
- It does not directly consume exact text embeddings, host score, or fused score.
- Host score participates only in optional `loss_weight_ret`, not `loss_ret`.

### Interpretation (support/gap role)

- `loss_support` and `loss_gap` are presented as stabilizers, but they still shape retrieval geometry indirectly because they alter `alpha` (support) and surrogate embedding margins (gap), which feed retrieval logits.

---

## 7. Stage 1 Pipeline

(Using `configs/head_type/itself/direction2_stage1_prototype_stability.yaml` as canonical Stage 1 in this codebase.)

### Observed in code/config

- Routing/input switches:
  - `routing_source: global` (`line 67`)
  - `use_host_deflated_input: false` (`line 73`)
- Loss setup:
  - `use_host_loss: false` (`line 99`)
  - `lambda_host: 0.0` (`line 118`)
  - `use_loss_ret: true`, `use_loss_diag: true`, `use_loss_support: true`, `use_loss_gap:false` (`lines 104-110,105-106`)
- Trainability schedule:
  - phase trainable groups: `[prototype_bank, prototype_projector, routing, fusion]` (`line 191`)
  - frozen groups: `[host_backbone, host_retrieval]` (`line 192`)
- Checkpoint init/load:
  - host load enabled from external path (`checkpointing.load.sources.host.enabled: true`, path line `308-309`)
  - prototype group loads disabled (`lines 310-318`)

### Runtime behavior

- At each epoch, freeze schedule applies `requires_grad` and rebuilds optimizer (`processor/processor.py:280-293`).
- Stage1 prototype losses are computed on a frozen host/backbone space (by schedule), while host loss is effectively off (`use_host_loss=false`, `lambda_host=0`).

### Interpretation

- Stage1 primarily learns prototype-side machinery under fixed host/backbone parameters (in the canonical direction2 setup).

---

## 8. Stage 2 Pipeline

(Using direction2 stage2 configs.)

### Stage2 Option A: frozen transfer

- Config:
  - `routing_source: local_evidence` (`direction2_stage2_optionA...yaml:67`)
  - `use_host_deflated_input: true` (`:73`)
- Trainability:
  - trainable `[host_backbone, host_retrieval, prototype_projector, fusion]` (`:191`)
  - frozen `[prototype_bank, routing]` (`:192`)
- Checkpoint loading:
  - host load disabled (`:302-304`)
  - prototype_bank/projector/fusion loaded from stage1 checkpoints (`:305-313`)

### Stage2 Option B: low-lr adaptation

- Same routing/input switches as Option A (`:67,73`).
- Trainability:
  - host + all prototype groups trainable (`:191-193`)
  - low lr overrides for prototype-side groups (`:194-199`)
- Checkpoint loading:
  - same prototype loads, host load disabled (`:309-320`)

### Initialization/load order details

- If prototype bank load is enabled and init mode would require dataset extraction, runtime forces `prototype_init='normalized_random'` before model build to avoid fallback extraction (`train.py:279-292`).
- After model construction, modular checkpoint loader restores configured groups (`train.py:299-305`, `utils/modular_checkpoint.py:379-437`).

### Host/prototype interaction in Stage2

- Train loss remains additive: `lambda_host * loss_host + loss_proto` (`model/pas_model.py:1135`).
- In direction2 stage2 configs, objective lambda host is `0.1` (`...yaml:118`), and freeze schedule does not override loss weights.

### Interpretation

- Stage2 Option A: mostly host recovery + projector/fusion adaptation around frozen bank/routing.
- Stage2 Option B: joint adaptation with prototype-side updates constrained by low LR.

---

## 9. Gradient / Trainable-Module Table

Legend: `Y/N` means parameter updates possible under stage schedule. `cond` means depends on config toggles (`use_loss_*`, `lambda_*`, scorer path). `pass-through` means no params in that transform.

### Canonical Stage1 (direction2_stage1) and Stage2 (direction2 OptionA / OptionB)

| Module / path | `loss_host` (S1/S2A/S2B) | `loss_ret` (S1/S2A/S2B) | `loss_support` (S1/S2A/S2B) | `loss_gap` (S1/S2A/S2B) | `loss_diag` (S1/S2A/S2B) | Notes |
|---|---|---|---|---|---|---|
| Visual/text backbone (`base_model.*`) | N / Y / Y | N / Y / Y | N / Y / Y | N / N / N (disabled) | N / Y / Y | S1 frozen by schedule (`stage1 yaml:192`) |
| Global image embedding path (`image_global -> image_adapter -> image_projector`) | N / cond / cond | Y / Y / Y | Y / Y / Y | N / N / N | Y / Y / Y | `image_global=projected_tokens[:,0,:]` (`pas_model.py:740`) |
| Host retrieval branch (`host_head.*`) | N / Y / Y | N / N / N | N / N / N | N / N / N | N / N / N | `host_retrieval` frozen in S1, trainable in S2A/B |
| Prototype bank (`prototype_head.prototype_bank`) | N / N / Y | Y / N / Y | Y / N / Y | N / N / N | Y / N / Y | Frozen in S2A (`stage2A yaml:192`) |
| Routing module (`router/local_routing_adapter/contextualizer`) | N / N / Y | Y / N / Y | Y / N / Y | N / N / N | Y / N / Y | Group `routing` frozen S2A |
| Host-deflated residual transform | N / pass-through / pass-through | N / pass-through / pass-through | N / pass-through / pass-through | N / N / N | N / pass-through / pass-through | No params; uses `img_global.detach()` (`head.py:463`) |
| Text basis/projector/surrogate constructor (`token_scorer/pooler/text_projector/...`) | N / Y / Y | Y / Y / Y | N / N / N | N / N / N | Y / Y / Y | Mainly in `fusion` + `prototype_projector` groups |

Important: even when a module group is frozen, gradients can still flow through its operations to upstream trainable tensors; table reflects parameter update eligibility under stage schedules.

---

## 10. Most Likely Source of Anchor-Like Behavior in the Current Implementation

### Observed in code

- V1 and V2 are effectively identical (host-deflated flag inactive under global routing).
- Prototype image score path starts from global image embedding and adds `proto_query_proj(summary)` where `proto_query_proj` is zero-initialized (`model/prototype/head.py:185-188,598`).
- Prototype and host both score against shared encoder-derived embeddings; eval combines them linearly (`model/fusion.py:54-56`).
- `loss_diag` and `loss_gap` use detached exact-text teacher targets (`model/prototype/losses.py:338,373`).

### Interpretation

- The strongest anchor-like pressure is more likely from the shared scoring interface and loss coupling than from v1/v2 input switch differences.
- Stage2 Option A can further reinforce anchoring because prototype bank/routing are frozen while host is trainable, making adaptation concentrate in host + lightweight prototype-side mappings.

---

## 11. Key Mismatches / Ambiguities / Risks in the Current Pipeline

1. **Version ambiguity in naming**
- No explicit "v1/v2/v3" identifiers in code; only recoverable via config combos.

2. **V2 no-op risk**
- `routing_source=global` with `use_host_deflated_input=true` does not activate host deflation.

3. **Comment/config mismatch around stage objectives**
- Some stage comments mention gap-focused stability, but inspected direction2 configs set `use_loss_gap:false` and `lambda_gap:0.0`.

4. **Training-vs-eval weighting mismatch risk**
- Train mixes losses with `lambda_host` (`pas_model.py:1135`), eval mixes similarities with fusion lambdas (`fusion.py:54-56`); these are separate knobs and can drift.

5. **Stage control ambiguity risk**
- `training.stage` is validated metadata, but real trainability/loss/lr behavior is from `training.freeze_schedule`.

6. **Backbone semantics abstraction**
- `projected_tokens` semantics ultimately depend on base CLIP runtime implementation; this code assumes CLS token at index 0 and valid projected token contracts.

---

## 12. Appendix: File Paths and Code References

### Config and argument plumbing

- `utils/options.py:149-173,290-443`
- `utils/config.py:12-89,223-260,1036-1061`
- `configs/base.yaml:65-79`

### Version configs (examples)

- `configs/head_type/itself/direction1_optionB_locked_prototype.yaml:67,73`
- `configs/head_type/itself/direction1_optionA_adaptive_prototype.yaml:67,73`
- `configs/head_type/itself/direction2_stage1_prototype_stability.yaml:67,73,97-130,186-193,268-318`
- `configs/head_type/itself/direction2_stage2_optionA_frozen_transfer.yaml:67,73,97-130,186-193,263-313`
- `configs/head_type/itself/direction2_stage2_optionB_lowlr_adaptation.yaml:67,73,97-130,186-199,270-320`

### Prototype model path

- `model/prototype/build.py:31-49,123-148`
- `model/prototype/head.py:114-127,448-475,493-559,568-621,767-777,813-817,887-910,936-1033,1037-1076`
- `model/prototype/router.py:57-117,119-145`
- `model/prototype/projector.py:46-49`
- `model/prototype/token_scorer.py:35-41`
- `model/prototype/token_pooler.py:18-25,45`
- `model/prototype/aggregator.py:15`
- `model/prototype/contextualizer.py:84-93,103`
- `model/prototype/losses.py:152-156,322-369,371-401,419-429,445-482,555-601,638-705`

### Host/model/fusion/retrieval path

- `model/pas_model.py:69,350-370,627-642,730-747,865-887,888-937,1083-1135`
- `model/host_heads.py:179-182,417-423,451-457,481-486,692-699,715-742,797,829-836`
- `model/fusion.py:23-30,46-56`
- `utils/metrics.py:370-397,443-461,525-536`

### Freeze schedule and checkpoints

- `utils/module_group_registry.py:8-44`
- `utils/freeze_schedule.py:15-27,301-307,310-391`
- `processor/processor.py:267-345`
- `train.py:279-305`
- `utils/modular_checkpoint.py:68-75,357-379,384-437`

