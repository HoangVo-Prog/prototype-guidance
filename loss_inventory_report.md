# 1. Executive summary

| Loss name | Purpose | Defined at | Called at | Activation condition | Weight / coefficient | Included in total loss? |
|---|---|---|---|---|---|---|
| `host_tal_loss` | ITSELF host retrieval objective using positive-pair soft assignment + margin hinge over negatives | `model/host_heads.py:142` (`compute_tal_components`) | `model/host_heads.py:716`, `model/host_heads.py:732` | `use_host_loss==True`, `pids` present, `"tal"` in `host.itself_loss_names` | Implicit `1.0` inside host loss sum | Yes (when ITSELF host is active) |
| `host_cid_loss` | ITSELF host classification/discrimination objective (`pair CE` + `id CE`) | `model/host_heads.py:559` (`_compute_cid_loss_components`), helpers at `model/host_heads.py:242`, `model/host_heads.py:247` | `model/host_heads.py:745`, `model/host_heads.py:774` | `use_host_loss==True`, `self.training`, `pids` present, `"cid"` in `host.itself_loss_names`, classifier exists | Implicit `1.0` inside host loss sum | Yes (training mode only, ITSELF host) |
| `host_clip_retrieval_loss` (PAS clip host adapter) | CLIP-style retrieval CE over paired batch (i2t + t2i) | `model/vanilla_clip.py:77` (`retrieval_loss`) | `model/vanilla_clip.py:140` via `VanillaCLIPHead.forward` at `model/vanilla_clip.py:316` and `CLIPHostAdapter.forward` at `model/host_heads.py:282` | Host type `clip` in PAS path | `lambda_ret` inside module, forced to `1.0` by `build_host_head` (`model/host_heads.py:882`) | Yes (when PAS + clip host) |
| `host_clip_retrieval_loss` (host-only clip model) | Same CE retrieval objective in host-only runtime | `model/hosts/clip.py:61` (`_VanillaClipLoss`) | `model/hosts/clip.py:376` | Host-only clip path (`model.use_prototype_branch=false`) | Internally `1.0`; then scaled by `lambda_host` at model level (`model/hosts/clip.py:380`) | Yes |
| `proto_surrogate_retrieval_loss` | Prototype surrogate row-wise retrieval CE on `surrogate_pairwise_logits` | `model/prototype/losses.py:419` | `model/prototype/losses.py:555` | `use_loss_ret` | `lambda_ret` | Yes |
| `proto_weighted_surrogate_retrieval_loss` | Weighted row-wise surrogate retrieval using host margin-derived per-sample weights | `model/prototype/losses.py:445` | `model/prototype/losses.py:556` | `use_loss_weight_ret` and host logits available | `lambda_weight_ret` | Yes |
| `proto_directional_diagonal_loss` | Student-vs-teacher diagonal fidelity CE (row+col symmetric) | `model/prototype/losses.py:322` | `model/prototype/losses.py:572` | `use_loss_dir`/`use_loss_diag` | `lambda_dir` (`lambda_diag` alias) | Yes |
| `proto_fidelity_gap_loss` | Margin hinge on positive cosine minus hardest negative cosine | `model/prototype/losses.py:371` | `model/prototype/losses.py:583` | `use_loss_gap` | `lambda_gap` | Yes |
| `proto_support_loss` | Penalize effective support above target (over-diffuse routing) | `model/prototype/losses.py:394` | `model/prototype/losses.py:589` | `use_loss_sup`/`use_loss_support` and routing weights present | `lambda_sup` (`lambda_support` alias) | Yes |
| `proto_proxy_losses` (`image/text/text_exact`) | Proxy classification CE against class proxies | `model/prototype/losses.py:178` | Would be in `model/prototype/losses.py:546-548` | **Currently hard-disabled** by `proxy_losses_active=False` (`model/prototype/losses.py:535`) | `lambda_proxy_*` defined | No (current runtime) |
| `proto_align_loss` | Cosine alignment between source and target embeddings | `model/prototype/losses.py:311` | Not used for objective; `loss_align` forced zero at `model/prototype/losses.py:570` | N/A in current forward | `lambda_align` defined | No (current runtime) |
| `proto_diversity_loss` | Encourage orthogonal/diverse prototype bank | `model/prototype/losses.py:402` | Not used in current forward (`loss_diversity=zero` at `model/prototype/losses.py:590`) | N/A in current forward | `lambda_div` defined | No (current runtime) |
| `proto_balance_loss` | Encourage uniform routing usage across prototypes | `model/prototype/losses.py:411` | Not used in current forward (`loss_balance=zero` at `model/prototype/losses.py:591`) | N/A in current forward | `lambda_bal` defined | No (current runtime) |

Notes:
- Final optimization always backpropagates `outputs['loss_total']` in `processor/processor.py:367`.
- Legacy duplicate loss implementations also exist under `prototype/legacy/PAS-dropping/model/...` (see Section 6), but they are not imported by the active training entrypoints.

# 2. Total loss assembly

## 2.1 Scalar used by optimizer

- Training loop uses:
  - `outputs = model(...)` then `total_loss = outputs['loss_total']` (`processor/processor.py:366-367`)
  - `total_loss.backward()` (or AMP-scaled equivalent) (`processor/processor.py:370`, `processor/processor.py:380`)

So the only scalar optimized is `outputs['loss_total']`.

## 2.2 Host-only CLIP runtime (`model/hosts/clip.py`)

Call path:
- `train.py -> build_model -> build_clip_host` (`model/build.py:24-30`, `model/hosts/clip.py:459`)
- `ClipHostModel.forward` computes host loss and then:
  - `loss_total = lambda_host * host_loss_total` (`model/hosts/clip.py:380`)

Formula:
- `L_total = lambda_host * L_host_clip`
- If `retrieval_mode == clip_bidirectional`:  
  `L_host_clip = 0.5 * (CE(logits_i2t, diag_targets) + CE(logits_t2i, diag_targets))` (`model/hosts/clip.py:99-103`)
- Else (`surrogate_i2t`):  
  `L_host_clip = CE(logits_i2t, diag_targets)` (`model/hosts/clip.py:104-105`)

## 2.3 PAS prototype-enabled runtime (`model/pas_model.py`)

Call path:
- `train.py -> build_model -> pas_model.build_model` (`model/build.py:34-38`)
- `PASModel.forward`:
  - host branch: `host_outputs = self.host_head(...)` (`model/pas_model.py:1083`)
  - prototype branch: `prototype_outputs = self.prototype_head(...)` (`model/pas_model.py:1120`)
  - assembly: `loss_total = (lambda_host * host_losses['loss_total']) + prototype_losses['loss_total']` (`model/pas_model.py:1135`)

Core formula:
- `L_total = lambda_host * L_host + L_proto`

### Host term by host type

- If `host.type=itself`: `L_host = L_tal + L_cid` (`model/host_heads.py:797`)
- If `host.type=clip` (PAS host adapter): `L_host = L_clip_ret` where adapter forces `retrieval_mode='clip_bidirectional'` and `lambda_ret=1.0` (`model/host_heads.py:881-884`)

### Prototype term (current active objective)

From `model/prototype/losses.py:595-601`:

`L_proto = lambda_ret*L_ret + lambda_weight_ret*L_weight_ret + lambda_dir*L_dir + lambda_gap*L_gap + lambda_sup*L_sup`

Important:
- `L_proxy`, `L_align`, `L_diversity`, `L_balance` are currently not added (forced zero / disabled in current forward).

## 2.4 Stage/mode effects on objective

- `training.stage=stage0` requires `use_prototype_branch=false` (`model/pas_model.py:247-248`) -> host-only training path.
- `stage1/2/3/joint` are validated but do not change objective formula in active PAS code (`model/pas_model.py:309-316`; no stage-branch in forward).
- Freeze schedule can change effective lambdas and booleans during training (`processor/processor.py:287`, `utils/freeze_schedule.py:310`).

## 2.5 Concrete phase formulas from locked prototype configs

For:
- `configs/head_type/itself/direction1_optionB_locked_prototype.yaml`
- `configs/head_type/clip/direction1_optionB_locked_prototype.yaml`

Warmup phase (`lambda_host=0.0`, `lambda_ret=1.0`, `lambda_diag=1.0`, `lambda_bal/div>0`) at config lines `191-196`:
- Effective objective in current runtime:  
  `L_total = 1.0*L_ret + 1.0*L_dir + lambda_gap*L_gap + lambda_sup*L_sup`
- In these configs, `lambda_gap` is not overridden in phase and defaults to parser/config finalized value; `lambda_sup` follows config.
- `lambda_bal/lambda_div` currently do not affect `L_total` (see Section 6).

Joint locked phase (`lambda_host=1.0`, `lambda_ret=0.5`, `lambda_diag=0.5`) at config lines `208-213`:
- If ITSELF host:  
  `L_total = 1.0*(L_tal + L_cid) + 0.5*L_ret + 0.5*L_dir + lambda_gap*L_gap + lambda_sup*L_sup`
- If CLIP host:  
  `L_total = 1.0*L_clip_ret + 0.5*L_ret + 0.5*L_dir + lambda_gap*L_gap + lambda_sup*L_sup`
# 3. Detailed loss-by-loss analysis

## host_tal_loss

### Definition location
- File: `model/host_heads.py`
- Function: `compute_tal_components` (`142-171`)

### What it computes
- Normalizes image/text embeddings.
- Builds score matrix `S = T_norm @ I_norm^T`.
- Builds positive mask from equal `pid`.
- Computes bidirectional margin objective with soft positive assignment (`alpha`) and negative log-sum-exp term.
- Returns `L_tal = L_i2t + L_t2i` (both summed over batch, not mean).

### Inputs
- `image_features` `[B, D]`: from ITSELF host image branch (`global_image_embedding` / optionally grab features).
- `text_features` `[B, D]`: from ITSELF host text branch.
- `pid` `[B]`: from batch `pids` (dataset provides `pids`, `image_pids`, `caption_pids` in `datasets/bases.py:126-133`).

### Activation path
- `compute_host_losses = use_host_loss and pids is not None` (`model/host_heads.py:692`)
- `compute_tal = compute_host_losses and ("tal" in loss_names)` (`model/host_heads.py:693`)
- Optional second TAL term for grab branch when `itself_only_global=false` and grab embeddings exist (`model/host_heads.py:727-741`).

### Weighting
- No separate lambda; enters host loss additively with coefficient `1`.

### Contribution to total loss
- `host_loss_total = tal_loss + cid_loss` (`model/host_heads.py:797`)
- Then in PAS: `lambda_host * host_loss_total` (`model/pas_model.py:1135`)

### Notes / ambiguities
- Orientation is text-by-image in this implementation (`scores = text @ image^T`).

## host_cid_loss

### Definition location
- File: `model/host_heads.py`
- Main function: `_compute_cid_loss_components` (`559-635`)
- Helper CE functions: `compute_cid` (`242-244`), `compute_id` (`247-249`)

### What it computes
- Hard-negative mining from image-text similarity.
- Pair classification CE:
  - build positive + two negative pairs per anchor (`create_sample_pairs`)
  - MLP on concatenated features in both orders
  - `L_pair = 0.5*(CE(logits1, labels) + CE(logits2, labels))`
- ID classification CE:
  - `L_id_img = CE(image_logits, pids)`
  - `L_id_txt = CE(text_logits, pids)`
- `L_cid = L_pair + L_id_img + L_id_txt`

### Inputs
- `image_features` `[B, D]`, `text_features` `[B, D]` from ITSELF global/grab branches.
- `pids` `[B]` from batch.
- `mlp`, `classifier`, `classifier_id` modules created in ITSELFHostHead init (`model/host_heads.py:321-345`).
- Pair labels are expanded to roughly `3B` samples by `create_sample_pairs`.

### Activation path
- Requires:
  - `use_host_loss`
  - `pids` present
  - model in training mode (`self.training`)
  - `"cid"` in `itself_loss_names`
  - classifiers initialized
  (`model/host_heads.py:694-699`)
- Optional grab-branch CID when `itself_only_global=false` (`model/host_heads.py:768-783`).

### Weighting
- No separate lambda; added directly into host total.

### Contribution to total loss
- Added into host total at `model/host_heads.py:797`, then scaled by `lambda_host` in PAS total.

### Notes / ambiguities
- `_compute_cid_loss` wrapper (`model/host_heads.py:638`) is defined but not called in active path.

## host_clip_retrieval_loss (PAS clip host adapter path)

### Definition location
- File: `model/vanilla_clip.py`
- Functions: `VanillaClipLosses.retrieval_loss` (`77-96`), `VanillaClipLosses.forward` (`123-202`)

### What it computes
- `targets = [0..B-1]`.
- `L_i2t = CE(logits_i2t, targets)`.
- If bidirectional: `L_t2i = CE(logits_i2t^T, targets)`, `L_ret = 0.5*(L_i2t + L_t2i)`.

### Inputs
- `logits_i2t` `[B, B]`, produced from projected embeddings:
  - `logits_i2t = (I_proj @ T_proj^T) * logit_scale` (`model/vanilla_clip.py:69-73`)
- `I_proj`, `T_proj` come from host projectors in `VanillaCLIPHead` (`model/vanilla_clip.py:313-315`).

### Activation path
- PAS host builder for clip sets:
  - `use_loss_ret = use_host_loss`
  - `lambda_ret = 1.0`
  - `retrieval_mode = clip_bidirectional`
  (`model/host_heads.py:881-884`)

### Weighting
- Internal: `loss_total = lambda_ret * loss_ret` (`model/vanilla_clip.py:150`), with forced `lambda_ret=1.0`.
- External: PAS applies `lambda_host` (`model/pas_model.py:1135`).

### Contribution to total loss
- `L_total = lambda_host * L_host_clip + L_proto`.

### Notes / ambiguities
- In PAS clip-host path, retrieval mode is hard-forced to bidirectional regardless of config `retrieval_mode`.

## host_clip_retrieval_loss (host-only ClipHostModel path)

### Definition location
- File: `model/hosts/clip.py`
- Class: `_VanillaClipLoss` (`61-155`)

### What it computes
- Same CE structure as above:
  - i2t CE always when enabled
  - optional t2i CE when `retrieval_mode=clip_bidirectional`

### Inputs
- `image_embed`, `text_embed` `[B, D]` from CLIP backbone + host projectors (`model/hosts/clip.py:372-376`).
- Logits computed by normalized dot-product times fixed logit scale (`model/hosts/clip.py:83-87`).

### Activation path
- Host-only clip model path selected when `use_prototype_branch=false` and `host.type=clip` (`model/build.py:24-30`).
- Loss module enabled by `use_host_loss && use_loss_ret` in constructor (`model/hosts/clip.py:204`).

### Weighting
- `_VanillaClipLoss` returns `loss_total=loss_ret`.
- Model applies `lambda_host`: `loss_total = lambda_host * host_loss_total` (`model/hosts/clip.py:380`).

### Contribution to total loss
- Entire optimization objective in host-only clip mode.

### Notes / ambiguities
- Supports both `surrogate_i2t` and `clip_bidirectional` retrieval modes in this path.

## proto_proxy_losses (`loss_proxy_image`, `loss_proxy_text`, `loss_proxy_text_exact`)

### Definition location
- File: `model/prototype/losses.py`
- Functions: `proxy_loss` (`178-183`), `proxy_logits` (`171-176`)

### What it computes
- Normalize embeddings and class proxies.
- Compute logits against class proxies.
- Cross entropy with class id labels.

### Inputs
- Embeddings `[B, D]`:
  - image projected
  - surrogate text projected
  - exact text projected
- `pids` `[B]` as class labels.
- `class_proxies` `[num_classes, D]` learnable parameter (`model/prototype/losses.py:126`).

### Activation path
- In current code: forcibly disabled by `proxy_losses_active = False` (`model/prototype/losses.py:535`).
- Even if flags `use_loss_proxy_*` and lambdas are set, proxy components stay zero.

### Weighting
- Weights exist: `lambda_proxy_image`, `lambda_proxy_text`, `lambda_proxy_text_exact`.
- In current runtime they multiply zeros.

### Contribution to total loss
- No current contribution.

### Notes / ambiguities
- `disable_proxy_losses` argument exists in signatures (`model/prototype/losses.py:507`) but is not used in active forward logic.

## proto_align_loss

### Definition location
- `model/prototype/losses.py:311` (`cosine_alignment_loss`)

### What it computes
- `mean(max(0, 1 - cos(source_i, target_i)))` in current implementation.

### Inputs
- Would use source and target embeddings `[B, D]`.

### Activation path
- Current forward sets `loss_align = zero` directly (`model/prototype/losses.py:570`); method is not used for objective.

### Weighting
- `lambda_align` exists, but weighted align term is hard-zero (`model/prototype/losses.py:662`).

### Contribution to total loss
- No current contribution.

### Notes / ambiguities
- Alignment is effectively ablated in active prototype objective.

## proto_directional_diagonal_loss (`loss_dir` / `loss_diag`)

### Definition location
- `model/prototype/losses.py:322` (`symmetric_relative_diagonal_loss`)
- Wrappers: `directional_fidelity_loss` (`316`), `diagonal_fidelity_loss` (`319`)

### What it computes
- Normalize student (`surrogate`) and teacher (`exact`, detached).
- Similarity matrix `M = student @ teacher^T`.
- Targets are diagonal index.
- `L_row = CE(M / diag_temperature, targets)`, `L_col = CE(M^T / diag_temperature, targets)`
- `L_dir = 0.5*(L_row + L_col)` (for batch > 1).

### Inputs
- `surrogate_text_embeddings` `[B, D]`: from prototype surrogate reconstruction + text projector.
- `exact_text_embeddings` `[B, D]`: from exact text pooling + text projector.

### Activation path
- Objective uses this loss when `use_loss_dir` is true (`model/prototype/losses.py:571-573`).
- `use_loss_diag` is aliased to same switch (`model/prototype/losses.py:106`).

### Weighting
- `lambda_dir` (alias `lambda_diag`) via `loss_dir_weighted = lambda_dir * loss_dir` (`model/prototype/losses.py:592`).

### Contribution to total loss
- Included in `loss_total` (`model/prototype/losses.py:598`).

### Notes / ambiguities
- `diagonal_fidelity_loss` wrapper is used for debug certification metrics in `model/prototype/head.py:330,336`, but objective term uses `symmetric_relative_diagonal_loss` directly.

## proto_fidelity_gap_loss (`loss_gap`)

### Definition location
- `model/prototype/losses.py:371`

### What it computes
- Normalize surrogate and exact embeddings.
- For each sample, compute diagonal positive cosine and hardest off-diagonal negative cosine.
- Hinge: `max(0, margin - pos + hardneg)`, averaged.

### Inputs
- Same surrogate/exact projected embeddings `[B, D]` as above.

### Activation path
- Always computed as `gap_info`, applied if `use_loss_gap` (`model/prototype/losses.py:583-584`).

### Weighting
- `loss_gap_weighted = lambda_gap * loss_gap` (`model/prototype/losses.py:593`).

### Contribution to total loss
- Included at `model/prototype/losses.py:599`.

### Notes / ambiguities
- In many configs, `use_loss_gap` is left to parser default `true`.

## proto_support_loss (`loss_sup` / `loss_support`)

### Definition location
- `model/prototype/losses.py:394` (`support_loss`)

### What it computes
- Effective support per sample: `s = 1 / sum_n(alpha_n^2)` (`model/prototype/losses.py:391-392`)
- Penalty: `mean( max(0, (s-target)/target )^2 )`

### Inputs
- `routing_weights` `[B, N]`: from prototype router (`model/prototype/head.py:1120` or direct head empty in no-bank mode).

### Activation path
- `support_loss` itself returns zero unless `routing_weights` exists and `use_loss_sup` true (`model/prototype/losses.py:395`).
- Called unconditionally in forward (`model/prototype/losses.py:589`), gating happens inside function.

### Weighting
- `loss_sup_weighted = lambda_sup * loss_sup` (`model/prototype/losses.py:594`).
- Aliases: `use_loss_support`, `lambda_support`.

### Contribution to total loss
- Included at `model/prototype/losses.py:600`.

### Notes / ambiguities
- With direct head (`use_prototype_bank=false`), routing is empty and support term remains zero.

## proto_diversity_loss

### Definition location
- `model/prototype/losses.py:402`

### What it computes
- Frobenius-like penalty on prototype cosine similarity matrix against identity:
  - `|| normalize(P) normalize(P)^T - I ||_F^2` (implemented as elementwise square sum).

### Inputs
- `prototypes` `[N, D]`.

### Activation path
- Method exists, but current forward sets `loss_diversity = zero` without calling method (`model/prototype/losses.py:590`).

### Weighting
- `lambda_div` tracked and reported, but no active term in objective.

### Contribution to total loss
- No current contribution.

### Notes / ambiguities
- Freeze schedule can modify `lambda_div`/`use_diversity_loss`, but objective still ignores it in current forward.

## proto_balance_loss

### Definition location
- `model/prototype/losses.py:411`

### What it computes
- Uses mean routing usage per prototype and penalizes squared deviation from uniform usage.

### Inputs
- `routing_weights` `[B, N]`.

### Activation path
- Method exists, but current forward sets `loss_balance = zero` without calling method (`model/prototype/losses.py:591`).

### Weighting
- `lambda_bal` tracked and reported, but no active term in objective.

### Contribution to total loss
- No current contribution.

### Notes / ambiguities
- Freeze schedule can modify `lambda_bal`/`use_balance_loss`, but objective still ignores it in current forward.

## proto_surrogate_retrieval_loss (`loss_ret`)

### Definition location
- `model/prototype/losses.py:419`

### What it computes
- Row-wise CE on `surrogate_pairwise_logits` with diagonal targets:
  - `L_ret = CE(logits[B,B], targets=arange(B))`

### Inputs
- `surrogate_pairwise_logits` `[B, B]`, produced in prototype head:
  - bank path: `compute_surrogate_pairwise_logits` (`model/prototype/head.py:958`, called at `1093`)
  - direct path: `compute_pairwise_similarity` (`model/prototype/direct_head.py:494`)

### Activation path
- Requires `use_loss_ret` true (`model/prototype/losses.py:515`, `555`).

### Weighting
- `loss_ret_weighted = lambda_ret * loss_ret` (`model/prototype/losses.py:586`).

### Contribution to total loss
- Included at `model/prototype/losses.py:596`.

### Notes / ambiguities
- Prototype retrieval mode is effectively row-wise i2t only in active prototype objective.

## proto_weighted_surrogate_retrieval_loss (`loss_weight_ret`)

### Definition location
- `model/prototype/losses.py:445`

### What it computes
- Host margin per row:
  - `m_i = pos_i - hardneg_i` from host pairwise logits.
- Weight per sample:
  - `w_i = sigmoid((delta - m_i) / tau)`
  - Optional normalization to mean 1.
- Weighted CE-like diagonal objective:
  - `L = -mean( w_i * log_softmax(proto_logits_i)[i] )`

### Inputs
- `surrogate_pairwise_logits` `[B,B]` from prototype side.
- `host_pairwise_logits` `[B,B]` from host side (`model/pas_model.py:1128`).

### Activation path
- Requires `use_loss_weight_ret=true` and both logits tensors present (`model/prototype/losses.py:523-532`).

### Weighting
- `loss_weight_ret_weighted = lambda_weight_ret * loss_weight_ret` (`model/prototype/losses.py:588`).

### Contribution to total loss
- Included at `model/prototype/losses.py:597`.

### Notes / ambiguities
- For ITSELF host, host similarity matrix orientation is text-by-image (`model/host_heads.py:182`), while this loss assumes image-row semantics. Shape is valid (`[B,B]`), but semantic orientation appears potentially inconsistent.
# 4. Config-to-loss map

| Config field(s) | Where defined | Where read | Effect on loss |
|---|---|---|---|
| `host.itself_loss_names` (`tal+cid`) | configs + parser default (`utils/options.py:78`) | `ITSELFHostHead.__init__` (`model/host_heads.py:310-312`) | Chooses host ITSELF terms (`tal`, `cid`) |
| `objectives.objectives.use_host_loss` | config map (`utils/config.py:90`), parser (`utils/options.py:76`) | Host heads and freeze override (`model/host_heads.py:312`, `utils/freeze_schedule.py:321-333`) | Enables/disables host loss computation |
| `objectives.lambda.host` / `lambda_host` | parser (`utils/options.py:77`), config map (`utils/config.py:110`) | PAS/clip host total assembly (`model/pas_model.py:1135`, `model/hosts/clip.py:380`) | Multiplies host loss contribution |
| `objectives.objectives.use_loss_ret` + `objectives.lambda.ret` | parser (`utils/options.py:122,124`), config map (`utils/config.py:100,119`) | Prototype losses (`model/prototype/losses.py:515,586`) and host-only clip constructor (`model/hosts/clip.py:204`) | Enables retrieval CE and scales with `lambda_ret` |
| `use_loss_weight_ret`, `lambda_weight_ret`, `weight_ret_*` | parser (`utils/options.py:125-130`), config map (`utils/config.py:101,120,103-106`) | Prototype losses (`model/prototype/losses.py:523-532`, `445-482`) | Enables weighted retrieval term and weight-shaping behavior |
| `use_loss_dir`/`use_loss_diag`, `lambda_dir`/`lambda_diag`, `diag_temperature` | parser (`utils/options.py:111-121`), final aliasing (`utils/options.py:381-385`) | Prototype losses (`model/prototype/losses.py:571-573`, `592`, `356`) | Enables/scales diagonal fidelity objective |
| `use_loss_gap`, `lambda_gap`, `prototype_gap_margin` | parser (`utils/options.py:113-115`), config map (`utils/config.py:96,117`) | Prototype losses (`model/prototype/losses.py:583-584`, `593`, `371`) | Enables/scales fidelity gap hinge |
| `use_loss_sup`/`use_loss_support`, `lambda_sup`/`lambda_support`, `prototype_support_target`/`support_min` | parser (`utils/options.py:116-118`, `131-133`), alias finalization (`utils/options.py:386-389`) | Prototype losses (`model/prototype/losses.py:395`, `594`) | Enables/scales support regularizer |
| `use_loss_proxy_*`, `lambda_proxy*` | parser (`utils/options.py:102-108`) | Prototype losses module fields (`model/prototype/losses.py:75-83`) | Currently no effect on final objective because proxy path is hard-disabled (`model/prototype/losses.py:535`) |
| `use_loss_align`, `lambda_align` | parser (`utils/options.py:109-110`) | Prototype losses fields (`model/prototype/losses.py:82-83`) | Currently no effect on final objective (`loss_align` forced zero at `model/prototype/losses.py:570`) |
| `use_balancing_loss` + `lambda_bal` and `use_diversity_loss` + `lambda_div` | parser (`utils/options.py:179-183`), config map (`utils/config.py:107-108,121-122`) | Prototype fields + freeze overrides (`model/prototype/losses.py:71-74`, `utils/freeze_schedule.py:384-389`) | Flags/lambdas update state, but objective currently ignores both terms (`model/prototype/losses.py:590-591`) |
| `model.use_prototype_branch`, `model.use_prototype_bank`, `model.use_image_conditioned_pooling` | parser (`utils/options.py:138-140`), finalized (`utils/options.py:303-315`) | Model routing and head selection (`model/build.py:14-38`, `model/prototype/build.py:107-109`) | Chooses host-only vs PAS, and prototype-bank vs direct-head loss input path |
| `host.type` (`clip`/`itself`) | parser (`utils/options.py:75`), config map (`utils/config.py:43`) | Host head/model builder (`model/build.py:25-29`, `model/host_heads.py:867`) | Chooses host loss family (CLIP retrieval vs ITSELF TAL/CID) |
| `training.freeze_schedule[].loss_weights.*` | schedule schema + parser loaded into `args.freeze_schedule` (`utils/options.py:323-324`) | Applied each phase (`processor/processor.py:287`, `utils/freeze_schedule.py:310-391`) | Dynamically changes active lambdas and some `use_loss_*` switches during training |
| `training.stage` / `training_stage` | parser (`utils/options.py:192`) | PAS validation (`model/pas_model.py:247-248`, `309-316`) | Enforces stage0 host-only; no per-stage objective branch otherwise |
| `objectives.objectives.retrieval_mode` | parser (`utils/options.py:123`) | Host-only clip loss (`model/hosts/clip.py:203`) and PAS config validation (`model/pas_model.py:239-240`) | Controls CLIP host retrieval symmetry in host-only clip path; not used to switch prototype retrieval formula |

# 5. Case-by-case effective losses

## Case A: Host-only CLIP baseline (`configs/clip_vanilla_bidirectional.yaml`)

Proof path:
- `use_prototype_branch=false` (`config: line 33`) -> host-only model builder path (`model/build.py:24-30`).
- Clip host forward objective at `model/hosts/clip.py:380`.

Active losses:
- Host CLIP retrieval (`_VanillaClipLoss`), bidirectional in this config (`retrieval_mode: clip_bidirectional`, config line 104).

Inactive losses:
- All prototype losses (no prototype branch).
- ITSELF TAL/CID (host type is clip).

Formula:
- `L_total = lambda_host * 0.5*(CE_i2t + CE_t2i)` with `lambda_host=1.0` (config line 110).

## Case B: Prototype + ITSELF host (`configs/head_type/itself/direction1_optionB_locked_prototype.yaml`)

Proof path:
- `host.type=itself` (line 39), `use_prototype_branch=true` (line 33), `use_prototype_bank=true` (line 34).
- PAS total assembly in `model/pas_model.py:1135`.

Warmup phase (`training.freeze_schedule` lines 181-196):
- Active host term: disabled by `lambda_host=0.0`.
- Active prototype terms: `L_ret`, `L_dir`, plus any nonzero/default `L_gap` and `L_sup`.
- Inactive in current runtime: proxy/align/balance/diversity.
- Effective formula:  
  `L_total = 1.0*L_ret + 1.0*L_dir + lambda_gap*L_gap + lambda_sup*L_sup`

Joint locked phase (`lines 198-213`):
- Host term active: ITSELF `L_tal + L_cid`.
- Prototype active: `0.5*L_ret + 0.5*L_dir + lambda_gap*L_gap + lambda_sup*L_sup`.
- Effective formula:  
  `L_total = 1.0*(L_tal + L_cid) + 0.5*L_ret + 0.5*L_dir + lambda_gap*L_gap + lambda_sup*L_sup`

## Case C: Prototype + CLIP host (`configs/head_type/clip/direction1_optionB_locked_prototype.yaml`)

Proof path:
- `host.type=clip` (line 39), prototype enabled (lines 33-35).
- Host clip adapter uses `VanillaCLIPHead` with forced bidirectional retrieval (`model/host_heads.py:881-884`).

Warmup phase:
- `lambda_host=0.0` (line 192) -> host term zeroed.
- Prototype term same structure as Case B warmup.

Joint locked phase:
- `lambda_host=1.0` (line 209) -> host CLIP retrieval active.
- Formula:  
  `L_total = 1.0*L_host_clip + 0.5*L_ret + 0.5*L_dir + lambda_gap*L_gap + lambda_sup*L_sup`

## Case D: Prototype + ITSELF host, no freeze schedule (`configs/itself_prototype_from_legacy.yaml`)

Proof path:
- `use_prototype_branch=true` (line 33), `host.type=itself` (line 39), `training.stage=stage1` (line 172).
- No freeze schedule in config.

Active losses from config/runtime:
- Host: TAL + CID (subject to `itself_loss_names` and training mode).
- Prototype: `L_ret` (`use_loss_ret: true`, line 94), `L_dir` (`use_loss_diag: true`, line 93), default `L_gap` unless disabled, and support only if enabled.

Configured-but-inactive in current runtime:
- `use_balancing_loss: true` / `use_diversity_loss: true` (lines 102-103) with nonzero lambdas (lines 115-116), but objective code currently sets both terms to zero.

## Case E: Host-only ITSELF original adapter (`configs/itself_repro_from_original.yaml`)

Proof path:
- `host.type=itself` (line 39) and `use_prototype_branch=false` (line 33) -> `should_use_original_itself_runtime` true (`model/hosts/itself.py:38-42`), and `train.py` switches to original adapter training loop (`train.py:430-431`).

Active losses:
- Not confirmed from local code in this workspace because adapter source path expected at `adapter/WACV2026-Oral-ITSELF` (`model/hosts/itself.py:23`) is missing locally.

Inactive losses:
- PAS prototype losses (prototype branch disabled).

Formula:
- Not confirmed from code available in workspace.

## Case F: Prototype branch with direct head (`use_prototype_bank=false`, code path)

Proof path:
- `build_prototype_head` returns `DirectImageConditionedTextHead` when `use_prototype_bank` false (`model/prototype/build.py:107-109`).
- Loss call in `model/prototype/direct_head.py:504-515`.

Active losses:
- Same prototype loss module and formula as PAS bank path.

Key difference:
- `surrogate_text_embeddings` and `exact_text_embeddings` passed as same tensor (`exact_outputs['text_projected']`, lines `506-507`), so some fidelity terms become degenerate/easier.

## Stage labels in code

- `stage0`: enforced host-only (`model/pas_model.py:247-248`).
- `stage1/2/3/joint`: no objective branching in active forward; only validation/metadata (`model/pas_model.py:309-316`).

# 6. Dead code / suspicious loss logic

1. Proxy losses are defined but hard-disabled in current prototype objective.
- Evidence: `proxy_losses_active = False` (`model/prototype/losses.py:535`).
- Impact: `use_loss_proxy_*`, `lambda_proxy*`, and `disable_proxy_losses` do not affect optimized loss currently.

2. `loss_align` is defined/configurable but not used in objective.
- Evidence: `loss_align = zero` (`model/prototype/losses.py:570`) and weighted align also zero (`662`).

3. Diversity and balance losses are defined/configurable but not used in objective.
- Evidence: `loss_diversity = zero`, `loss_balance = zero` (`model/prototype/losses.py:590-591`).
- Even when phase overrides set `lambda_bal`/`lambda_div`, no objective contribution in current forward.

4. `disable_proxy_losses` plumbing is largely ineffective in active PAS path.
- Evaluator sets `disable_proxy_losses=True` (`processor/processor.py:88`).
- Passed through PAS/prototype heads (`model/pas_model.py:1130`, `model/prototype/head.py:1110`).
- Prototype loss forward accepts the arg (`model/prototype/losses.py:507`) but does not use it.

5. Orientation mismatch risk for weighted surrogate retrieval with ITSELF host logits.
- Weighted retrieval expects host logits with image rows (`model/prototype/losses.py:431-443`).
- ITSELF similarity helper is text-by-image (`model/host_heads.py:182`).
- Shape matches (`[B,B]`), but semantic row meaning appears inconsistent.

6. Stage labels (`stage1/2/3`) do not currently change objective behavior.
- Validated only (`model/pas_model.py:309-316`), no stage-dependent loss branch in forward.

7. Host ITSELF lambda placeholders appear unused in active runtime.
- `itself_lambda1_weight` / `itself_lambda2_weight` are mapped/configured (`utils/config.py:57-58`, many configs) but no active read found in runtime loss computation.

8. Legacy duplicate training/loss stack exists but is not wired into active entrypoints.
- Duplicate files under `prototype/legacy/PAS-dropping/model/...` define older objective composition (including proxy/align/div/bal in total).
- No active imports/references from current train/build path were found.

9. Host-only ITSELF loss internals cannot be fully traced in this workspace.
- Adapter path expected at `adapter/WACV2026-Oral-ITSELF` (`model/hosts/itself.py:23`) is absent locally.
- Status: definition references found, local call-site internals not found.

