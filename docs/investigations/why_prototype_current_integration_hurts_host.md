# Why Prototype Current Integration Hurts Host

## 1. Executive summary
- The prototype branch is attached inside the same `PASModel` graph as the host; both branches consume the same CLIP backbone outputs from one forward pass (`model/pas_model.py:1079-1124`).
- Training is explicitly joint: `loss_total = (lambda_host * host_loss_total) + prototype_loss_total` (`model/pas_model.py:1133-1136`), so prototype optimization is not isolated from host optimization when shared modules are trainable.
- In the active config, stage transition is not a separate model; it is `freeze_schedule` mutating `requires_grad`, loss weights, and optimizer groups at epoch boundaries (`processor/processor.py:283-295`, `utils/freeze_schedule.py:301-391`, `configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml:185-220`).
- Prototype image features are explicitly host-derived plus routed prototype summary injection: `image_proxy_features = image_features + proto_query_proj(summary)` (`model/prototype/head.py:647-670`).
- Prototype retrieval training uses surrogate row-wise `[B,B]` logits (`model/prototype/losses.py:419-429`), while inference ranking uses fused text-to-image matrices (`utils/metrics.py:509-554`); the fused score itself is not directly trained.
- Fusion is a direct weighted sum, not calibration-only post-processing: `(lambda_host * host_similarity) + (lambda_prototype * prototype_similarity)` (`model/fusion.py:41-56`).
- With the active schedule, warmup freezes host then joint training reopens host with small host weight (`lambda_host=0.1`) and large prototype weights (`lambda_ret=5.0`, `lambda_diag=10.0`, `lambda_support=1.0`) (`configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml:217-220`).
- Static freeze flags are not final: `freeze_prototype_side` is applied at model build, then schedule phases can unfreeze selected prototype groups (`model/pas_model.py:327-348`, `utils/freeze_schedule.py:301-307`).

## 2. The exact integration map
### 2.1 Where prototype is attached
- `model/build.py::_build_model_impl` routes to PAS runtime when prototype branch is enabled (`model/build.py:33-38`).
- `PASModel.__init__` constructs one shared graph:
  - `self.base_model` (shared CLIP backbone) (`model/pas_model.py:47`)
  - `self.host_head` (`model/pas_model.py:95-99`)
  - `self.prototype_head` (`model/pas_model.py:100-112`)
  - `self.fusion_module` (`model/pas_model.py:113-118`)

### 2.2 Shared vs isolated components
| Component | Shared or isolated | Evidence |
|---|---|---|
| `base_model.visual`, `base_model.transformer`, `token_embedding`, `text_projection` | Shared | Host and prototype both consume `extract_image_features` / `extract_text_features` outputs in one forward (`model/pas_model.py:1079-1124`). |
| `host_head` (`CLIPHostAdapter -> VanillaCLIPHead` for `host.type=clip`) | Host-specific module, but fed by shared backbone outputs | Host branch encoding/loss path (`model/host_heads.py:251-292`, `model/vanilla_clip.py:269-346`). |
| `prototype_head` (`PrototypeConditionedTextHead`) | Prototype-specific module, but fed by shared backbone outputs and optional host logits | Prototype forward inputs include shared image/text features and `host_pairwise_logits` (`model/pas_model.py:1120-1129`, `model/prototype/head.py:1007-1111`). |
| `fusion_module` (`ResidualScoreFusion`) | Shared inference combiner | Takes host/prototype similarity matrices and returns weighted sum (`model/pas_model.py:871-886`, `model/fusion.py:32-56`). |

### 2.3 Coupled tensors/features/scores
- Shared features produced once:
  - `image_output.projected_pooled`, `image_output.projected_tokens`
  - `text_output` states/masks (`model/pas_model.py:1079-1080`, `730-800`)
- Host outputs:
  - `host_outputs['surrogate_pairwise_logits']` (host retrieval logits) (`model/vanilla_clip.py:315-333`)
- Prototype inputs:
  - `image_embeddings=image_output.projected_pooled`
  - `image_local_tokens=image_output.projected_tokens`
  - `text_token_states=_resolve_text_states(text_output)`
  - `host_pairwise_logits=host_outputs.get('surrogate_pairwise_logits')` (`model/pas_model.py:1120-1129`)
- Prototype branch internal coupling:
  - Routing summary from prototypes is injected into image proxy features (`model/prototype/head.py:667-670`)
  - Surrogate text is reconstructed as `einsum('bn,bnd->bd', routing_weights, basis_bank)` (`model/prototype/head.py:838-846`)

## 3. End-to-end training trace
### 3.1 Stage 1 (active config warmup: epochs 1-10)
Evidence: `configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml:187-201`.

Batch flow and updates:
1. At epoch start, active phase is resolved and applied (`processor/processor.py:283-287`).
2. Trainability is set by logical groups via prefix matching (`utils/freeze_schedule.py:292-307`, `utils/module_group_registry.py:8-44`).
3. Loss weights are overridden at runtime; `lambda_host=0.0` disables host loss switches (`utils/freeze_schedule.py:319-337`, `365-390`).
4. Optimizer is rebuilt from currently trainable params only; frozen params are excluded (`processor/processor.py:289-292`, `solver/build.py:172-205`, `model/pas_model.py:1015-1017`).
5. Forward still runs both branches (`model/pas_model.py:1083-1131`), but total loss is weighted as configured (`model/pas_model.py:1135`).
6. Backward is on `outputs['loss_total']` only (`processor/processor.py:366-381`).

What gets gradients in this stage:
- Trainable by config: `prototype_bank`, `prototype_projector`, `routing`, `fusion` (`configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml:190-191`).
- Frozen by config: `host_backbone`, `host_retrieval` (`configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml:191`).
- Important runtime detail: static `freeze_prototype_side=true` from config (`configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml:159`) is not final; warmup phase can re-enable prototype subsets (`model/pas_model.py:331,347-348` + `utils/freeze_schedule.py:301-307`).

### 3.2 Stage 2 (active config joint_adaptive: epochs 11-25)
Evidence: `configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml:203-220`.

Batch flow and updates:
1. Phase sets both host and prototype groups trainable (`configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml:206-208`).
2. Runtime loss overrides set `lambda_host=0.1`, `lambda_ret=5.0`, `lambda_diag=10.0`, `lambda_support=1.0` (`configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml:217-220`).
3. Total loss remains `lambda_host * host + prototype` (`model/pas_model.py:1135`).
4. Because prototype inputs are shared backbone features and are not detached (`model/pas_model.py:1121-1124`), prototype losses backprop through shared representations when host/backbone are unfrozen.

### 3.3 Stage logic and checkpoint/loading behavior
- `training_stage` is validated (`model/pas_model.py:309-316`) but not used as a per-stage training router in `do_train`; epoch behavior is `freeze_schedule`-driven (`processor/processor.py:270-295`).
- No automatic stage1->stage2 checkpoint handoff logic exists in `do_train`; phase transition happens in one run by mutating trainability/loss/optimizer in place (`processor/processor.py:283-295`).
- External loading is generic (`train.py:299-306`, `utils/modular_checkpoint.py:379-483`), not tied to `training_stage`.

## 4. End-to-end inference trace
1. Evaluator encodes all text/image features (`utils/metrics.py:472-494`).
2. For `retrieval_scorer=exact`, model computes:
   - host similarity via host head (`model/pas_model.py:865-869`, `888-894`)
   - prototype similarity via exact image-conditioned pairwise path (`model/pas_model.py:895-906`, `model/prototype/head.py:894-955`).
3. Fused similarity is computed by `fuse_retrieval_similarity` (`model/pas_model.py:871-886`) with linear fusion (`model/fusion.py:41-56`).
4. Evaluator also creates sweep rows (`host-t2i`, `prototype-t2i`, configured subsets) (`utils/metrics.py:399-461`).
5. If fusion lambdas are omitted in config and eval subsets exist, evaluator can select subset-based row for checkpoint metric and overwrite `pas-t2i` metrics row (`utils/metrics.py:194-203`, `565-610`).

Active-config implication:
- `fusion` block has no explicit `lambda_host/lambda_prototype` (`configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml:81-96`), so defaults are resolved by options finalization (`utils/options.py:337-356`).

## 5. Evidence for why this is not plug-and-play
### 5.1 Direct code evidence of non-modular coupling
- Joint optimization is hard-coded in one scalar objective (`model/pas_model.py:1135`).
- Prototype branch consumes host/backbone-derived tensors directly (`model/pas_model.py:1121-1124`).
- Prototype branch can also consume host logits (`host_pairwise_logits`) (`model/pas_model.py:1128`, `model/prototype/losses.py:523-558`).
- Prototype image representation explicitly adds routed prototype summary to adapted image features (`model/prototype/head.py:667-670`).

### 5.2 Hidden-failure-mechanism checklist (explicit status)
| Mechanism from request | Status | Code evidence |
|---|---|---|
| Prototype depends heavily on host features (not independent expert) | **Confirmed by code** | Prototype forward consumes shared image/text features (`model/pas_model.py:1121-1124`) and injects prototype summary into image proxy features (`model/prototype/head.py:667-670`). |
| Prototype training changes shared reps and can hurt host | **Confirmed by code (coupling), harm magnitude needs experiment** | Shared backbone outputs feed both branches (`model/pas_model.py:1079-1124`), and total loss includes prototype term (`1135`). |
| Fusion is not calibration-only; bad prototype can poison final score | **Confirmed by code** | Fusion is linear weighted sum (`model/fusion.py:41-56`). |
| Score scales are mismatched | **Supported but needs experiment** | Host and prototype each apply their own logit scales (`model/vanilla_clip.py:71-75`, `model/prototype/losses.py:149-156`), then are linearly fused (`model/fusion.py:56`). |
| Stage 2 retrains host to accommodate frozen prototype artifacts | **Supported by code; direct impact size needs experiment** | Frozen-transfer schedules keep prototype losses while freezing prototype-bank/routing and training host (`configs/head_type/clip/direction2_stage2_optionA_frozen_transfer.yaml:181-194`). |
| Prototype training objective misaligned with inference usage | **Confirmed by code** | Training uses surrogate row-wise `[B,B]` CE (`model/prototype/losses.py:419-429`), inference uses full text×image fused matrices (`utils/metrics.py:509-554`). |
| “Freeze prototype/train host” still coupled through shared outputs/loss paths | **Confirmed by code** | Freezing affects params only (`utils/freeze_schedule.py:292-307`); prototype losses still depend on shared input tensors (`model/pas_model.py:1121-1124`, `model/prototype/head.py:1043-1111`). |
| Host anchor pushed away from host optimum by prototype objective | **Supported but needs experiment** | Stage2 active config heavily upweights prototype losses vs host (`configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml:217-220`). |

### 5.3 What code disproves (important)
- Proxy and align losses are currently not active in `PrototypeLosses.forward` regardless config flags:
  - `proxy_losses_active = False` (`model/prototype/losses.py:534-553`)
  - `loss_align = zero` (`model/prototype/losses.py:570`)
- Therefore current host degradation is not explained by proxy-loss competition in this code path.

## 6. Root-cause analysis
### Confirmed by code
1. **Optimization coupling**: host and prototype are optimized in one objective scalar (`model/pas_model.py:1135`).
2. **Representation coupling**: both branches consume shared backbone outputs without detach barriers (`model/pas_model.py:1079-1124`).
3. **Fusion coupling**: final score is direct host/prototype weighted sum (`model/fusion.py:41-56`).
4. **Training/inference mismatch**: prototype is trained on surrogate row-wise retrieval but evaluated with fused matrix ranking (`model/prototype/losses.py:419-429`, `utils/metrics.py:509-554`).
5. **Stage-control reality**: practical stage behavior comes from `freeze_schedule` mutation (trainability/loss/optimizer), not `training_stage` router (`processor/processor.py:283-295`, `model/pas_model.py:309-316`).
6. **Freeze-policy override behavior**: static freezes at model build can be overridden each epoch by schedule (`model/pas_model.py:327-348`, `utils/freeze_schedule.py:301-307`).

### Supported but needs experiment
1. **Loss competition severity**: active schedule sets low host weight and high prototype weights (`configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml:217-220`), but exact contribution to host drop needs ablation.
2. **Scale/calibration mismatch impact**: independent branch scales plus linear fusion may destabilize ranking (`model/vanilla_clip.py:71-75`, `model/prototype/losses.py:149-156`, `model/fusion.py:56`).
3. **Stage-transition optimization mismatch**: optimizer rebuild preserves state where possible but newly unfrozen params start without prior optimizer state (`processor/processor.py:289-295`, `178-214`).

### Not supported by current code evidence
1. **BatchNorm running-stat drift in active clip-host path**: repo-wide `BatchNorm` appears only in ITSELF host module (`model/host_heads.py:37`), not CLIP host/projector path (`model/vanilla_clip.py:205-346`, `model/prototype/projector.py:15-57`).
2. **Proxy-loss conflict as current root cause**: proxy branch is hard-disabled in `PrototypeLosses.forward` (`model/prototype/losses.py:534-553`).

## 7. File-by-file evidence table
| File | Class/function | Role in coupling or degradation | Why it matters |
|---|---|---|---|
| `model/build.py` | `_build_model_impl` | Chooses PAS shared runtime when prototype enabled | Confirms prototype is attached in core model, not external plugin. |
| `model/pas_model.py` | `__init__` | Creates backbone + host + prototype + fusion together | Defines hard attachment point. |
| `model/pas_model.py` | `forward` | Runs shared feature extraction and joint loss combine | Primary source of optimization coupling. |
| `model/pas_model.py` | `named_optimizer_groups` | Groups trainable params by branch/function | Used by freeze schedule and optimizer rebuild. |
| `model/prototype/head.py` | `encode_image_branch`, `build_text_basis_bank`, `forward` | Builds routing, basis bank, surrogate/exact text and prototype losses | Shows prototype dependence on shared host/backbone features. |
| `model/prototype/losses.py` | `forward`, `surrogate_retrieval_loss`, `weighted_surrogate_retrieval_loss` | Defines prototype objective terms and optional host-logit weighting | Core loss coupling and objective mismatch evidence. |
| `model/fusion.py` | `ResidualScoreFusion.forward` | Final host/prototype score fusion | Confirms linear mixing behavior. |
| `processor/processor.py` | `do_train` | Applies phase trainability/loss overrides and rebuilds optimizer per phase | Shows practical stage behavior and update mechanics. |
| `utils/freeze_schedule.py` | `apply_phase_trainability`, `apply_loss_weight_overrides` | Mutates `requires_grad`, runtime loss weights, and loss-enable flags | Explains why schedule, not static flags, controls coupling. |
| `utils/module_group_registry.py` | `LOGICAL_MODULE_GROUP_PREFIXES` | Maps logical freeze/load groups to parameter prefixes | Determines what “freeze host/prototype/fusion” actually means. |
| `solver/build.py` | `build_optimizer` | Optimizer includes only `requires_grad=True` params | Verifies truly frozen params are excluded from optimizer updates. |
| `utils/metrics.py` | `_compute_similarity`, `_build_similarity_rows`, `eval` | Inference scoring, fusion sweep rows, subset-based selection | Shows inference behavior and reporting/selection side effects. |
| `utils/options.py` | `_finalize_args` | Resolves default fusion lambdas and runtime flags | Important for default fusion behavior when config omits lambdas. |
| `configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml` | `training.freeze_schedule`, `fusion.eval_subsets` | Active stage behavior and fusion sweep settings | Concrete run-level coupling policy. |

## 8. Config switches that most affect host degradation
| Config name | What it controls | Why it matters (code path) |
|---|---|---|
| `training.freeze_schedule[*].trainable_groups` / `frozen_groups` | Which logical groups can receive gradients | Applied each epoch via `apply_phase_trainability` (`utils/freeze_schedule.py:301-307`). |
| `training.freeze_schedule[*].loss_weights.lambda_host` | Host loss weight and host-loss enable switch | Mutates `model.lambda_host` and host loss flags (`utils/freeze_schedule.py:319-337`). |
| `training.freeze_schedule[*].loss_weights.lambda_ret/lambda_diag/lambda_support` | Prototype objective strength and enable switches | Runtime toggles prototype loss usage (`utils/freeze_schedule.py:341-390`). |
| `training.freeze_prototype_side` | Static initial freeze of full prototype head | Applied at model init, then may be overridden by schedule (`model/pas_model.py:331,347-348`, `utils/freeze_schedule.py:301-307`). |
| `fusion.lambda_host`, `fusion.lambda_prototype` (or omission) | Default inference fusion weights | Used by fusion module/evaluator; omission triggers defaults in options finalization (`utils/options.py:337-356`, `model/fusion.py:41-56`). |
| `fusion.eval_subsets` | Extra fusion sweep rows + selection behavior | Evaluator can select subset-based row and rewrite `pas-t2i` metrics (`utils/metrics.py:137-163`, `565-610`). |
| `evaluation.retrieval_scorer` | Exact vs approximate prototype inference path | Selects exact/approximate component computation (`utils/metrics.py:509-538`). |
| `prototype.routing_source` and `prototype.use_host_deflated_input` | Routing evidence source and host-deflation preprocessing | Alters routing dependence on global/local host-derived inputs (`model/prototype/head.py:564-637`, `519-547`). |
| `objectives.objectives.use_loss_weight_ret` + `weight_ret_detach_host` | Host-margin-conditioned weighting of prototype retrieval loss | Adds explicit host-score-conditioned optimization coupling (`model/prototype/losses.py:445-482`). |

## 9. What must be experimentally verified next
1. Quantify host-only metric drift across phase transition with prototype losses on vs off while keeping same trainable groups.
2. Compare current prototype training objective (surrogate row-wise) vs a fused-score-consistent training objective, without changing architecture.
3. Measure sensitivity to fusion scale mismatch by calibrating host/prototype scores before fusion at inference.
4. In frozen-transfer schedules, ablate prototype losses while keeping frozen prototype modules to isolate “host recovery to frozen prototype artifacts” effect.
5. Report both true default fusion row and subset-selected row to avoid metric-selection masking.

## 10. Final answer to the main question
The current prototype integration is not plug-and-play because the code integrates it as a **jointly optimized branch sharing backbone representations and participating in the same training objective**, then combines branch scores with direct weighted fusion at inference. In practice, host behavior is altered whenever shared modules are trainable and prototype losses are active (`model/pas_model.py:1079-1136`), and stage control further enforces this coupling through runtime freeze/loss/optimizer mutation (`processor/processor.py:283-295`). This is why adding/training the prototype branch can pull host performance down instead of behaving like an independent additive module.
