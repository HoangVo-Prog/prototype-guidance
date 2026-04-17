# Pre-Refactor True Plug-and-Play Implementation Audit

Evidence base:
- Target architecture decision: `docs/investigations/true_plug_and_play_target_architecture.md`
- Prior code audits: `docs/investigations/prototype_plug_and_play_full_pipeline_audit_v2.md`, `docs/investigations/prototype_plug_and_play_redesign_proposals_v2.md`
- Current code paths cited inline.

## 1. Executive summary

What is already settled:
- [Proven by code + prior design docs] The refactor target is explicit component split (HostCore / PrototypePlugin / Composer) with explicit modes, interface, metric authority, and checkpoint authority.
- [Proven by code] Current PAS runtime is monolithic and mixes host/prototype/composition semantics in one class (`model/pas_model.py:41-118`, `model/pas_model.py:1057-1135`).
- [Proven by code] Trainer semantics are phase/schedule-driven (`processor/processor.py:270-295`; `utils/freeze_schedule.py:301-391`).
- [Proven by code] Evaluator selection and displayed row can diverge (`utils/metrics.py:565-624`).

What is not settled yet (implementation-grade unknowns):
- Exact minimum HostCore -> PrototypePlugin interface surface per mode.
- Exact function-level ownership split and migration strategy (move/wrap/split/rewrite).
- Exact instrumentation plan to prove host isolation/parity claims during refactor.
- Exact evaluator/checkpoint authority rewrite boundaries so semantics stop being row-policy artifacts.
- Exact checkpoint schema and compatibility checks for separable component lifecycle.

What must still be investigated before refactor coding:
1. Interface minimization with mode-aware artifact necessity.
2. Function-by-function ownership/split map with risk levels.
3. Mode-driven trainer entrypoints and extraction of reusable utilities.
4. Proof-obligation instrumentation insertion plan.
5. Metric authority and checkpoint authority control-flow rewrite plan.
6. Component checkpoint schema and mixed-source compatibility guards.

## 2. Interface minimization audit

### 2.1 Current prototype input contract (as implemented)
- Training-time prototype forward currently accepts:
  - `image_embeddings`, `text_token_states`, `token_ids` (required signature)
  - `image_local_tokens`, `attention_mask`, `special_token_positions`, `host_pairwise_logits` (optional signature)
  - Source: `PrototypeConditionedTextHead.forward` (`model/prototype/head.py:1007-1019`).
- PAS passes all of these from live host/backbone outputs in one forward call (`model/pas_model.py:1120-1129`).

### 2.2 Artifact-level minimization table
| Artifact | Current source (file/function) | Current consumer (file/function) | Required for which mode | Detached snapshot possible? | Can be removed from interface? | Coupling risk if kept |
|---|---|---|---|---|---|---|
| `image_embeddings` (`projected_pooled`) | `model/pas_model.py::forward` (`:1121`) | `model/prototype/head.py::forward` (`:1009,1021`) | `prototype_only`, `fused_external`, `joint_training` | Yes | No (core prototype image path) | Medium (representation dependence) |
| `image_local_tokens` (`projected_tokens`) | `model/pas_model.py::forward` (`:1122`), retrieval encoder (`:810`) | `model/prototype/head.py::forward` (`:1012,1023`), routing path | Needed when local routing enabled; optional in simplified modes | Yes | Potentially yes if local routing disabled | Medium-High (interface widening, memory) |
| `text_token_states` | `model/pas_model.py::forward` (`:1123`), retrieval encoder (`:837`) | `model/prototype/head.py::forward` (`:1010,1028,1035,1047`) | `prototype_only`, `fused_external`, `joint_training`; exact scorer | Yes | Not for current exact prototype scoring | Medium |
| `token_ids` | `model/pas_model.py::forward` (`:1124`), retrieval encoder (`:838`) | `model/prototype/head.py::forward` (`:1011,1030,1036,1048`) | All current prototype scoring modes | Yes | Unclear without changing token masking/pooling logic | Medium |
| `attention_mask` | `model/pas_model.py::forward` (`:1126`), retrieval encoder (`:839`) | `model/prototype/head.py::forward` (`:1014,1031,1038,1049`) | Required when token policy/mask semantics active | Yes | Possibly if token policy becomes mask-free (not current) | Low-Medium |
| `special_token_positions` | `model/pas_model.py::forward` (`:1127`), retrieval encoder (`:840`) | `model/prototype/head.py::forward` (`:1015,1032,1039,1050`) | Needed by text preparation paths using special token contracts | Yes | Possibly with simplified pooling policy | Low-Medium |
| `host_pairwise_logits` | `model/pas_model.py::forward` (`:1128`) from host output | `model/prototype/head.py::forward` -> `model/prototype/losses.py` (`:1108`, `losses.py:523-531`) | Only when `use_loss_weight_ret=true` | Yes (current option via `weight_ret_detach_host`) | Yes for true external mode (recommended) | High (semantic coupling path) |
| `routing_weights` | Produced inside prototype branch (`model/prototype/head.py:1021-1027`) or host image branch (`model/pas_model.py:805-807`) | Approximate scoring path (`model/pas_model.py:931`) and prototype losses (`model/prototype/losses.py:589`) | Needed for approximate scorer and support loss | Yes | Not if approximate scorer/support loss kept | Medium |
| `summary` | Produced in host/prototype image branch (`model/pas_model.py:805,813`) | Exact pairwise prototype similarity (`model/pas_model.py:898`; `model/prototype/head.py:986`) | Needed for exact prototype score path | Yes | Not without changing exact similarity computation | Medium |
| `basis_bank` | `model/pas_model.py::encode_text_basis_for_retrieval` (`:851-863`) | Approximate prototype similarity (`model/pas_model.py:932`; `model/prototype/head.py:850`) | `retrieval_scorer=approximate` only | Yes | Yes if approximate mode dropped | Medium |
| `host_text_features` / `host_text_projected` | `model/pas_model.py::encode_text_basis_for_retrieval` (`:859-862`) | Host similarity fallback in approximate components (`model/pas_model.py:922-928`) | Approximate scorer host side only | Yes | Possibly, if host score computed from separate HostCore call | Low-Medium |

### 2.3 Inputs that are convenience dependencies vs necessities
- [Proven by code] `host_pairwise_logits` is convenience/optional dependency tied to `use_loss_weight_ret` path (`model/prototype/losses.py:523-531`), not mandatory for core prototype forward.
- [Proven by code] `image_local_tokens` is optional in signature (`model/prototype/head.py:1012`) and needed only for local routing-enabled behavior.
- [Inferred from control flow] `attention_mask` and `special_token_positions` are functionally required for current token-policy correctness in prototype text processing, but could be reducible only with explicit pooling-policy simplification.

### 2.4 Live-tensor assumptions that must be surfaced before split
- [Proven by code] PAS currently passes live tensors directly from encoder outputs to prototype (`model/pas_model.py:1121-1127`).
- [Proven by code] No explicit interface object/version exists in current path; artifacts are ad-hoc dict fields in encode/retrieval methods (`model/pas_model.py:802-863`).

## 3. Ownership and code split audit

### 3.1 File/function mapping table
| Current file | Symbol | Current responsibility | Target component | Move strategy | Migration risk | Why |
|---|---|---|---|---|---|---|
| `model/pas_model.py` | `encode_image_for_retrieval` (`:802`) | Builds host retrieval image features + optional prototype image features | Split: HostCore + PrototypePlugin adapter | Split | High | Returns mixed host/prototype fields in one dict (`host_image_projected`, `prototype_image_projected`, `summary`, `routing_weights`). |
| `model/pas_model.py` | `encode_text_for_retrieval` (`:829`) | Builds host text features and prototype text artifacts | Split: HostCore + Interface exporter | Split | High | Host and prototype artifact concerns are mixed. |
| `model/pas_model.py` | `encode_text_basis_for_retrieval` (`:845`) | Approximate scorer basis build + host text output | PrototypePlugin + HostCore interface | Split | High | Combines host and prototype-basis semantics. |
| `model/pas_model.py` | `compute_retrieval_similarity_components` (`:888`) | Computes host and prototype similarity in one method | HostCore + PrototypePlugin + Composer orchestrator | Split | High | Hard-codes host-first compute and optional prototype branch. |
| `model/pas_model.py` | `compute_approximate_retrieval_similarity_components` (`:915`) | Approximate host/prototype component scoring | HostCore + PrototypePlugin | Split | High | Pulls host and prototype approximate paths into one API. |
| `model/pas_model.py` | `fuse_retrieval_similarity` (`:871`) | Delegates to fusion module | Composer | Move as-is then wrap | Low | Already clean scorer-composition function. |
| `model/pas_model.py` | `forward` (`:1057`) | End-to-end training forward, host+prototype+joint loss | Mode-specific orchestrator | Rewrite | Very High | Core monolith glue point; mixes all semantics. |
| `model/host_heads.py` | `CLIPHostAdapter` (`:251`) | Host-only image/text encoding and similarity | HostCore | Move mostly as-is | Low-Medium | Already host-scoped adapter. |
| `model/host_heads.py` | `ITSELFHostHead` (`:295`) | Alternate host implementation | HostCore | Move mostly as-is | Medium | Separate host type path must retain compatibility. |
| `model/vanilla_clip.py` | `VanillaCLIPHead` + losses | Host retrieval scoring/loss | HostCore | Move mostly as-is | Low-Medium | Good internal cohesion for HostCore training. |
| `model/prototype/head.py` | `PrototypeConditionedTextHead.forward` (`:1007`) | Prototype branch encode/score/loss wiring | PrototypePlugin | Move mostly as-is with interface signature update | Medium | Inputs currently assume live tensors and optional host logits. |
| `model/prototype/losses.py` | `PrototypeLosses.forward` (`:496`) | Prototype objectives incl. weighted host-logit path | PrototypePlugin | Move with mode-guarded options | Medium-High | Hidden coupling via `host_pairwise_logits`. |
| `model/prototype/build.py` | `build_prototype_head` (`:10`) | Prototype construction from args | PrototypePlugin factory | Move mostly as-is | Low-Medium | Configuration dense but localized. |
| `model/fusion.py` | `ResidualScoreFusion.forward` (`:32`) | Score composition | Composer | Move mostly as-is | Low | Already isolated combiner. |
| `processor/processor.py` | `do_train` (`:223`) | Phase-driven training + eval + ckpt selection | Mode-specific trainer/orchestrator | Split + rewrite | Very High | Currently owns semantics via schedule and selected row forwarding. |
| `utils/freeze_schedule.py` | phase parsing/apply (`:145`, `:301`, `:310`) | Schedule parser + semantic mutator | Utility only (optional) | Wrap/restrict | High | Must stop defining mode semantics. |
| `utils/metrics.py` | `_compute_similarity`, `_build_similarity_rows`, `eval` (`:399-624`) | Computes rows + chooses authoritative row | Evaluator reporter only | Rewrite selection policy | High | Currently mixes reporting and semantic selection. |
| `utils/modular_checkpoint.py` | save/load manager (`:165+`) | Group ckpt save/load with single metric row | Component checkpoint lifecycle | Reuse + extend | Medium-High | Good scaffolding; lacks strict component compatibility/authority gates. |
| `solver/build.py` | `build_optimizer` (`:172`) | Optimizer from named groups | Reusable utility | Move as-is with per-component wrappers | Low-Medium | Generic if components expose `named_optimizer_groups`. |
| `solver/lr_scheduler.py` | `LRSchedulerWithWarmup` (`:7`) | Scheduler policy | Reusable utility | Move as-is | Low | Mode-agnostic scheduler class. |
| `model/build.py` | `_build_model_impl` (`:33`) | Host-only vs PAS routing | New mode router | Rewrite | Medium | Must evolve from prototype-flag routing to explicit runtime mode. |

### 3.2 Reusable scaffolding vs monolith glue
Reusable scaffolding (high confidence):
- `solver/build.py` optimizer/scheduler builders.
- `solver/lr_scheduler.py` scheduler implementation.
- `model/fusion.py` combiner primitive.
- `model/prototype/build.py` factory skeleton.
- `utils/module_group_registry.py` group-prefix utilities (with remap updates).

Monolith glue that must be broken:
- `PASModel.forward` combined loss and branch invocation.
- Mixed retrieval feature assembly methods in PASModel.
- `processor.do_train` phase activation as semantic controller.
- Evaluator row-selection logic as authority source.

## 4. Runtime and trainer disentanglement audit

### 4.1 What must stop being schedule-driven
- [Proven by code] Current phase activation mutates trainability and loss semantics at epoch boundaries (`processor/processor.py:283-295`; `utils/freeze_schedule.py:310-391`).
- Target requirement: runtime mode must define semantics; schedule may only tune LR/within-mode freezes.

Dangerous behaviors if kept unchanged:
1. Loss semantics change via `apply_loss_weight_overrides` (e.g., toggling `use_loss_ret`, `use_loss_weight_ret`, `lambda_host`) (`utils/freeze_schedule.py:319-390`).
2. Optimizer/scheduler reset each phase with state copy (`processor/processor.py:289-295`, `:178-214`) can silently alter training dynamics when modes should be stable.
3. "No phase covers epoch" fallback preserves previous state (`processor/processor.py:348-353`), which is unsafe for mode contracts.

### 4.2 What can remain in `utils/freeze_schedule.py`
Can remain as utility:
- Phase schema parsing/validation (`parse_freeze_schedule_config`, `FreezePhase`) for optional within-mode schedules.
- Group-level `requires_grad` toggling helper (`apply_phase_trainability`) if constrained by mode.
- LR override helper (`apply_optimizer_lr_overrides`) for optional schedule knobs.

Must no longer define semantics:
- Loss-objective switching (`apply_loss_weight_overrides`) as primary mode controller.

### 4.3 Optimizer builder reusability
- [Proven by code] `build_optimizer` is reusable if each component exposes correct `named_optimizer_groups` (`solver/build.py:172-205`).
- [Proven by code] It already validates duplicates and missing trainable params (`solver/build.py:101-150`).
- Risk: each new component must define group names consistently with config surface.

### 4.4 Recommended target trainer entrypoints
Required entrypoints:
1. `train_host_core(...)`
   - Depends on: host model forward/loss, optimizer, scheduler, host evaluator.
2. `train_prototype_external(...)`
   - Depends on: frozen HostCore export API + PrototypePlugin forward/loss + optimizer.
3. `train_composer_calibration(...)`
   - Depends on: frozen HostCore+PrototypePlugin scores + Composer objective.
4. `train_joint(...)` (optional)
   - Depends on explicit joint orchestrator; separate claim path.

Current code dependencies each entrypoint can reuse:
- Batch loop, AMP context, grad scaler, metric logging from `processor/processor.py`.
- Optimizer/scheduler from `solver/build.py` and `solver/lr_scheduler.py`.
- Scalar metric collection from `utils/metric_logging.py` flow invoked in processor.

## 5. Proof-obligation instrumentation audit

### 5.1 Host gradient isolation
| Obligation | Exact check | Insertion point | Existing support | Missing instrumentation | Difficulty |
|---|---|---|---|---|---|
| Host gradient isolation | Assert grad norm for host prefixes is zero in external modes | After backward, before optimizer step in trainer loop | `_collect_gradient_metrics` already computes host/backbone grad norms (`processor/processor.py:136-153,381-383`) | Hard assertion + mode-tagged failure logs | Low |

### 5.2 Optimizer isolation
| Obligation | Exact check | Insertion point | Existing support | Missing instrumentation | Difficulty |
|---|---|---|---|---|---|
| Optimizer isolation | Assert optimizer param ids do not intersect HostCore param ids in external prototype/composer training | Right after optimizer construction | `build_optimizer` validates duplicates/missing among trainables (`solver/build.py:101-150`) | Explicit component-ownership assertion utility | Medium |

### 5.3 Host parameter delta isolation
| Obligation | Exact check | Insertion point | Existing support | Missing instrumentation | Difficulty |
|---|---|---|---|---|---|
| Host parameter delta isolation | Snapshot host params at start/end epoch; assert delta == 0 (or strict tol) in external modes | Trainer epoch boundaries | No direct host-delta check exists | Param snapshot/hash utility + report in metrics | Medium |

### 5.4 Host output parity
| Obligation | Exact check | Insertion point | Existing support | Missing instrumentation | Difficulty |
|---|---|---|---|---|---|
| Host output parity | Compare host-only similarity outputs/metrics against baseline checkpoint on fixed eval split | Eval hook after each epoch and attach/remove tests | Evaluator already computes `host-t2i` row when components available (`utils/metrics.py:413-415`) | Baseline reference loader + deterministic parity harness + threshold policy | Medium-High |

### 5.5 Attach/remove parity
| Obligation | Exact check | Insertion point | Existing support | Missing instrumentation | Difficulty |
|---|---|---|---|---|---|
| Attach/remove parity | Run HostCore with plugin attached vs detached; host outputs and optimizer semantics must match in host-only mode | Dedicated integration test + startup checks | Build routing already supports host-only path (`model/build.py:24-38`) | Automated attach/detach parity test pipeline | Medium |

### 5.6 Row provenance correctness
| Obligation | Exact check | Insertion point | Existing support | Missing instrumentation | Difficulty |
|---|---|---|---|---|---|
| Row provenance correctness | Ensure selected metric row and source row are explicit and consistent with authority policy | Evaluator finalization + trainer checkpoint callsite | `val/top1_row` and `val/top1_source_row` currently logged (`utils/metrics.py:623-624`) | Hard validation that saved authority row matches mode policy | Medium |

### 5.7 Checkpoint authority correctness
| Obligation | Exact check | Insertion point | Existing support | Missing instrumentation | Difficulty |
|---|---|---|---|---|---|
| Checkpoint authority correctness | For each component save, assert metric row belongs to allowed authority rows | Before `save_best_if_improved` and inside checkpoint manager | `metric_row` is already passed to save methods (`processor/processor.py:478-491`; `utils/modular_checkpoint.py:327-355`) | Component-aware authority gate + reject-on-violation | Medium |

## 6. Metric authority and evaluator rewrite audit

### 6.1 Current control flow (detailed)
1. Similarity components are computed in evaluator (`utils/metrics.py:_compute_similarity`, `:463-554`).
2. Row set is built via `_build_similarity_rows`, always including `pas-t2i`, host row, optional prototype row, subsets, default pair (`utils/metrics.py:399-461`).
3. Row selection chooses `metrics` row using subset policy or fallback policy (`utils/metrics.py:565-604`).
4. In subset mode, selected subset row can overwrite `pas-t2i` metric values (`utils/metrics.py:573-580`).
5. Display row can be forced to `pas-t2i` while source row differs (`utils/metrics.py:606-610`).
6. `val/top1_row` and `val/top1_source_row` are recorded (`utils/metrics.py:623-624`).
7. Trainer forwards `val/top1_row` into checkpoint save calls (`processor/processor.py:478-491`).

### 6.2 Divergence points that can break authority
- Row label vs source row divergence (`selected_display_row` vs `selected_source_row`) (`utils/metrics.py:606-624`).
- Subset-driven overwrite of `pas-t2i` values (`utils/metrics.py:573-580`).
- Checkpoint manager currently receives only one row string, no component-specific authority logic (`processor/processor.py:478-491`; `utils/modular_checkpoint.py:327-355`).

### 6.3 Rewrite plan (reporting vs semantics)
- Evaluator should:
  - compute rows and metrics,
  - publish full row table + provenance,
  - not decide cross-component authority.
- Trainer/orchestrator should:
  - select authoritative row by active mode/component,
  - pass explicit authority row into component checkpoint saves.
- Checkpoint manager should:
  - enforce authority policy per component,
  - reject save attempts with disallowed rows.

### 6.4 Required authority mapping
- HostCore best-save: host-only row only.
- PrototypePlugin best-save: prototype-only row only.
- Composer best-save: fused row only.

### 6.5 Current behaviors to forbid
1. Rewriting `pas-t2i` metrics from subset row for authority decisions.
2. Passing display row when source row differs for checkpoint authority.
3. Excluding host-only rows globally from eligibility in modes where host authority should be primary.

## 7. Checkpoint compatibility investigation

### 7.1 Current reusable machinery
Reusable utilities:
- Group extraction/loading utilities: `get_group_state_dict`, `load_group_state_dict` (`utils/module_group_registry.py:95-103`).
- Modular save/load flow and payload wrappers (`utils/modular_checkpoint.py:206-355,379-475`).
- Existing group membership mapping as starting scaffold (`utils/module_group_registry.py:48-55`).

### 7.2 Current gaps for true component lifecycle
- [Proven by code] Payload metadata is minimal (`host_type`, `run_name`, `training_stage`) (`utils/modular_checkpoint.py:229-233`).
- Missing:
  - interface version compatibility,
  - component semantic version,
  - required counterpart checkpoints,
  - mode-specific authority metadata,
  - explicit score-schema version for Composer.

### 7.3 Proposed checkpoint schema (grounded extension)
HostCore checkpoint metadata additions:
- `component=host_core`
- `interface_version_export`
- `host_arch_fingerprint`
- `authoritative_metric_row=host-t2i`

PrototypePlugin checkpoint metadata additions:
- `component=prototype_plugin`
- `interface_version_min/max`
- `prototype_arch_fingerprint`
- `requires_host_arch_fingerprint` (optional strict)
- `authoritative_metric_row=prototype-t2i`

Composer checkpoint metadata additions:
- `component=composer`
- `expected_host_score_schema_version`
- `expected_prototype_score_schema_version`
- `authoritative_metric_row=fused-*`

### 7.4 Mandatory compatibility checks
1. Component type matches load target.
2. Interface version compatibility (host export vs prototype requirement).
3. Score schema compatibility for Composer.
4. Authority row compatibility for best-save policy.
5. Optional strict fingerprint compatibility when reproducibility mode is enabled.

### 7.5 Reusable vs rewrite summary
- Reuse directly: payload read/write plumbing, group state extraction, history rotation, load diagnostics.
- Rewrite/extend: metadata schema, compatibility validator, component authority gate, mode-aware save routing.

## 8. Concrete unknowns that still require experiment
(Only items not answerable by code inspection)
1. Whether removing `image_local_tokens` from external interface materially hurts prototype retrieval quality.
2. Whether detached `text_token_states` alone preserve enough prototype utility across datasets.
3. Whether approximate scorer (`basis_bank` path) remains beneficial after interface reduction.
4. Whether composer calibration can recover fused gains after strict authority and decoupling constraints.

## 9. Final pre-refactor checklist

What is already known:
- Monolith coupling points and control-flow risks are identified (`model/pas_model.py`, `processor/processor.py`, `utils/metrics.py`, `utils/modular_checkpoint.py`).
- Reusable scaffolding and monolith glue boundaries are mapped.

What must be investigated before coding:
1. Final minimal interface matrix by mode and routing configuration.
2. Component-aware metric authority policy spec with exact allowed rows.
3. Checkpoint metadata schema and compatibility validator spec.
4. Proof-obligation assertion policy (hard fail vs warning) per mode.

What can be coded immediately:
1. Add explicit instrumentation hooks for gradient/optimizer/param-delta parity checks.
2. Introduce component-aware checkpoint metadata fields (non-breaking additive).
3. Add mode-aware authority validation utilities (initially dry-run).

What must be guarded by tests from day 1:
1. Host gradient isolation in external modes.
2. Optimizer membership isolation by component.
3. Host parameter delta = 0 in external prototype/composer runs.
4. Host output parity under attach/remove.
5. Row provenance consistency (`top1_row` vs `top1_source_row`) and authority enforcement.
6. Checkpoint load compatibility acceptance/rejection cases.

## Smallest safe coding start
Single best first implementation step:
- Implement **component-aware metric/checkpoint authority validation in the existing evaluator->trainer->modular-checkpoint path** (without moving model code yet).

Why this minimizes refactor risk:
- It addresses the highest semantic-risk area (measurement and best-save correctness) before structural changes.
- It is additive and can reuse current plumbing (`utils/metrics.py:623-624`, `processor/processor.py:478-491`, `utils/modular_checkpoint.py:327-355`).
- It immediately prevents false positives during subsequent HostCore/PrototypePlugin/Composer split work.
