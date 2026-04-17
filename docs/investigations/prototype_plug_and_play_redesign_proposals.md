# Prototype Plug-and-Play Redesign Proposals (Code-Grounded)

## 1. Design target
A plug-and-play prototype in this repository should behave as follows:
- Host baseline behavior is preserved when prototype is attached.
- Prototype can be attached/removed without host relearning.
- Integration mode is explicit (`external`, `partial`, `joint`) rather than implied by mixed flags.
- Inference semantics are explicit and stable: `host-only`, `prototype-only`, `fused`.
- Reporting/checkpoint selection does not silently rewrite which mode is being measured.

Why this target is necessary in current code:
- Current PAS forward is jointly optimized with shared representations (`model/pas_model.py:1079-1136`).
- Current stage behavior is phase-mutation driven (`processor/processor.py:283-295`).
- Current evaluation can remap `pas-t2i` selection from subset sweeps (`utils/metrics.py:565-580`).

## 2. Non-negotiable requirements
- Host baseline must remain recoverable and measurable as a first-class mode.
  - Evidence motivating this: host/prototype are currently mixed in one objective (`model/pas_model.py:1135`).
- Prototype must be disable-able without changing host graph/training behavior.
  - Evidence: runtime route changes with `use_prototype_branch` (`model/build.py:14-38`).
- Fusion must be safe by default and must not poison host ranking silently.
  - Evidence: direct additive fusion (`model/fusion.py:54-56`).
- Training/inference semantics must align (or mismatch must be explicit and documented).
  - Evidence: prototype training uses row-wise surrogate `[B,B]` CE (`model/prototype/losses.py:419-429`), inference ranks text-image matrices (`utils/metrics.py:556-561`).
- Stage semantics must be explicit in code and logs.
  - Evidence: `training_stage` label is not epoch router (`model/pas_model.py:309-316`; `processor/processor.py:270-295`).

## 3. Root causes that proposals must address
Only causes supported by audit evidence are listed.
- Joint objective coupling: `loss_total = lambda_host * host_loss + prototype_loss` (`model/pas_model.py:1135`).
- Shared representation coupling: prototype input tensors are from shared host/backbone outputs without detach barriers (`model/pas_model.py:1121-1124`).
- Fusion coupling: final ranking score is direct weighted sum (`model/fusion.py:54-56`).
- Runtime phase mutation coupling: `freeze_schedule` can reconfigure trainability/loss switches/optimizer on epoch boundaries (`processor/processor.py:283-295`; `utils/freeze_schedule.py:301-391`).
- Reporting/selection coupling: subset row selection can overwrite `pas-t2i` display/selection behavior (`utils/metrics.py:565-580`).

## 4. Proposal set
### Option A (Minimal-intrusion): External Add-on Mode with Detach Barrier + Strict Phase Guards
Short description:
- Add an explicit integration mode (for example `integration_mode=external_addon`) that keeps current PAS architecture but enforces host invariance by policy.

Which coupling it removes:
- Removes gradient coupling from prototype losses into shared host/backbone representations by feeding detached shared tensors into prototype path during `external_addon` training.
- Removes silent host-trainability drift in this mode by validating/failing phases that try to unfreeze host groups.
- Removes reporting ambiguity by forbidding `pas-t2i` overwrite when evaluating external mode.

Which coupling it keeps:
- Keeps inference-time score coupling (host+prototype fusion still possible).
- Keeps representational dependency (prototype still reads host/backbone features, but detached in external mode).

Exact code areas to change:
- `model/pas_model.py`:
  - `forward(...)` inputs to `prototype_head(...)` (`:1121-1124`) to use `.detach()` in external mode.
  - optional `loss_total` policy branch for external mode (host loss excluded from optimization in prototype-training phases).
- `processor/processor.py` + `utils/freeze_schedule.py`:
  - enforce host-group freeze constraints in external mode at phase activation (`processor/processor.py:283-295`; `utils/freeze_schedule.py:301-307`).
- `utils/options.py` / `utils/config.py`:
  - add explicit integration mode enum.
- `utils/metrics.py`:
  - disable `pas-t2i` metric overwrite from eval subsets in external mode (`utils/metrics.py:565-580`).

Change-surface assessment:
- Model graph: **No structural split**.
- Forward pass: **Yes**.
- Loss path: **Yes (mode-conditional total loss policy)**.
- Optimizer groups: **No new groups required**.
- Freeze schedule: **Yes (mode constraints/validation)**.
- Checkpoint logic: **Minor metadata addition**.
- Evaluator: **Yes (selection semantics guard)**.
- Fusion logic: **No formula change required**.

Backward compatibility:
- High, if default integration mode remains current behavior.

Risks:
- Prototype may underperform if detached host features are too restrictive.
- More explicit mode branching in forward/train may increase config complexity.

Expected benefit:
- Strong host safety improvement with relatively low disruption.
- First honest step toward plug-and-play claims.

Research cost: **Low-Medium**.
Engineering cost: **Low-Medium**.

### Option B (Structural decoupling): Two-Component Runtime (HostCore + PrototypePlugin)
Short description:
- Split PAS monolith into explicit host runtime and prototype plugin runtime with a narrow interface contract. Prototype training runs against frozen host artifacts by default.

Which coupling it removes:
- Removes model-graph co-location coupling.
- Removes accidental joint optimization by default (unless explicitly requested in `joint` mode).
- Makes attach/remove semantics explicit and enforceable at build time.

Which coupling it keeps:
- Keeps optional inference coupling through a dedicated fusion combiner.
- Keeps optional explicit joint mode (if configured).

Exact code areas to change:
- `model/build.py`: build explicit compositions (`HostCore`, `PrototypePlugin`, `ComposedRetriever`) instead of single PAS monolith.
- `model/pas_model.py`: split responsibilities into separate modules/interfaces (host encoder/retriever vs prototype plugin).
- `processor/processor.py`: separate training flows for `external` vs `joint`; avoid implicit coupling.
- `utils/module_group_registry.py` and `solver/build.py`: group ownership by component boundaries.
- `utils/modular_checkpoint.py`: checkpoint per component plus explicit compatibility checks across components.
- `utils/metrics.py`: evaluate each mode from explicit components without row overwrite side effects.

Change-surface assessment:
- Model graph: **Yes (major)**.
- Forward pass: **Yes (major)**.
- Loss path: **Yes (major)**.
- Optimizer groups: **Yes**.
- Freeze schedule: **Yes (reframed by mode)**.
- Checkpoint logic: **Yes (major)**.
- Evaluator: **Yes**.
- Fusion logic: **Potentially no formula change, but moved boundary**.

Backward compatibility:
- Medium-Low unless a compatibility adapter layer is added.

Risks:
- Highest integration risk and migration complexity.
- Potential checkpoint compatibility churn.

Expected benefit:
- Highest plug-and-play purity and explicitness.

Research cost: **Medium-High**.
Engineering cost: **High**.

### Option C (Calibration-first): Safe Additive Fusion without Training-Graph Changes
Short description:
- Keep current training graph, but treat prototype as inference-side additive signal with calibrated fusion and strict host-safe defaults.

Which coupling it removes:
- Reduces fusion poisoning risk and reporting ambiguity.
- Improves stability of host+prototype combination without changing host training dynamics.

Which coupling it keeps:
- Keeps shared representation and joint optimization coupling during training.
- Keeps stage mutation coupling in training loop.

Exact code areas to change:
- `model/fusion.py`: add calibrated combiner (for example per-branch temperature/affine or bounded gate) with host-safe defaults.
- `utils/metrics.py`: evaluate/report calibrated fused rows distinctly from raw rows; no silent `pas-t2i` overwrite.
- `utils/modular_checkpoint.py`: persist calibration artifact in `fusion` checkpoint group.
- `utils/options.py`/`utils/config.py`: add calibration config keys.

Change-surface assessment:
- Model graph: **No**.
- Forward pass: **Minimal**.
- Loss path: **No (if calibration is post-hoc)**.
- Optimizer groups: **Optional/minimal**.
- Freeze schedule: **No**.
- Checkpoint logic: **Minor**.
- Evaluator: **Yes**.
- Fusion logic: **Yes**.

Backward compatibility:
- High.

Risks:
- Does not make training truly plug-and-play.
- Can be mistaken as full modularity if not documented clearly.

Expected benefit:
- Fastest path to safer fused inference.

Research cost: **Low**.
Engineering cost: **Low**.

## 5. Required option types mapping
- Minimal-intrusion option: **Option A**.
- Structural decoupling option: **Option B**.
- Calibration-first option: **Option C**.

## 6. Compare proposals
| Criterion | Option A: External Add-on Mode | Option B: Two-Component Runtime | Option C: Calibration-first |
|---|---|---|---|
| Plug-and-play purity | Medium-High | Very High | Low-Medium |
| Host safety | High | Very High | Medium-High (inference), Low-Medium (training) |
| Prototype usefulness | Medium | High (long-term) | Medium |
| Implementation complexity | Medium | High | Low |
| Research novelty | Medium | High | Low-Medium |
| Code disruption | Medium | High | Low |
| Debugging difficulty | Medium | High | Low-Medium |
| Compatibility with current configs/checkpoints | High | Medium-Low | High |
| Faithfulness to plug-and-play principle | High (if enforced strictly) | Very High | Low |
| Likelihood of preserving host performance | High | Very High | Medium |
| Research risk | Medium | High | Low |

Ranking (for current codebase reality):
1. Option A
2. Option C
3. Option B

## 7. Best recommendation
Recommended now: **Option A (External Add-on Mode with Detach Barrier + Strict Phase Guards)**.

Why this is the best current choice:
- It addresses the two most damaging confirmed couplings quickly: shared-gradient coupling and silent phase drift (`model/pas_model.py:1121-1124,1135`; `processor/processor.py:283-295`).
- It preserves most of the current code path and checkpoint ecosystem.
- It enables honest plug-and-play claims under an explicit mode without requiring immediate large-scale refactor.

Why not Option C as the primary recommendation:
- Fusion-only fixes cannot remove training-time optimization/representation coupling.

Why not Option B first:
- It is the best end-state for modular purity, but near-term engineering and migration risk is highest.

## 8. Migration plan (for Option A)
1. Add explicit integration mode in config/args.
   - Change: `utils/options.py`, `utils/config.py` enum validation.
   - Keep unchanged: existing default mode behavior.
2. Add prototype input detach policy in PAS forward for external mode.
   - Change: `model/pas_model.py:1121-1124` data path.
   - Keep unchanged: host forward path and existing prototype module internals.
3. Add phase-guard validation for external mode.
   - Change: `processor/processor.py` phase activation, enforce host groups frozen.
   - Keep unchanged: current freeze_schedule parsing format.
4. Make evaluation semantics explicit for external mode.
   - Change: `utils/metrics.py` selection behavior to avoid silent `pas-t2i` overwrite.
   - Keep unchanged: sweep row generation itself.
5. Add mode-aware checkpoint metadata and reporting.
   - Change: `utils/modular_checkpoint.py` metadata field extension.

How to validate correctness:
- Verify host parameters remain unchanged in external mode by parameter-delta checks across epochs.
- Verify optimizer excludes host groups in external mode.
- Verify reported `val/top1_row` matches displayed row without remap ambiguity.

How to compare with current baseline:
- Run same seed/config with and without external mode guardrails; compare host-only row drift and fused row behavior.

Mandatory ablations:
- Detach on vs off under identical freeze schedule.
- External mode with host frozen vs current joint schedule.
- Subset-selection rewrite on vs off.

## 9. Validation plan
Required experiments to confirm true plug-and-play behavior:
1. Host-only baseline before integration vs after attaching prototype codepath (prototype disabled).
2. Attach prototype but disable fusion (`lambda_prototype=0`) and verify host metrics parity.
3. Attach prototype with inference-only fusion (no host updates from prototype training path).
4. Prototype trainable while host frozen (external mode) and compare host drift.
5. Host trainable while prototype frozen (frozen-transfer) and quantify host adaptation pressure.
6. Calibration-only fusion vs direct weighted fusion on identical host/prototype similarities.
7. Score-scale checks: logit-scale statistics per branch and post-fusion margin behavior.
8. Stage transition checks: optimizer-state continuity and newly-unfrozen parameter behavior (`processor/processor.py:178-214,289-295`).
9. Exact scorer vs approximate scorer sensitivity under same trained weights (`utils/metrics.py:509-538`).
10. Report all rows (`host`, `prototype`, `pas/fused`) and selected row source (`val/top1_row`, `val/top1_source_row`) to prevent selection masking (`utils/metrics.py:621-645`).

## 10. Final recommendation
- Implement Option A now to establish an explicit, enforceable external-add-on mode.
- Keep Option C as a complementary safety layer for fused inference.
- Do not claim true plug-and-play based on fusion changes alone.
- Plan Option B as a second-phase architectural cleanup once Option A validates host safety and preserves useful prototype gains.

## Appendix: direct answers to explicit questions
1. What prevents plug-and-play now?
- Shared forward tensors + shared objective + direct additive fusion (`model/pas_model.py:1079-1136`; `model/fusion.py:54-56`).
2. Most fundamental coupling?
- Shared-gradient optimization coupling via one total loss (`model/pas_model.py:1135`).
3. Easiest safe coupling to remove?
- Prototype gradient flow into shared host/backbone tensors in external mode (detach barrier at prototype inputs in `model/pas_model.py:1121-1124`).
4. Best redesign for host safety with acceptable usefulness?
- Option A now.
5. Can fusion-only changes achieve plug-and-play?
- No; structural/optimization decoupling is still required.
6. Minimum acceptable redesign for honest plug-and-play claim?
- Explicit external mode + host-freeze enforcement + detach barrier + unambiguous mode-specific reporting.
7. Reusable as-is parts?
- Host head, prototype head internals, evaluator row computation, modular checkpoint framework.
8. Must be rewritten for true modularity?
- Monolithic PAS orchestration (`model/pas_model.py`), mode routing in train loop, and selection semantics in evaluator.
9. What should training look like for true add-on?
- Host frozen/baseline-stable by default, prototype optimized externally, joint mode opt-in only.
10. What should inference look like for true add-on?
- Explicit deterministic modes: host-only, prototype-only, fused-calibrated; no silent row remapping.

Recommended next implementation step:
Implement an explicit `integration_mode=external_addon` path in `model/pas_model.py::forward` that passes detached shared features into `prototype_head` and enforce host-group freeze in `processor/processor.py` at phase activation, because this is the smallest code change that directly removes confirmed gradient coupling while keeping the current architecture intact.
