# Prototype Plug-and-Play Redesign Proposals V2 (Claim-Honest, Code-Grounded)

## 1. Design target (for this repository)
A true plug-and-play prototype architecture in this codebase must satisfy all of the following simultaneously:
1. Host baseline behavior remains recoverable and stable when prototype is attached.
2. Prototype can be attached/removed without forcing host relearning.
3. Integration semantics are explicit (`external`, `partial`, `joint`) and enforced by runtime contracts.
4. Inference semantics are explicit and unambiguous (`host-only`, `prototype-only`, `fused`).
5. Reporting/checkpoint selection cannot hide host degradation behind fused gains.

Motivation from current code:
- Joint objective and shared forward tensors (`model/pas_model.py:1079-1135`).
- Runtime phase mutation (`processor/processor.py:283-295`; `utils/freeze_schedule.py:310-391`).
- Direct additive fusion and mutable row selection (`model/fusion.py:54-56`; `utils/metrics.py:565-624`).

## 2. Non-negotiable requirements
1. Host baseline invariance must be measurable and enforced in external mode.
2. Prototype must be disable-able without changing host optimization behavior.
3. Fused metrics must be additive-value metrics, not host replacement metrics.
4. Training/inference mode contracts must be explicit in config and runtime logs.
5. Checkpoint authority policy must be explicit per mode.

## 3. Claim ladder (what each option can honestly claim)
| Option | Highest defensible claim | What it cannot claim |
|---|---|---|
| Option C (Calibration-first) | Safer fused inference behavior | Cannot claim host-safe training or modular architecture |
| Option A (Externalized with guardrails) | Host-safe external add-on mode (policy-enforced) | Cannot claim true architectural plug-and-play |
| Option B (Structural decoupling) | Strongest basis for true modular/plug-and-play architecture | Higher migration risk and implementation cost |

Interpretation rule:
- "Safer than before" is not equal to "plug-and-play".
- Option A is an intermediate safety mode, not end-state modularity.

## 4. Root causes proposals must address (from audit evidence)
1. **Joint optimization coupling**: one scalar objective combines host and prototype losses (`model/pas_model.py:1135`).
2. **Shared representation coupling**: prototype consumes live shared tensors without detach boundary (`model/pas_model.py:1121-1124`).
3. **Decision coupling**: direct additive fusion (`model/fusion.py:54-56`).
4. **Stage/phase mutability**: runtime toggles trainability/loss/optimizer (`processor/processor.py:283-295`; `utils/freeze_schedule.py:310-391`).
5. **Reporting/selection mutability**: subset-row selection may overwrite displayed `pas-t2i` semantics (`utils/metrics.py:565-610`).

## 5. Proposal set

### Option A: Host-safe external add-on mode with guardrails (intermediate)
Short description:
- Keep current PAS graph, but introduce an explicit `integration_mode=external_addon` contract that enforces host safety by policy and runtime checks.

What it removes:
- Prototype-to-host gradient coupling in external mode (via detach barrier at prototype inputs).
- Silent host update drift in external mode (via phase/optimizer constraints).
- Metric authority ambiguity in external mode (via strict row-selection policy).

What it keeps:
- Shared runtime graph (`PASModel`) still exists.
- Prototype remains representationally dependent on host/backbone artifacts.
- Fused inference coupling remains possible by design.

Code evidence that motivates this option:
- Prototype currently reads live shared tensors (`model/pas_model.py:1121-1124`).
- Total loss is joint scalar (`model/pas_model.py:1135`).
- Phase mutation can re-enable host training (`processor/processor.py:283-295`; `utils/freeze_schedule.py:301-391`).
- Selection/display semantics can diverge (`utils/metrics.py:565-624`).

Exact code areas to modify:
- `model/pas_model.py`
  - Add mode-gated detach before `prototype_head(...)` inputs at current callsite (`:1120-1129`).
  - Add guardrail: forbid host-logit input for prototype loss in external mode.
- `processor/processor.py`
  - On phase activation, enforce host groups frozen in external mode before optimizer rebuild (`:283-292`).
- `utils/freeze_schedule.py`
  - Add external-mode validation: reject phase configs that set host groups trainable.
- `solver/build.py` and/or pre-build checks
  - Assert host params absent from optimizer in external mode.
- `utils/metrics.py`
  - External-mode metric authority policy (no subset-row overwrite for checkpoint/primary metric).
- `utils/options.py` / `utils/config.py`
  - Add explicit `integration_mode` enum and validation.

Change surface:
- Model graph: No structural split.
- Forward pass: Yes.
- Loss path: Yes (mode guards).
- Optimizer groups: No schema change required, but add constraints.
- Freeze schedule: Yes (validation constraints).
- Checkpoint logic: Yes (authoritative row policy).
- Evaluator: Yes.
- Fusion logic: Optional (can keep formula unchanged).

Backward compatibility:
- High if default mode preserves current behavior.

Risks/tradeoffs:
- Prototype quality may drop if detached artifacts are insufficient.
- Contract enforcement adds configuration rigidity.

Expected benefit:
- Stronger host safety evidence with limited disruption.

Research cost: Low-Medium.
Engineering cost: Low-Medium.

#### 5.A External mode interface contract (mandatory)
This contract is required for Option A to be claim-honest.

1. Allowed host artifacts that prototype may read (read-only):
- `image_output.projected_pooled`, `image_output.projected_tokens`, resolved text token states, token masks, token positions (current source tensors at `model/pas_model.py:1121-1127`).

2. Detach requirements:
- All host/backbone artifacts passed into prototype must be detached in `external_addon` mode before `prototype_head(...)`.
- Rationale: current path has no detach boundary (`model/pas_model.py:1121-1124`).

3. Host logits in prototype loss:
- Forbidden in external mode.
- Enforce by passing `host_pairwise_logits=None` and disallowing `use_loss_weight_ret` in external mode.
- Rationale: current weighted prototype retrieval explicitly consumes host logits (`model/prototype/losses.py:523-531`).

4. Text state sharing:
- Shared text states may be consumed only as detached artifacts in external mode.

5. Evaluator-time host data usage:
- Prototype training cannot consume evaluator-time host outputs.
- Rationale: keep train/eval contracts separable.

6. Fusion during training:
- Forbidden as optimization target in external mode.
- Fused row allowed for evaluation diagnostics only.

7. Host loss coexistence in external mode:
- External prototype-training phase: host loss must not drive updates.
- If host-loss logging is needed, it is diagnostic-only.

8. Authoritative checkpoint row policy in external mode:
- Host checkpoints: authoritative row must be `host-t2i`.
- Prototype/fusion artifacts: may use additional rows, but cannot override host authority.
- Rationale: current selection path can remap/overwrite (`utils/metrics.py:565-624`).

### Option B: Structural decoupling (HostCore + PrototypePlugin + Composer)
Short description:
- Replace PAS monolith with explicit component boundaries and mode-specific composition.

What it removes:
- Single-graph structural coupling by default.
- Accidental joint optimization unless explicitly configured.
- Ambiguous mode semantics from mixed flags.

What it keeps:
- Optional fused inference at composition boundary.
- Optional explicit joint mode (as deliberate mode, not default side effect).

Code evidence that motivates this option:
- Monolithic construction in `PASModel` (`model/pas_model.py:41-118`).
- Shared forward path and joint loss (`model/pas_model.py:1079-1135`).

Exact code areas to modify:
- `model/build.py`: compose `HostCore` and `PrototypePlugin` explicitly.
- `model/pas_model.py`: split into components or replace with orchestrator.
- `processor/processor.py`: mode-specific training loops/optimizers.
- `utils/module_group_registry.py`: ownership by component boundary.
- `utils/modular_checkpoint.py`: per-component compatibility contracts.
- `utils/metrics.py`: explicit authority policy by mode.

Change surface:
- Model graph: Major.
- Forward pass: Major.
- Loss path: Major.
- Optimizer groups: Major.
- Freeze schedule: Reframed by mode.
- Checkpoint logic: Major.
- Evaluator: Moderate-Major.
- Fusion logic: Can be retained but moved behind composer boundary.

Backward compatibility:
- Medium-Low without compatibility adapters.

Risks/tradeoffs:
- Higher migration complexity and regression risk.

Expected benefit:
- Strongest architectural basis for true plug-and-play claims.

Research cost: Medium-High.
Engineering cost: High.

### Option C: Calibration-first fused inference safety
Short description:
- Keep current training graph; improve fused inference safety and reporting clarity.

What it removes:
- Some inference poisoning risk from raw additive fusion.
- Some reporting ambiguity in fused-row interpretation.

What it keeps:
- Shared-graph and joint-optimization coupling in training.
- Phase mutability.

Code evidence that motivates this option:
- Direct additive fusion in `ResidualScoreFusion` (`model/fusion.py:54-56`).
- Row selection/reporting mutability (`utils/metrics.py:565-624`).

Exact code areas to modify:
- `model/fusion.py`: add calibration layer (e.g., branch affine/temperature gate).
- `utils/metrics.py`: separate calibrated vs raw fused rows and enforce explicit metric labels.
- `utils/modular_checkpoint.py`: persist calibration artifact in fusion group.
- `utils/options.py`/`utils/config.py`: calibration config schema.

Change surface:
- Model graph: Low.
- Forward pass: Low.
- Loss path: Optional.
- Optimizer groups: Optional.
- Freeze schedule: None.
- Checkpoint logic: Low.
- Evaluator: Moderate.
- Fusion logic: Yes.

Backward compatibility:
- High.

Risks/tradeoffs:
- Can be misrepresented as plug-and-play if claim discipline is weak.

Expected benefit:
- Faster path to safer fused inference behavior.

Research cost: Low.
Engineering cost: Low.

## 6. Metric contract (mandatory)

### 6.1 External add-on mode metric authority
- Primary metric: `host-t2i` (host invariance metric).
- Secondary metrics: `prototype-t2i` (prototype standalone utility), fused rows (additive value only).
- Forbidden interpretation: fused improvement cannot be used to justify host regression.
- Forbidden selection policy: no fused/subset row may override host authority for host-safety claims.

### 6.2 Joint mode metric authority
- Primary metric can be configured for task objective, but must explicitly state source row (`val/top1_source_row`).
- Secondary metrics must include host row to monitor host drift.
- Forbidden interpretation: claiming host preservation without host-row parity checks.

### 6.3 Calibration-first mode metric authority
- Primary metric: fused calibrated row for deployment candidate evaluation.
- Required companion metrics: host-only row and prototype-only row.
- Forbidden interpretation: calling calibration-only improvements "modular plug-and-play".

## 7. Proof obligations before any host-safe claim
Any "host-safe external add-on" claim must satisfy all checks below:
1. **Host gradient check**: host/backbone gradients are zero in external mode prototype-training steps.
2. **Optimizer membership check**: host params absent from optimizer groups in external mode.
3. **Parameter-delta check**: host parameter deltas remain exactly zero (or below strict tolerance) across external-mode training.
4. **Host-output parity check**: host-only forward outputs/metrics match pre-integration baseline within predefined tolerance.
5. **Checkpoint authority check**: host checkpoint selection uses host row only in external mode.
6. **Row provenance check**: log and verify `val/top1_row` and `val/top1_source_row` consistency.
7. **Failure criteria**: if any check fails, external-mode host-safe claim is invalid.

Code motivation for these obligations:
- Current phase logic can mutate trainability/objective (`processor/processor.py:283-295`; `utils/freeze_schedule.py:310-391`).
- Current evaluator can remap source/display semantics (`utils/metrics.py:565-624`).
- Current optimizer is built from currently trainable groups (`model/pas_model.py:1002-1017`; `solver/build.py:172-205`).

## 8. Compare proposals (revised)
| Criterion | Option C | Option A | Option B |
|---|---|---|---|
| Claim honesty (what can be claimed today) | High (narrow claim) | High (intermediate claim) | High (strong claim potential) |
| True architectural modularity | Low | Medium-Low | High |
| Host safety potential | Medium (inference-centric) | High (if proof obligations pass) | Very High |
| Implementation practicality now | Very High | High | Low-Medium |
| Code disruption | Low | Medium | High |
| Debugging difficulty | Low-Medium | Medium | High |
| Backward compatibility | High | High | Medium-Low |
| Research risk | Low | Medium | High |

Important interpretation:
- Option A should **not** score close to Option B on true modularity. Option A is policy-enforced externalization inside the same runtime graph.

## 9. Best recommendation now
Recommend **Option A now**, explicitly as an intermediate step:
- Build `external_addon` with hard guardrails and proof obligations.
- Use it to test whether gradient/phase decoupling is sufficient to preserve host behavior in this codebase.
- Do **not** describe Option A as full plug-and-play architecture.
- Keep Option B as architectural end-state for strong modularity claims.

## 10. What Option A still fails to solve
1. Shared runtime still exists (`PASModel` monolith remains).
2. Prototype still depends on host/backbone-produced artifacts (even if detached).
3. Inference coupling still exists when fused rows are used.
4. External boundary is policy-enforced, not hard architectural isolation.
5. Misconfiguration risk remains if contract validation is incomplete.

## 11. When Option B becomes mandatory
Move from A to B when any of these occurs:
1. Host-safe proof obligations fail under realistic runs/seeds.
2. Team needs to claim true architectural plug-and-play, not policy-guarded safety.
3. Multiple prototype plugins or independent deployment lifecycle is required.
4. External-mode policy complexity becomes larger than structural split cost.
5. Checkpoint compatibility/traceability demands strict component boundaries.

## 12. Migration plan (recommended path: A first, then decision gate)
1. Add `integration_mode` config + validation.
2. Implement external-mode detach and host-logit prohibition in `PASModel.forward`.
3. Enforce phase and optimizer constraints for external mode.
4. Enforce metric authority/selection policy for external mode.
5. Add proof-check instrumentation (grad/optimizer/param-delta/parity/provenance).
6. Run mandatory experiments.
7. If obligations fail or remain fragile, escalate to Option B.

## 13. Validation plan (must-run experiments)
1. Host-only baseline before any prototype attachment.
2. Attach prototype codepath, fusion disabled (`lambda_prototype=0`), verify host parity.
3. External mode: prototype trainable, host frozen, verify all proof obligations.
4. Host trainable, prototype frozen (control), quantify host adaptation pressure.
5. Inference-only fusion (post-training) vs direct joint training fusion.
6. Calibration-only fusion vs raw additive fusion.
7. Exact vs approximate scorer stability under same weights (`utils/metrics.py:509-538`).
8. Stage-transition robustness checks (optimizer rebuild/state carryover).
9. Checkpoint selection audit: ensure host authority not overwritten in external mode.
10. Failure-triggered gate: any host-safety proof violation blocks host-safe claim.

## 14. Decision memo for implementation
- What to build now:
  - Option A external add-on mode with explicit contracts and proof checks.
- What NOT to claim after Option A:
  - Do not claim true modular/plug-and-play architecture.
- Which experiment decides whether A is enough:
  - Full proof-obligation suite under multi-seed runs with host parity checks.
- What result forces move to B:
  - Any repeated host drift, optimizer leakage, or metric-authority ambiguity in external mode.

## 15. Final recommendation
Implement Option A now as the most defensible intermediate step, with strict contract enforcement and proof obligations. Treat Option B as the mandatory architectural end-state for any strong claim of true plug-and-play modularity. Use Option C only as a complementary inference-safety layer, not as a substitute for decoupled training semantics.

## 16. Bottom line for the team
1. Current system is coupled by architecture, optimization, and decision path.
2. We can make it safer quickly, but safety is not the same as modularity.
3. Option A is the right next move because it is testable and low-disruption.
4. Option A must ship with strict contracts, not just config suggestions.
5. Host-only row must remain authoritative in external mode.
6. Fused gains can never excuse host degradation.
7. If host gradients or parameter deltas are non-zero, host-safe claim is invalid.
8. If row-selection policy can hide source metrics, claims are invalid.
9. Passing Option A proves a safer add-on mode, not true plug-and-play architecture.
10. True plug-and-play claims require Option B-level structural decoupling.
