# IMPLEMENT PLAN

## 1. Goal

This document defines the implementation execution plan for adding the prototype branch into the project runtime while preserving host semantics. The implementation MUST follow the approved integration contract and MUST remain consistent with the prototype method specification.

This is not a rewrite of ITSELF.

This is:
- ITSELF host preserved
- prototype branch attached externally
- score-level fusion only
- no host source modification
- careful support for both `train_mode = itself` and `train_mode = clip`

---

## 2. Non-Negotiable Constraints

## Global Clarifications for Implementation

- For `train_mode = clip`, Codex MUST define and use one canonical CLIP host runtime and one canonical `s_host^clip` score surface before implementing later phases.
- In Stage 1, host forward is feature extraction only. Host loss MUST NOT be part of the optimized objective.
- In Stage 2, initialization precedence MUST be explicit and config-driven:
  1. CLIP backbone initialization source
  2. optional Stage 1 prototype checkpoint load
  3. optional Stage 0 host checkpoint load only if declared compatible
- In Stage 3, calibration MUST NOT update representation parameters. Validation sweep/selection for `lambda_f` is allowed; representation training is forbidden.
- Codex MUST enumerate the exact supported stage/mode matrix before Phase F smoke runs.

- The directory `prototype/adapter/WACV2026-Oral-ITSELF/**` MUST remain untouched.
- Host runtime behavior MUST NOT be monkey-patched, overridden, or semantically redefined.
- Fusion MUST happen only at score level:
  - `s_total = s_host + lambda_f * s_proto`
- Prototype basis MUST consume token-level text states, not pooled text.
- Routing MUST consume the active host global image feature.
- `s_host` MUST always be interpreted according to the active `train_mode`:
  - `train_mode = itself` -> `s_host = s_host^itself`
  - `train_mode = clip` -> `s_host = s_host^clip`
- Legacy code is reference only. No legacy semantic behavior may be imported without explicit verification.

---

## 3. Definition of Done

Implementation is considered complete only when all of the following are true:

- all required modules exist under new `prototype/` implementation paths
- all stage behaviors are implemented and config-driven
- host parity passes exactly when prototype is disabled or `lambda_f = 0`
- Stage 0 supports both ITSELF reproduction and CLIP baseline mode
- Stage 1 supports frozen-host prototype stabilization using `L_div`, `L_diag`, and `L_ret`
- Stage 2 supports unfrozen backbone retraining from CLIP initialization for both `train_mode = itself` and `train_mode = clip`
- Stage 3 supports fusion calibration only
- all required verification tests pass
- no file under the host adapter tree is modified

---

## 4. Execution Rules for Codex

Codex MUST work phase by phase.

Codex MUST NOT start a later phase until the current phase deliverables are complete.

Codex MUST update the checklist in this file by changing:
- `[ ]` -> `[x]` only after the phase gate is satisfied

Codex MUST add a short completion note under each completed phase summarizing:
- files created or edited
- major decisions made
- tests run
- known limitations still open

Codex MUST NOT tick a phase based only on partial scaffolding.

---

## 5. Phase Overview

1. Phase A: Repository Setup and Safe Skeleton
2. Phase B: Host Runtime Adapters and Feature Surface Extraction
3. Phase C: Prototype Branch Core Modules
4. Phase D: Fusion and Integrated Scoring Runtime
5. Phase E: Config and Stage Control
6. Phase F: Training Loop Integration
7. Phase G: Verification and Safety Tests
8. Phase H: Minimal Run Recipes and Handoff Notes

---

## 6. Phase A: Repository Setup and Safe Skeleton

### Objective
Create the new implementation surface without touching host source.

### Required work
- Create new directories only under:
  - `prototype/integration/`
  - `prototype/prototype_branch/`
  - `prototype/fusion/`
  - `prototype/config/`
  - `prototype/tests/`
- Add package init files where needed.
- Add placeholder module files with stable names and docstrings.
- Add a top-level implementation README or short module map if useful.

### Suggested file skeleton
- `prototype/integration/host_runtime.py`
- `prototype/integration/feature_surface.py`
- `prototype/integration/stage_controller.py`
- `prototype/integration/model_runtime.py`
- `prototype/prototype_branch/prototype_bank.py`
- `prototype/prototype_branch/contextualizer.py`
- `prototype/prototype_branch/router.py`
- `prototype/prototype_branch/basis_builder.py`
- `prototype/prototype_branch/surrogate_builder.py`
- `prototype/prototype_branch/projector.py`
- `prototype/prototype_branch/scorer.py`
- `prototype/fusion/residual_fusion.py`
- `prototype/config/schema.py`
- `prototype/config/defaults.py`
- `prototype/tests/test_host_parity.py`
- `prototype/tests/test_feature_provenance.py`
- `prototype/tests/test_stage_freeze.py`
- `prototype/tests/test_fusion_contract.py`

### Gate
This phase is complete only if:
- directory structure exists
- no host file is edited
- all skeleton files import cleanly

### Checklist
- [x] Create safe implementation directory structure
- [x] Add module skeleton files with stable interfaces
- [x] Verify no host adapter file changed

### Completion note
- Completed on 2026-04-08.
- Files created: `prototype/integration/*`, `prototype/prototype_branch/*`, `prototype/fusion/*`, `prototype/config/*`, `prototype/tests/*` skeleton modules and package init files.
- Major decisions: interface-only contracts with docstrings; no runtime semantics implemented; no host adapter edits.
- Tests run: Python import smoke for all new skeleton modules (`IMPORT_OK`), plus no-touch host diff check (`git status --porcelain -- prototype/adapter/WACV2026-Oral-ITSELF` returned empty).
- Known limitations: implementations and contract tests are placeholders and deferred to later phases.

---

## 7. Phase B: Host Runtime Adapters and Feature Surface Extraction

### Objective
Build read-only adapters that extract the exact active-host tensors required by the prototype branch.

### Required work
- Implement host-mode aware runtime loading.
- Support:
  - `train_mode = itself`
  - `train_mode = clip`
- Implement a single feature-surface API that returns the exact tensors required downstream.

### Required outputs by mode

#### For `train_mode = itself`
Return at minimum:
- `v_i_global`
- `H_j`
- `t_j_global`
- `s_host^itself` surface inputs or components needed to compute it faithfully
- any required host retrieval image feature for prototype scoring

#### For `train_mode = clip`
Return at minimum:
- canonical CLIP global image retrieval feature
- token-level text states in the same representation space used immediately before CLIP text pooling
- canonical CLIP pooled text embedding
- `s_host^clip` surface inputs or components needed to compute it faithfully

### Mandatory constraints
- No rewriting of host logic.
- No semantic reinterpretation of host outputs.
- No projected / pre-projected mixing.
- Feature provenance MUST be explicit in code comments or named outputs.

### Gate
This phase is complete only if:
- feature extraction works in both modes
- tensor shapes are validated
- provenance tests pass
- `s_host` is never used without mode binding in integration code

### Checklist
- [ ] Implement host-mode aware runtime loader
- [ ] Implement feature surface extraction for `train_mode = itself`
- [ ] Implement feature surface extraction for `train_mode = clip`
- [ ] Bind `s_host` explicitly to `s_host^itself` or `s_host^clip`
- [ ] Add provenance assertions and shape checks

### Completion note
- Not completed yet.

---

## 8. Phase C: Prototype Branch Core Modules

### Objective
Implement the prototype branch as an additive branch that consumes host-derived tensors.

### Required work
- Implement learnable prototype bank
- Implement optional contextualization
- Implement routing over prototypes
- Implement basis construction from token states `H_j`
- Implement surrogate construction with row-wise semantics
- Implement prototype-side projector
- Implement prototype score computation

### Mandatory semantic rules
- Routing input MUST come from active host global image feature.
- Basis input MUST be token states `H_j`, never pooled text.
- Surrogate construction MUST preserve row-wise semantics:
  - row `i` corresponds to image `i`
  - columns correspond to captions `j`
- Prototype scoring MUST compare host-derived image retrieval feature against prototype-derived surrogate text embedding.

### Legacy allowance
Codex MAY borrow low-level implementation patterns from legacy only after semantic validation.
Codex MUST NOT copy legacy head logic, loss composition, or host wrappers.

### Gate
This phase is complete only if:
- the branch runs end to end on a dummy batch
- pairwise score shape is `[B, B]`
- row-wise perturbation behavior is correct
- diagonal exact branch is isolated from off-diagonal leakage

### Checklist
- [ ] Implement prototype bank
- [ ] Implement optional contextualizer
- [ ] Implement router with row-sum=1 guarantee
- [ ] Implement basis builder from token states
- [ ] Implement surrogate builder with row-wise semantics
- [ ] Implement prototype projector
- [ ] Implement prototype scorer returning `[B, B]`

### Completion note
- Not completed yet.

---

## 9. Phase D: Fusion and Integrated Scoring Runtime

### Objective
Combine active host score and prototype score at score level only.

### Required work
- Implement residual score fusion module
- Enforce shape equality between `s_host` and `s_proto`
- Support both:
  - `s_total = s_host^itself + lambda_f * s_proto`
  - `s_total = s_host^clip + lambda_f * s_proto`
- Add hard assertions against embedding-level fusion

### Mandatory constraints
- No modification of host scorer
- No fusion before score tensors exist
- No evaluator grid-search semantics in canonical runtime

### Gate
This phase is complete only if:
- fusion arithmetic is exact
- `lambda_f = 0` gives exact host parity
- fusion works in both host modes

### Checklist
- [ ] Implement residual fusion module
- [ ] Enforce score-level-only fusion contract
- [ ] Support both ITSELF-host and CLIP-host fusion surfaces
- [ ] Add parity assertion path for `lambda_f = 0`

### Completion note
- Not completed yet.

---

## 10. Phase E: Config and Stage Control

### Objective
Make all behavior config-driven and implement stage-specific gating correctly.

### Required work
- Implement config schema and defaults
- Add support for:
  - `train_mode`
  - `training.stage`
  - `prototype.enabled`
  - `fusion.lambda_f`
  - `host.lambda_s` when relevant
  - prototype temperatures and regularization weights
  - freeze policy toggles
- Implement stage controller with exact stage behavior

### Required stage behavior

#### Stage 0
- Support both:
  - ITSELF reproduction
  - CLIP baseline
- Prototype disabled
- Host-only losses active

#### Stage 1
- Prototype enabled
- Host fully frozen
- Active losses MUST include:
  - `L_div`
  - `L_diag`
  - `L_ret`

#### Stage 2
- Prototype enabled
- Backbone unfrozen
- Must support retraining from CLIP initialization for:
  - `train_mode = itself`
  - `train_mode = clip`
- Prototype freeze policy may be full, partial, or light, but MUST be explicit in config

#### Stage 3
- Representation parameters frozen
- Fusion calibration only

### Gate
This phase is complete only if:
- invalid stage/mode combinations are rejected
- stage transitions enforce freeze behavior correctly
- all required config keys validate correctly

### Checklist
- [ ] Implement config schema and defaults
- [ ] Add `train_mode` and `training.stage` support
- [ ] Implement Stage 0 behavior for both ITSELF and CLIP host-only runs
- [ ] Implement Stage 1 frozen-host prototype stabilization
- [ ] Implement Stage 2 unfrozen retraining from CLIP initialization
- [ ] Implement Stage 3 calibration-only behavior
- [ ] Validate mode-dependent use of `host.lambda_s`

### Completion note
- Not completed yet.

---

## 11. Phase F: Training Loop Integration

### Objective
Wire the integrated runtime into trainable execution without changing host source.

### Required work
- Build a project-side training runtime that selects stage and host mode
- Route losses correctly by stage
- Ensure optimizer parameter groups match freeze policy
- Ensure host-only, prototype-only, and joint-update phases behave exactly as declared

### Required loss behavior

#### Host mode = ITSELF
- Host loss path MUST remain faithful to ITSELF host behavior

#### Host mode = CLIP
- Host loss path MUST remain faithful to CLIP baseline path

#### Prototype branch
- Stage 1 MUST support `L_div`, `L_diag`, `L_ret`
- Stage 2 MUST support configured host and prototype losses together
- Stage 3 MUST disable representation learning updates

### Gate
This phase is complete only if:
- a single training step runs in every declared stage/mode combination that is meant to be supported
- optimizer groups match freeze expectations
- no forbidden parameter updates occur

### Checklist
- [ ] Integrate stage-aware training runtime
- [ ] Route host losses correctly for ITSELF mode
- [ ] Route host losses correctly for CLIP mode
- [ ] Route prototype losses correctly by stage
- [ ] Enforce optimizer parameter groups from freeze policy
- [ ] Run smoke training step for each supported stage/mode combination

### Completion note
- Not completed yet.

---

## 12. Phase G: Verification and Safety Tests

### Objective
Prove the implementation preserves semantics and does not silently drift.

### Mandatory tests
- Host parity test
- No-touch host diff test
- Routing row-sum test
- Tensor provenance test
- Diagonal fidelity isolation test
- Freeze gradient test
- Shape/orientation test
- Fusion-level arithmetic test
- Row-wise perturbation test

### Additional mode-sensitive requirements
- Host parity MUST be checked separately for:
  - `train_mode = itself`
  - `train_mode = clip`
- `lambda_f = 0` MUST reduce exactly to the active host score surface
- When `train_mode = clip`, no GRAB/local assumption may appear anywhere in test logic

### Gate
This phase is complete only if all mandatory tests pass.

### Checklist
- [ ] Add host parity tests for ITSELF mode
- [ ] Add host parity tests for CLIP mode
- [ ] Add no-touch host diff test
- [ ] Add routing row-sum test
- [ ] Add tensor provenance test
- [ ] Add diagonal fidelity isolation test
- [ ] Add freeze gradient test
- [ ] Add shape/orientation test
- [ ] Add fusion arithmetic test
- [ ] Add row-wise perturbation test
- [ ] Run full verification suite successfully

### Completion note
- Not completed yet.

---

## 13. Phase H: Minimal Run Recipes and Handoff Notes

### Objective
Leave the repository in a state that is runnable and reviewable.

### Required work
- Provide minimal run recipes for:
  - Stage 0 ITSELF baseline
  - Stage 0 CLIP baseline
  - Stage 1 prototype stabilization
  - Stage 2 prototype-enabled retraining
  - Stage 3 fusion calibration
- Provide a short handoff summary listing:
  - files added
  - files edited
  - supported stage/mode combinations
  - tests added
  - known unresolved risks

### Gate
This phase is complete only if a reviewer can identify how to launch each supported phase without reading the full codebase.

### Checklist
- [ ] Add minimal run recipe for Stage 0 ITSELF baseline
- [ ] Add minimal run recipe for Stage 0 CLIP baseline
- [ ] Add minimal run recipe for Stage 1 prototype stabilization
- [ ] Add minimal run recipe for Stage 2 prototype-enabled retraining
- [ ] Add minimal run recipe for Stage 3 fusion calibration
- [ ] Add final handoff summary

### Completion note
- Not completed yet.

---

## 14. Final Milestone Summary

Codex MUST tick these only when the corresponding phase gate is satisfied.

- [x] Phase A complete
- [ ] Phase B complete
- [ ] Phase C complete
- [ ] Phase D complete
- [ ] Phase E complete
- [ ] Phase F complete
- [ ] Phase G complete
- [ ] Phase H complete

---

## 15. Stop Conditions

Codex MUST stop and report instead of continuing silently if any of the following occurs:

- a required host tensor cannot be extracted without modifying host source
- `train_mode = clip` cannot be supported with a clean canonical host score surface
- Stage behavior cannot be implemented without violating the integration contract
- legacy code appears to be required for core semantics rather than optional scaffolding
- host parity fails and the cause is not yet isolated

In such cases, Codex MUST document:
- exact blocker
- affected files
- why the blocker violates the contract
- the smallest safe next step
