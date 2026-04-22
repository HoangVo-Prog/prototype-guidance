# Prototype-Side Next Implementation Plan

## 1. Purpose
This note defines the next implementation pass focused on the **prototype side** after the host-only retrieval refactor.

Target truth to preserve:
- Retrieval ranking is host-only.
- Prototype remains a semantic-structure/training regularization module.
- No prototype retrieval branch semantics are exposed through APIs, config names, or docs.

## 2. Current Snapshot (What Is Already Good)
- Runtime retrieval path in evaluator is host-only (`utils/metrics.py`).
- Removed runtime modes are enforced (`model/runtime_modes.py`, `tests/test_structural_split_runtime.py`).
- Fusion config and retrieval scorer flags are rejected (`utils/config.py`, `utils/options.py`).
- Checkpoint group authority no longer includes fusion (`utils/module_group_registry.py`, `utils/modular_checkpoint.py`, `tests/test_metric_checkpoint_authority.py`).

## 3. Prototype-Side Gaps To Address Next

### 3.1 Naming debt still leaks old retrieval meaning
- `freeze_fusion` is still used as a live training/config control even though fusion retrieval was removed:
  - `utils/options.py`
  - `utils/config.py`
  - `model/pas_model.py`
- Logical group `fusion` still means prototype pooling/token modules:
  - `utils/module_group_registry.py`
  - `utils/freeze_schedule.py`
  - `processor/processor.py` (legacy_fusion_prefixes)

Impact: behavior may be correct, but names still teach the old architecture.

### 3.2 Prototype retrieval-scoring APIs still exist in module surface
- `PrototypePlugin` still exposes dead retrieval scorer methods:
  - `compute_exact_similarity`
  - `compute_approximate_similarity`
  - `build_text_basis_bank`
  - file: `model/plug_and_play.py`
- `PrototypeConditionedTextHead` still contains pairwise similarity functions named for retrieval scoring:
  - file: `model/prototype/head.py`

Impact: unused APIs preserve the old “prototype scorer” mental model and risk accidental reuse.

### 3.3 Legacy schema/docs still contradict runtime truth
- `configs/schema_pas_reference.yaml` still declares removed concepts:
  - `prototype_method_role: retrieval_branch`
  - `prototype_inference_mode`
  - `fusion.*`
  - `evaluation.retrieval_scorer`
  - checkpoint `fusion` group
- `docs/cli_flag_reference.md` still lists removed fusion flags.

Impact: users can copy invalid configuration patterns from official docs/templates.

### 3.4 Prototype-side debug naming still implies retrieval
- Training/debug keys still include retrieval-centric labels (`surrogate_retrieval_logits`, `surrogate_pairwise_logits`, `surrogate_retrieval_grad_norm`):
  - `model/pas_model.py`
  - `processor/processor.py`
  - `utils/metric_logging.py`

Impact: logging language suggests prototype is still a retrieval scorer.

## 4. Implementation Work Plan (Prototype-Focused)

### Phase A: Rename semantic surfaces to truth-first names
1. Replace `freeze_fusion` with `freeze_prototype_pooling` across parser/config/runtime.
2. Replace logical group name `fusion` with `prototype_pooling` in:
   - `utils/module_group_registry.py`
   - `utils/freeze_schedule.py`
   - `processor/processor.py`
   - any schedule parsing/validation text.
3. Remove compatibility aliases that keep old naming alive unless strictly required for one-time migration error messages.

### Phase B: Delete dead prototype retrieval scorer APIs
1. Remove unused prototype scorer entry points from `PrototypePlugin` (`model/plug_and_play.py`).
2. Restrict prototype head public surface to structure/training artifacts (routing, basis, diagonal teacher/student, semantic losses).
3. If internal pairwise helpers are still needed for loss diagnostics, rename to non-retrieval semantics and keep them private.

### Phase C: Clean config/template/docs at root
1. Rewrite `configs/schema_pas_reference.yaml` to host-only retrieval semantics.
2. Remove removed fusion/retrieval flags from `docs/cli_flag_reference.md`.
3. Update any investigation notes that still describe active fusion/composer lifecycle as current behavior.

### Phase D: Logging contract cleanup
1. Rename debug keys that imply prototype retrieval scoring.
2. Keep metric continuity where necessary by explicit migration notes, not by dual semantic naming.

## 5. Required Tests For This Next Pass

1. **No prototype retrieval API exposure**
   - Assert structural split model/prototype plugin no longer exposes retrieval scorer methods intended for ranking.

2. **No fusion naming in runtime freeze/group API**
   - `freeze_fusion` rejected.
   - `fusion` logical group rejected.
   - `prototype_pooling` accepted.

3. **Schema truth test**
   - `configs/schema_pas_reference.yaml` must not contain:
     - `fusion:`
     - `retrieval_scorer`
     - `prototype_inference_mode`
     - `retrieval_branch`.

4. **Debug naming regression test**
   - Ensure removed retrieval-like debug keys are absent from canonical tracked scalar maps.

5. **Prototype losses still functional**
   - Existing semantic-loss tests continue to pass (`tests/test_semantic_*`, `tests/test_prototype_init_schedule.py`).

## 6. Backward Compatibility Policy For This Pass
- Break old names that preserve wrong semantics.
- Emit clear migration errors for removed keys (`freeze_fusion`, `fusion` group, retrieval_branch role).
- Do not keep silent aliases that continue old architecture language.

## 7. Execution Order Recommendation
1. Phase A (renaming/runtime contracts)
2. Phase B (API deletion in prototype plugin/head)
3. Phase C (schema/docs)
4. Phase D + tests

Reason: runtime contracts and names should stabilize first so tests/docs can lock the final semantics.

## 8. Definition of Done (Prototype-Side)
- Prototype cannot be interpreted as a retrieval scorer from public APIs, config keys, runtime mode names, group names, or docs.
- Host-only retrieval semantics remain unchanged and enforced.
- Prototype semantic-structure losses and diagnostics still run and are covered by tests.
