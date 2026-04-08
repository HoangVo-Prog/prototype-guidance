# Prototype Integration Handoff Notes

## Scope
This handoff summarizes current repository state for the plug-and-play prototype integration.

Core invariants restated:
- Host adapter tree is read-only:
  - `prototype/adapter/WACV2026-Oral-ITSELF/**` untouched.
- Fusion remains score-level only:
  - `s_total = s_host + lambda_f * s_proto`
- `s_host` is mode-bound:
  - `train_mode=itself -> s_host^itself`
  - `train_mode=clip -> s_host^clip`
- CLIP mode does not assume GRAB/local branch.

## Canonical User Launcher Surface
- Root launcher: `train.py`
- Config directory: `configs/`
- Canonical command pattern:
  - `python train.py --config configs/<stage_mode>.yaml`
- No test helper and no python one-liner is required for primary launch.

## Status Snapshot

### Phase Gate Certification
- Phase A: certified complete.
- Phase B: certified complete (user-validated state).
- Phase C: certified complete (user-validated state).
- Phase D: certified complete (user-validated state).
- Phase E: certified complete under unchanged user-validated state.
- Phase F: implemented, not certified in this environment.
- Phase G: verification tests expanded/implemented, not certified in this environment.
- Phase H: documentation deliverables complete in this repository state.

### Environment Limitation (Current Session)
- `pyyaml` missing.
- `torch` missing.
- `pytest` missing.
- Consequence:
  - full runtime/verification gates (F/G) cannot be certified in this session.
  - launcher execution cannot be re-validated in this session.

## Supported Stage/Mode Combinations
Declared and implemented runtime matrix:
- `itself + stage0`
- `itself + stage1`
- `itself + stage2`
- `itself + stage3`
- `clip + stage0`
- `clip + stage1`
- `clip + stage2`
- `clip + stage3`

## Files Added
- `prototype/integration/training_runtime.py`
- `prototype/integration/synthetic_host_runtime.py`
- `train.py`
- `configs/base.yaml`
- `configs/stage0_itself.yaml`
- `configs/stage0_clip.yaml`
- `configs/stage1_itself.yaml`
- `configs/stage1_clip.yaml`
- `configs/stage2_itself.yaml`
- `configs/stage2_clip.yaml`
- `configs/stage3_itself.yaml`
- `configs/stage3_clip.yaml`
- `prototype/tests/_runtime_test_utils.py`
- `prototype/tests/test_no_touch_host_boundary.py`
- `prototype/RUN_RECIPES.md`
- `prototype/HANDOFF_NOTES.md`

## Files Edited
- `prototype/integration/__init__.py`
- `prototype/tests/test_host_parity.py`
- `prototype/tests/test_stage_freeze.py`
- `prototype/tests/test_feature_provenance.py`
- `prototype/tests/test_prototype_branch_phase_c.py`
- `prototype/tests/test_training_runtime_phase_f.py`

## Tests Added/Expanded
- Host parity by mode:
  - `prototype/tests/test_host_parity.py`
- No-touch host boundary:
  - `prototype/tests/test_no_touch_host_boundary.py`
- Freeze/update behavior:
  - `prototype/tests/test_stage_freeze.py`
- Tensor provenance + pooled-text rejection:
  - `prototype/tests/test_feature_provenance.py`
- Routing finite + row-sum checks:
  - `prototype/tests/test_prototype_branch_phase_c.py`
- Stage/runtime extension checks:
  - `prototype/tests/test_training_runtime_phase_f.py`

## Known Unresolved Blockers/Risks
- Runtime certification blocker:
  - inability to execute `pytest` suite without `pytest` and `torch`.
- Remaining gate uncertainty until runnable validation:
  - Phase F smoke/update behavior across all stage/mode pairs.
  - Phase G full mandatory verification pass.

## Reviewer Next Actions
1. Install launcher/runtime dependencies (`pyyaml`, `torch`) in target environment.
2. Execute canonical stage recipes:
   - `python train.py --config configs/stage0_itself.yaml`
   - `python train.py --config configs/stage0_clip.yaml`
   - `python train.py --config configs/stage1_itself.yaml`
   - `python train.py --config configs/stage1_clip.yaml`
   - `python train.py --config configs/stage2_itself.yaml`
   - `python train.py --config configs/stage2_clip.yaml`
   - `python train.py --config configs/stage3_itself.yaml`
   - `python train.py --config configs/stage3_clip.yaml`
3. After runtime dependencies are available, install `pytest` and run verification suite for Phase F/G certification.
4. If verification passes, update milestone certification for Phase F and Phase G.

## Run Entry References
- Minimal phase launch recipes:
  - `prototype/RUN_RECIPES.md`
- Canonical launcher:
  - `train.py`
- Canonical configs:
  - `configs/stage0_itself.yaml`
  - `configs/stage0_clip.yaml`
  - `configs/stage1_itself.yaml`
  - `configs/stage1_clip.yaml`
  - `configs/stage2_itself.yaml`
  - `configs/stage2_clip.yaml`
  - `configs/stage3_itself.yaml`
  - `configs/stage3_clip.yaml`
- Training runtime entrypoint:
  - `prototype/integration/training_runtime.py`
- Stage/mode schema contract:
  - `prototype/config/schema.py`
