# LEGACY_CLEANUP_REPORT

## Scope

Phase E retired the old ITSELF-specific retrieval path and recentered the repository on the PAS model as the only active model path.

## Files Removed

- `configs/baseline_legacy.yaml`
  Removed because the repository no longer preserves the legacy baseline as a runnable primary config.
- `configs/pas_v1_draft.yaml`
  Removed because Phase E replaces the draft surface with real experiment configs.
- `model/objectives.py`
  Removed because TAL/CID loss helpers are no longer part of the active training path.
- `model/grab.py`
  Removed because the GRAB local branch is no longer used by the active model.
- `model/pooling.py`
  Removed because image-conditioned pooling is now implemented in `model/prototype/` and the old interim pooling boundary is no longer used.
- `tests/test_phase_d_integration.py`
  Removed because it validated the deleted dual-path Phase D wrapper.
- `scripts/phase_d_smoke.py`
  Removed because Phase E replaces it with a smoke harness centered on the active model.

## Files Added

- `configs/train_pas_v1.yaml`
- `configs/debug_pas_v1.yaml`
- `configs/ablation_pas_no_context.yaml`
- `configs/ablation_pas_no_diversity.yaml`
- `scripts/phase_e_smoke.py`
- `tests/test_phase_e_integration.py`
- `LEGACY_CLEANUP_REPORT.md`
- `PHASE_E_REPORT.md`

## Files Heavily Modified

- `model/build.py`
  Replaced the old compatibility-heavy wrapper with `PASModel` as the primary model class.
- `utils/options.py`
  Replaced the ITSELF-era CLI surface with a prototype-centered argument set.
- `utils/config.py`
  Removed legacy config mappings and aligned YAML parsing with the new experiment schema.
- `train.py`
  Removed ITSELF naming and legacy run-directory logic; now launches the PAS model directly.
- `test.py`
  Removed legacy logger naming and aligned evaluation to the active model path.
- `solver/build.py`
  Uses explicit optimizer param groups rather than old heuristic overrides.
- `processor/processor.py`
  Uses structured loss/debug outputs and centralized metric extraction.
- `utils/metrics.py`
  Uses only the prototype retrieval interface for evaluation.
- `utils/metric_logging.py`
  Centralizes scalar extraction and debug metric naming.
- `README.md`
  Updated active documentation to reflect the current model.
- `ARCHITECTURE.md`
  Updated architecture notes to the current system.
- `MODEL_INTERFACE_CONTRACT.md`
  Updated the public model contract to `PASModel`.
- `MODULE_CONTRACTS.md`
  Updated module contracts to the active model/training surface.

## ITSELF-Specific Branches Removed

- Legacy baseline-vs-prototype branching in `model/build.py`.
- TAL/CID loss assembly and bookkeeping.
- GRAB/local retrieval evaluation assumptions.
- Old experiment naming based on `loss_names`.
- Legacy smoke/integration harnesses that depended on the removed ITSELF wrapper.

## Config Keys Removed Or Retired

The following obsolete keys were removed from the active parser/config mapping:
- `loss_names`
- `only_global`
- `return_all`
- `topk_type`
- `layer_index`
- `modify_k`
- `tau`
- `select_ratio`
- `margin`
- `lambda1_weight`
- `lambda2_weight`
- `lr_override_prototypes`
- `lr_override_projectors`
- `lr_override_logit_scale`
- `lr_override_backbones`

These were replaced by explicit prototype-path settings and explicit optimizer-group LR keys.

## Logging Names Changed

- Logger namespaces now use `pas.*` instead of ITSELF-era names.
- Active wandb default project name is now `PAS-retrieval`.
- Training loss logs now use `train/loss_total`, `train/loss_infonce`, `train/loss_diversity`, `train/loss_balance`.
- Debug logs now use structured `debug/*` names tied to prototype routing, pooling, and geometry.

## Intentionally Retained Legacy Components

- `model/clip_model.py`
  Retained because the CLIP encoder backbone and intermediate-state access are still core infrastructure.
- `datasets/`, `utils/checkpoint.py`, `utils/logger.py`, `utils/comm.py`
  Retained because they remain useful generic infrastructure.
- `README.upstream.md`
  Retained as a historical upstream snapshot.
- `PHASE_A_REPORT.md`, `PHASE_B_REPORT.md`, `PHASE_C_REPORT.md`, `PHASE_D_REPORT.md`, `LEGACY_CODE_MAPPING.md`
  Retained as historical documentation of the staged migration, even though they describe superseded intermediate states.

## Rationale Summary

The cleanup goal was not to erase history. It was to stop carrying dead runtime code and misleading active configuration around a model path that is no longer the intended system. Phase E keeps reusable infrastructure, removes dead ITSELF-specific execution paths, and makes the repository easier to read as a PAS retrieval codebase.

