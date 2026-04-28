# Metric/Checkpoint Authority Validation Implementation

## 1. Executive summary
This change implements the smallest safe pre-refactor step: explicit row provenance and component-aware authority validation across evaluator -> trainer -> modular checkpoint manager.

The implementation is additive and does **not** refactor PAS model architecture. It enforces measurement/checkpoint semantics so future HostCore/PrototypePlugin/Composer refactor work has trustworthy validation signals.

## 2. Current problem being fixed
Before this change, row semantics could silently drift across:
- displayed row (`val/top1_row`),
- source row (`val/top1_source_row`),
- row-selection policy,
- checkpoint metric row used for save decisions.

In subset-selection behavior, evaluator could display `pas-t2i` while selecting another source row, and trainer forwarded display row to checkpoint saves. This made checkpoint authority ambiguous.

## 3. Files changed
1. `utils/metrics.py`
2. `processor/processor.py`
3. `utils/modular_checkpoint.py`
4. `utils/config.py`
5. `tests/test_metric_checkpoint_authority.py` (new)
6. `tests/test_config_surface.py`

## 4. New data flow: evaluator -> trainer -> checkpoint manager

### Evaluator (`utils/metrics.py`)
- Added explicit authority-role classification for rows (`host`, `prototype`, `fused`).
- `_build_similarity_rows(...)` now returns both row tensors and row metadata (lambdas + authority role).
- Added `Evaluator.build_authority_context(...)` to produce stable provenance payload:
  - `display_row`
  - `source_row`
  - `mismatch`
  - `selected_source_role`
  - `candidates` (best row by authority role)
  - `row_roles`
  - `row_metrics`
- Preserved display/source divergence explicitly; added mismatch warning log.
- Stopped mutating `pas-t2i` metric values from subset-selected rows (provenance now preserved rather than overwritten).
- Exposed authority fields in `latest_metrics` under `val/authority/*`.

### Trainer (`processor/processor.py`)
- Reads evaluator authority context and selects checkpoint authority row from `source_row` (not display row).
- Logs display/source mismatch explicitly.
- Passes provenance and authority context to checkpoint manager on both latest and best save calls.

### Checkpoint manager (`utils/modular_checkpoint.py`)
- Added `MetricAuthorityPolicy` + `AuthorityValidationResult`.
- Added component bucket mapping:
  - `host` -> `host`
  - `prototype_bank`, `prototype_projector` -> `prototype`
  - `fusion` -> `fused`
- Added per-group metric-row resolution using authority candidates/source/display/row metrics.
- Added explicit validation before save; invalid row authority now fails loudly (strict mode) or rejects save (warn/log mode).
- Save payload now stores extra metric provenance:
  - `row`, `display_row`, `source_row`
  - `authority_bucket`, `authority_valid`
  - `selection_reason`
- Best-save tracking is now per-group (`best_metric_value_by_group`) so each component uses its own authority row/metric.

## 5. Authority rules implemented
- Host group saves require host-authorized rows.
- Prototype groups require prototype-authorized rows.
- Fusion group requires fused-authorized rows.
- Row role resolution order:
  1. evaluator-provided `row_roles`
  2. optional row-name fallback classification (configurable)
- Group save is skipped/rejected if authority validation fails.

## 6. What this now protects against
1. Checkpoint saves using display-row alias while source row differs.
2. Subset-driven row-selection ambiguity silently changing checkpoint authority.
3. Cross-component row leakage (e.g., saving host artifact with prototype-authorized row).
4. Hidden row provenance loss in checkpoint payloads.

## 7. What remains for future HostCore/PrototypePlugin/Composer split
This change does **not**:
- split model/runtime components,
- change PAS forward/loss coupling,
- replace schedule-driven trainer semantics,
- redesign evaluator architecture.

It only hardens measurement/checkpoint semantics so future refactor phases can trust validation outcomes.

## 8. Backward compatibility caveats
- Added checkpointing config section key: `checkpointing.authority_validation` with boolean fields:
  - `enabled`, `strict`, `warn_only`, `allow_fallback_row_name_classification`.
- If strict authority validation is enabled (default), invalid component-row pairing now raises instead of silently saving.
- Evaluator now preserves row provenance explicitly and no longer rewrites `pas-t2i` metrics from subset selection.

## 9. Tests added/updated
- Added: `tests/test_metric_checkpoint_authority.py`
  - default display/source parity case
  - subset divergence provenance case
  - valid host authority save case
  - disallowed-row save rejection case
  - deterministic policy behavior case
- Updated: `tests/test_config_surface.py`
  - validates `checkpointing.authority_validation` config surface acceptance

## 10. Validation run notes
In this environment, unittest discovery for these suites is import-gated and currently reports skipped tests. Python compilation of modified files was verified successfully.
