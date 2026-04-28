# True Plug-and-Play Structural Split Implementation

## 1. Executive Summary
- Implemented a real structural split path with explicit `HostCore`, `PrototypePlugin`, and `Composer` components in code.
- Added first-class runtime mode semantics (`host_only`, `prototype_only`, `fused_external`, `joint_training`, `calibration_only`) and mode-driven trainer routing.
- Added an explicit versioned HostCore -> PrototypePlugin interface object with detach policy and host-logit guardrails.
- Kept authority/provenance checkpoint validation active and extended checkpoint metadata with component/runtime fields.
- Preserved compatibility by keeping existing host-only model builders and modular checkpoint group names.

## 2. What Was Implemented
- New runtime mode module:
  - [model/runtime_modes.py](/D:/Programming/Python/prototype-guidance/model/runtime_modes.py)
- New explicit interface contract:
  - [model/interface_contract.py](/D:/Programming/Python/prototype-guidance/model/interface_contract.py)
- New split runtime components and orchestrator:
  - [model/plug_and_play.py](/D:/Programming/Python/prototype-guidance/model/plug_and_play.py)
- Build path now routes prototype-enabled runtime to structural split builder:
  - [model/build.py](/D:/Programming/Python/prototype-guidance/model/build.py)
- Trainer now has explicit entrypoint routing and mode-driven trainability enforcement:
  - [processor/processor.py](/D:/Programming/Python/prototype-guidance/processor/processor.py)
- Module-group/checkpoint prefixes extended for new component key paths:
  - [utils/module_group_registry.py](/D:/Programming/Python/prototype-guidance/utils/module_group_registry.py)
- Checkpoint payload metadata extended for component/runtime lifecycle:
  - [utils/modular_checkpoint.py](/D:/Programming/Python/prototype-guidance/utils/modular_checkpoint.py)
- Config/CLI runtime mode surface added:
  - [utils/options.py](/D:/Programming/Python/prototype-guidance/utils/options.py)
  - [utils/config.py](/D:/Programming/Python/prototype-guidance/utils/config.py)

## 3. New Component Boundaries
- `HostCore`:
  - Owns backbone/text-image extraction and host retrieval head execution.
  - Exposes host export builder via explicit `HostPluginInterface`.
- `PrototypePlugin`:
  - Owns prototype head execution, basis-bank path, exact/approx similarity calls.
  - Consumes only `HostPluginInterface` for external-mode forward.
- `Composer`:
  - Owns score composition through the fusion module.
- `PASRuntimeModel`:
  - Orchestrates mode semantics; no longer relies on `PASModel.forward` as sole owner.

## 4. New Runtime Modes
- Implemented and normalized in `model/runtime_modes.py`:
  - `host_only`
  - `prototype_only`
  - `fused_external`
  - `joint_training`
  - `calibration_only`
- `processor/processor.py` now dispatches to explicit train entrypoints:
  - `train_host_core`
  - `train_prototype_external`
  - `train_composer_calibration`
  - `train_joint`

## 5. New Interface Contract in Code
- `HostPluginInterface` and `HostExportPolicy` enforce:
  - explicit artifact names
  - shape/batch validation
  - version check (`host_export_v1`)
  - detachable snapshot behavior
  - host pairwise logits excluded by default in external modes
- In `PASRuntimeModel`:
  - `joint_training` policy allows live tensors + host logits.
  - external/calibration policies detach and block host logits.

## 6. Trainer Split Summary
- `do_train(...)` now resolves runtime mode and dispatches explicit mode entrypoints.
- Mode semantics are enforced before training loop via `_apply_runtime_mode_trainability(...)`.
- Freeze schedule is demoted:
  - active as primary phase controller only in `joint_training`.
  - ignored with explicit warning in other runtime modes.

## 7. Checkpoint Lifecycle Changes
- Existing authority validation path is reused (no regression).
- Payload metadata now records:
  - `runtime_mode`
  - `component_name` (HostCore / PrototypePlugin / Composer)
  - `authority_bucket_expected`
- Loading path now validates component mismatch when metadata is present.

## 8. Compatibility Notes
- Host-only builders (`build_clip_host`, `build_itself_host`) remain available.
- Modular checkpoint group names remain stable (`host`, `prototype_bank`, `prototype_projector`, `fusion`).
- Prefix registries now support both legacy monolith paths and new split component paths.
- `PASModel` still exists; structural split path is now the prototype-enabled default in `model/build.py`.

## 9. Tests Added/Updated
- New split-runtime tests:
  - [tests/test_structural_split_runtime.py](/D:/Programming/Python/prototype-guidance/tests/test_structural_split_runtime.py)
- Updated authority tests for prototype/fusion component saves:
  - [tests/test_metric_checkpoint_authority.py](/D:/Programming/Python/prototype-guidance/tests/test_metric_checkpoint_authority.py)
- In this environment, `torch` is unavailable, so runtime tests are skipped under `unittest` guards.

## 10. What Remains for Follow-Up
- Implement full composer-calibration objective/trainer logic (current path is structural scaffold).
- Add stricter checkpoint compatibility assertions for cross-version interface metadata.
- Expand end-to-end GPU-backed integration tests once `torch` test runtime is available.
- Continue reducing legacy coupling by gradually moving remaining PAS utility logic out of `model/pas_model.py`.
