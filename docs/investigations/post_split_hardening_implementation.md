# Post-Split Hardening Implementation

## 1. Executive summary
- Implemented a real `calibration_only` training objective in the structural split runtime (`PASRuntimeModel`) so composer parameters are optimized with an explicit calibration loss instead of a zero-loss scaffold.
- Added strict compatibility validation in modular checkpoint loading for component/schema/interface/runtime semantics (for modular payloads), with explicit rejection on incompatible mixed-source loads.
- Expanded proof-oriented runtime tests for external-mode isolation, calibration isolation, interface contract validation, and checkpoint compatibility failures.
- Kept architecture boundaries (`HostCore` / `PrototypePlugin` / `Composer`) intact; this is hardening of semantics, not a redesign.

## 2. What was implemented

### Calibration path
- `model/plug_and_play.py`
  - Added composer calibration objective path in `PASRuntimeModel.forward(...)` for `runtime_mode=calibration_only`.
  - Added `PASRuntimeModel._compute_composer_calibration_loss(...)`:
    - builds host/prototype pairwise logits from outputs,
    - fuses via `Composer.fuse(...)`,
    - optimizes symmetric retrieval CE (`i2t` + `t2i`) over fused logits.
  - Added `loss_composer_calibration` to outputs and debug logging.
  - `fuse_retrieval_similarity(...)` now passes explicit score schema versions into composer validation.
  - Added composer calibration parameters to optimizer grouping (`named_optimizer_groups`).

### Runtime/trainability hardening
- `processor/processor.py`
  - Runtime-mode trainability now includes composer calibration parameters (`composer.host_log_scale`, `composer.prototype_log_scale`, `composer.log_temperature`).
  - Training loop now passes `disable_proxy_losses=True` in `calibration_only` mode to keep calibration path isolated.

### Interface and schema controls
- `model/interface_contract.py`
  - Enforced supported host interface versions.
  - Enforced presence of `special_token_positions['eos']`.
  - Enforced hard failure when host logits are provided under `HostExportPolicy(allow_host_pairwise_logits=False)`.

### Checkpoint compatibility gates
- `utils/modular_checkpoint.py`
  - Added compatibility extraction/normalization and model compatibility query hooks.
  - Save payloads now embed component compatibility metadata (component/schema/interface/score schema/runtime compatibility).
  - Load path now validates modular payload compatibility before applying state dict:
    - component name match,
    - component schema match,
    - runtime mode compatibility (against explicit compatible mode list when provided),
    - prototype interface version compatibility,
    - composer host/prototype score schema compatibility.
  - In strict mode, incompatible payloads now fail loudly.

### Module-group lifecycle alignment
- `utils/module_group_registry.py`
  - Added composer calibration params to fusion logical group prefixes so modular fusion checkpointing covers calibration params.

### Config/CLI surface
- `utils/options.py`
  - Added `--composer_calibration_enabled`.
- `utils/config.py`
  - Added `fusion.composer_calibration_enabled` mapping support.

## 3. Real `calibration_only` semantics now supported
- `runtime_mode=calibration_only` now computes a non-zero composer objective.
- Host/prototype branches remain frozen by runtime trainability policy; calibration loss is composer-side only.
- Composer calibration parameters are explicitly trainable and included in optimizer grouping.
- Composer calibration is now active for:
  - `calibration_only` training,
  - fused inference paths via `Composer.fuse(...)` (same composer parameters/schemas).

## 4. Compatibility rules now enforced
- Modular checkpoint loads now validate:
  - `component_name` consistency with group ownership.
  - `component_schema_version` match when available.
  - `runtime_mode` compatibility (allow-list aware).
  - Prototype groups: `host_export_interface_version` must be accepted by model plugin compatibility.
  - Fusion group: host/prototype score schema versions must match composer expectations.
- Invalid combinations are rejected (strict load mode), preventing silent mixed-source drift.

## 5. Tests added/updated

### Updated
- `tests/test_structural_split_runtime.py`
  - Updated interface test for strict host-logit export guard behavior.
  - Added interface validation failure checks (missing `eos`, unsupported interface version).
  - Added host-only runtime behavior assertion.
  - Added external-mode optimizer/gradient/parameter isolation test for host parameters.
  - Added calibration-only isolation test (composer-only trainable, host/prototype unchanged).
  - Added checkpoint compatibility rejection test for incompatible prototype interface version.

### Added
- `tests/test_metric_checkpoint_authority.py`
  - Added compatible component model fixture with `get_group_checkpoint_compatibility(...)`.
  - Added valid host+prototype modular load test.
  - Added rejection tests for:
    - wrong component metadata,
    - incompatible composer schema.

### Execution note
- In this environment, torch-dependent tests are skipped because torch is unavailable.
- Python compilation checks for all modified files passed.

## 6. Plug-and-play proof obligations now covered
- External mode isolation:
  - host params excluded from effective training updates in prototype external mode (trainability + optimizer tests).
  - host gradients/parameter deltas checked in tests.
- Calibration-only isolation:
  - only composer params trainable and updated.
  - host/prototype parameter invariance checked.
- Interface contract proof:
  - forbidden host-logit export rejection,
  - required artifact validation (`eos`),
  - interface version rejection.
- Checkpoint compatibility proof:
  - strict rejection on component/schema/interface incompatibility.

## 7. What remains before a strong final claim
- Run full torch-backed test suite in the target training environment (current environment cannot execute torch tests).
- Add integration-level calibration quality checks (e.g., fused additive value under fixed host/prototype checkpoints) to validate performance impact, not only isolation semantics.
- Expand compatibility tests to cover mixed runtime-mode transitions with real saved artifacts from training runs.
