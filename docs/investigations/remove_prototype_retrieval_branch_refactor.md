# Remove Prototype Retrieval Branch Refactor

## 1. Executive Summary
This refactor removes prototype-as-retrieval semantics from active runtime behavior. Retrieval ranking now uses HostCore similarity only. PrototypePlugin remains for training-time structure losses and diagnostics.

## 2. Retrieval Semantics Removed
- Removed host/prototype/fused retrieval row sweep in evaluation.
- Removed runtime support for prototype-only, fused-external, and calibration-only retrieval semantics.
- Removed approximate prototype retrieval scorer path from active evaluator/model retrieval flow.
- Removed composer/fusion-based retrieval selection from active runtime.

## 3. Runtime Modes Deleted or Changed
- Removed: `prototype_only`, `fused_external`, `calibration_only`.
- Kept: `host_only`, `joint_training`, `auto`.
- `auto` now resolves to `host_only` for inference and `joint_training` for training when prototype branch is enabled.

## 4. Config Flags Deleted
Removed from active config/CLI semantics:
- `fusion.*` (including `lambda_host`, `lambda_prototype`, `eval_subsets`, `coefficient_source`, composer calibration knobs)
- `evaluation.retrieval_scorer`
- `model.prototype_inference_mode`
- `model.prototype_method_role=retrieval_branch` (semantic-structure only)

Validation now fails clearly when removed keys are provided.

## 5. Evaluator / Checkpoint Simplification
- Evaluator now emits a single retrieval row: `host-t2i`.
- Authority context is host-only.
- Removed fusion checkpoint group from module/checkpoint group registries and default checkpointing config.
- Checkpoint authority no longer depends on fused/composer candidates.

## 6. Prototype Capabilities Retained
- Prototype training loss paths remain (including semantic structure losses such as `loss_semantic_pbt`).
- Prototype modules remain available for training-time regularization/structure supervision in `joint_training`.

## 7. Backward Compatibility Breaks
- Configs and CLI flags using fusion/composer/retrieval_scorer are rejected.
- Obsolete runtime modes are rejected.
- Evaluation no longer produces `pas-t2i`, `prototype-t2i`, or fused sweep rows.

## 8. Remaining Work
- Legacy config files in `configs/` still contain removed keys and runtime modes; they now fail validation and should be migrated to host-only retrieval semantics.
- `model/pas_model.py` still contains legacy-named freeze/log fields (`freeze_fusion`) for prototype pooling freeze control naming continuity and should be renamed in a follow-up cleanup if strict terminology alignment is required.
