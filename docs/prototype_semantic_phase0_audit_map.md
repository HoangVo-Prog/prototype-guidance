# Phase 0 Audit Map: Prototype Method Rewrite

This note records current ownership before the semantic-structure rewrite path.

## Runtime Owners
- Host core runtime orchestration: `model/plug_and_play.py` (`HostCore`, `PrototypePlugin`, `Composer`, `PASRuntimeModel`)
- Legacy monolithic PAS path (still supported): `model/pas_model.py`
- Runtime mode selection/router: `model/build.py`, `model/runtime_modes.py`

## Prototype Path Owners
- Prototype head build/wiring: `model/prototype/build.py`
- Prototype bank definition/init: `model/prototype/prototype_bank.py`
- Routing (global/local): `model/prototype/router.py`
- Contextualization: `model/prototype/contextualizer.py`
- Basis-bank construction: `model/prototype/head.py::build_text_basis_bank`
- Surrogate text reconstruction: `model/prototype/head.py::reconstruct_surrogate_text`
- Exact diagonal teacher text path: `model/prototype/head.py::pool_text_with_summary`
- Prototype retrieval score path (exact/approx): `model/prototype/head.py::{compute_pairwise_similarity, compute_surrogate_pairwise_logits, compute_approximate_pairwise_similarity}`
- Prototype-only direct/no-bank head: `model/prototype/direct_head.py`

## Loss Owners
- Legacy prototype losses (ret/diag/support/div/balance and aliases): `model/prototype/losses.py`
- Trainer loss aggregation and logging surfaces: `model/plug_and_play.py`, `model/pas_model.py`, `processor/processor.py`, `utils/metric_logging.py`

## Config/Schema Owners
- CLI defaults + arg finalization: `utils/options.py`
- YAML schema mapping/validation + aliases: `utils/config.py`
- Canonical config examples: `configs/base.yaml`, `configs/schema_pas_reference.yaml`

## Composer/Fusion Owners
- Similarity fusion implementation: `model/fusion.py`
- Runtime fusion invocation: `model/plug_and_play.py`, `model/pas_model.py`

## Checkpoint/Interface Owners
- Host->plugin interface contract: `model/interface_contract.py`
- Modular checkpoint utilities: `utils/modular_checkpoint.py`

## Notes
- Legacy retrieval-style prototype path remains intact and selectable by config.
- New semantic-structure mode is added behind explicit config gating.
