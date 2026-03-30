# Integration Plan

## Scope

This repository is now centered on the PAS image-text retrieval model. The inherited CLIP backbone and generic training infrastructure are retained, while the old ITSELF-specific retrieval path has been retired from the active code path.

## Active Rules

1. Keep training, evaluation, checkpointing, and dataset code generic and reusable.
2. Keep the PAS method config-driven so ablations are easy to express.
3. Prefer small, composable modules over large monolithic forward logic.
4. Add diagnostics through shared metric utilities rather than ad hoc logging.
5. Preserve historical phase reports as documentation, but keep active docs aligned with the current model.

## Completed Phases

### Phase A
- Repository audit, YAML config groundwork, and wandb hooks.

### Phase B
- Interface locking for image/text encoders, pooling, projection, similarity, and debug outputs.

### Phase C
- Implementation of prototype bank, contextualizer, router, aggregator, token scorer, masking, pooling, projectors, and losses.

### Phase D
- End-to-end forward integration into training and retrieval evaluation.

### Phase E
- Promotion of the PAS model to the primary path.
- Cleanup of obsolete ITSELF-specific branches, configs, smoke harnesses, and naming.
- Experiment-ready configs, optimizer groups, freeze policy, and structured diagnostics.

## Recommended Next Phase

1. Run real dataset training with `configs/train_pas_v1.yaml`.
2. Validate ablations in `configs/ablation_pas_no_context.yaml` and `configs/ablation_pas_no_diversity.yaml`.
3. Add richer prototype-analysis metrics and visualization utilities.
4. If needed, extend the method to support patch-token image routing as a later variant rather than changing the v1 path.

