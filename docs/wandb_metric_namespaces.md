# W&B Metric Namespaces

This repository follows the metric taxonomy in [`prototype_metrics_taxonomy.md`](../prototype_metrics_taxonomy.md).

## Source Of Truth In Code

- Central mapping logic lives in [`utils/metric_logging.py`](../utils/metric_logging.py).
- Training key mapping is handled by:
  - `map_train_scalar_to_wandb_key(...)`
  - `map_train_diagnostic_key(...)`
- Validation key mapping is handled by:
  - `_map_val_loss_key(...)`
  - `map_validation_metric_key(...)`
  - `build_validation_debug_metrics(...)`
  - `build_validation_retrieval_metrics(...)`

## Current Namespace Layout

- Training:
  - `train/epoch`, `train/step`, `train/lr`
  - `train/loss/*`
  - `train/loss_weighted/*`
  - `train/model/*`
  - `train/prototype_usage/*`
  - `train/routing/*`
  - `train/fidelity/*`
  - `train/geometry/*`
  - `train/proxy/*`
  - `train/prototype_geometry/*`
  - `train/token_pool/*`
  - `train/norm/*`
  - `train/grad/*`
- Validation:
  - `val/loss/*`
  - `val/loss_weighted/*`
  - `val/retrieval/*`
  - `val/model/*`
  - `val/data/*`
  - `val/geometry/*`
  - `val/norm/*`

## Adding A New Metric

1. Keep metric computation where it already belongs (model/processor/evaluator).
2. Add the raw scalar key to existing metric collection flow if needed.
3. Add exactly one namespace rule in `utils/metric_logging.py`:
   - training diagnostics: `map_train_diagnostic_key(...)`
   - validation debug metrics: `_VAL_DEBUG_NAMESPACE_MAP`
   - loss metrics: `_LOSS_BASE_SUFFIX_MAP` or `_LOSS_WEIGHTED_SUFFIX_MAP`
4. Add or update a focused test in [`tests/test_metric_logging.py`](../tests/test_metric_logging.py).

The intended pattern is: preserve metric semantics, centralize naming at emission time, and avoid ad hoc renames across call sites.
