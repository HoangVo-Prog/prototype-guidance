# Phase A Report

## Files Inspected

- `train.py`
- `test.py`
- `datasets/build.py`
- `datasets/bases.py`
- `model/build.py`
- `model/clip_model.py`
- `model/grab.py`
- `model/objectives.py`
- `processor/processor.py`
- `solver/build.py`
- `solver/lr_scheduler.py`
- `utils/checkpoint.py`
- `utils/comm.py`
- `utils/iotools.py`
- `utils/logger.py`
- `utils/metrics.py`
- `utils/options.py`
- `INTEGRATION_PLAN.md`

## Files Added

- `LEGACY_CODE_MAPPING.md`
- `PHASE_A_REPORT.md`
- `configs/default.yaml`
- `configs/baseline_legacy.yaml`
- `utils/config.py`
- `utils/experiment.py`

## Files Modified

- `train.py`
- `test.py`
- `model/build.py`
- `processor/processor.py`
- `utils/iotools.py`
- `utils/logger.py`
- `utils/metrics.py`
- `utils/options.py`

## Files Removed To Restore Phase A Scope

- `model/prototype/`
- `tests/test_prototype_modules.py`

## Summary Of Current Training / Eval Pipeline

Training starts in `train.py`, which parses CLI plus YAML config, creates the output directory, saves resolved configs, initializes logging and optional W&B tracking, then builds dataloaders, model, optimizer, scheduler, evaluator, and checkpoint manager. The main training loop lives in `processor/processor.py:do_train(...)`, where the model returns a dict of losses and metadata, losses are summed by key name, TensorBoard metrics are written, checkpoints are saved, and retrieval evaluation is run periodically.

Evaluation starts in `test.py`, which can load either an explicit YAML config or a saved run config from the output directory. It restores the checkpoint with `utils/checkpoint.py`, builds the evaluation dataloaders, and runs `processor/processor.py:do_inference(...)`. Retrieval metrics are computed by `utils/metrics.py` using `model.encode_text(...)`, `model.encode_image(...)`, optional GRAB embeddings, cosine similarity, and the existing fusion table.

## Summary Of Config System Introduced

A backward-compatible YAML config system was added through `utils/config.py` and wired into `utils/options.py`. The parser still exists and remains the public CLI surface, but values can now come from:

1. parser defaults
2. `configs/default.yaml`
3. an optional override YAML via `--config_file`
4. explicit CLI args, which still take precedence

Training now saves both a flat legacy-compatible `configs.yaml` and a structured `resolved_config.yaml`. Evaluation can reload either one. The config schema now includes sections for experiment, model, prototype, training, optimizer, dataset, logging, evaluation, and loss.

## Summary Of W&B Integration

W&B integration is implemented in `utils/experiment.py` and only initializes when enabled by config or CLI. The training loop uses `ExperimentTracker` to log:

- `train/loss`
- `train/lr`
- `train/epoch`
- `train/step`
- validation retrieval metrics exported by `utils/metrics.py`

The logging utility also includes reserved debug metric hooks for future phases:

- `debug/prototype_usage_entropy`
- `debug/prototype_dead_count`
- `debug/token_pool_entropy`
- `debug/token_special_mass`
- `debug/routing_max_prob`
- `debug/logit_scale`

These keys are optional and only log if the model output provides them later.

## Known Blockers For Phase B And Phase C

- `model/build.py` is still the most crowded integration point because it mixes baseline global retrieval, GRAB logic, and loss assembly in one wrapper.
- There is still no standalone retrieval projector module or structured model output object.
- The evaluation pipeline does not compute validation loss, so `val/loss` is not currently available.
- `train.py` and `test.py` still assume CUDA by default.
- `datasets/build.py` still has asymmetric return signatures between train and eval modes and an incomplete distributed sampler path.
- The current environment used for verification does not include runtime dependencies such as PyTorch and PyYAML, so verification in this phase was limited to static code checks rather than execution.

## Phase A Outcome

The repository now supports the legacy baseline path with added YAML config support and optional W&B logging, while keeping prototype-guided interaction reserved for future phases only. No prototype-bank or image-conditioned pooling method logic is implemented in the active model path.
