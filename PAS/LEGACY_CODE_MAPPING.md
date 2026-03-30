# Legacy Code Mapping

## Scope

This document maps the current `PAS` repository as the legacy baseline that will be extended in later phases. Phase A adds config and logging groundwork only. The prototype-bank method is not implemented in the active codepath.

## Main Entry Points

### Training

- `train.py`
  - Parses args through `utils/options.py:get_args(...)`
  - Builds dataloaders through `datasets.build_dataloader(args)`
  - Builds model through `model.build_model(args, num_classes)`
  - Builds optimizer and scheduler through `solver.build_optimizer(...)` and `solver.build_lr_scheduler(...)`
  - Creates:
    - `utils.checkpoint.Checkpointer`
    - `utils.metrics.Evaluator`
    - `utils.experiment.ExperimentTracker`
  - Calls `processor.processor.do_train(...)`

### Evaluation

- `test.py`
  - Parses args through `utils/options.py:get_args(...)`
  - Optionally reloads `resolved_config.yaml` or `configs.yaml` from a saved run
  - Builds dataloaders through `datasets.build_dataloader(args)`
  - Builds model through `model.build_model(args, num_classes)`
  - Loads checkpoint through `utils.checkpoint.Checkpointer.load(...)`
  - Calls `processor.processor.do_inference(...)`

## Model Mapping

### Main model entrypoint

- `model/build.py`
  - `build_model(args, num_classes=11003)`
  - `class ITSELF(nn.Module)`

### Backbone wrappers

- `model/clip_model.py`
  - `CLIP.encode_image(image)`
  - `CLIP.encode_text(text)`
  - `CLIP.encode_image_all_atten(image, average_attn_weights=True)`
  - `CLIP.encode_text_all_atten(text, average_attn_weights=True)`
  - `CLIP.forward(image, text, return_all=False, average_attn_weights=True)`

### Image encoder wrapper path

- `model/build.py:80` `ITSELF.extract_image_outputs(image)`
  - Returns `(image_global, image_tokens, atten_i)`
- `model/build.py:92` `ITSELF.encode_image(image)`
  - Public inference API used by evaluation

### Text encoder wrapper path

- `model/build.py:86` `ITSELF.extract_text_outputs(text)`
  - Returns `(text_global, text_tokens, atten_t)`
- `model/build.py:96` `ITSELF.encode_text(text)`
  - Public inference API used by evaluation

### Text pooling logic

- `model/build.py:77` `ITSELF._pool_text_tokens(text_tokens, token_ids)`
  - Current baseline behavior: EOS pooling only
- Important:
  - `text_pooling_type` exists in config as a reserved field for future work
  - In Phase A, any non-`eos` value raises `NotImplementedError` to preserve legacy math

### Local GRAB branch

- `model/grab.py`
- `model/build.py`
  - `ITSELF.encode_image_grab(image)`
  - `ITSELF.encode_text_grab(text)`
  - `_compute_grab_features(...)`
- Controlled by existing legacy flags:
  - `only_global`
  - `return_all`
  - `topk_type`
  - `layer_index`
  - `modify_k`

### Projection heads

- Backbone-level CLIP projections in `model/clip_model.py`
  - `self.visual.proj`
  - `self.text_projection`
- Wrapper-level heads in `model/build.py`
  - `self.mlp_global`
  - `self.classifier_global`
  - `self.classifier_id_global`
  - `self.mlp_grab`
  - `self.classifier_grab`
  - `self.classifier_id_grab`
- Current state:
  - Projection/classification logic is embedded in `ITSELF.forward(...)`
  - There is no dedicated standalone projector module yet

## Loss Mapping

- `model/objectives.py`
  - `compute_TAL(...)`
  - `cosine_similarity_matrix(...)`
  - `sample_hard_negatives(...)`
  - `update_labels_for_negatives(...)`
  - `create_sample_pairs(...)`
  - `compute_cid(...)`
  - `compute_id(...)`
  - `compute_prototype_diversity_loss(...)`
- Active loss assembly lives in `model/build.py:169` `ITSELF.forward(...)`
- Current active baseline loss path:
  - `tal`
  - `cid`
- Reserved future loss path:
  - `use_diversity_loss`
  - Currently blocked in Phase A with `NotImplementedError`

## Train / Eval / Checkpoint Flow

### Train loop

- `processor/processor.py`
  - `do_train(...)`
  - Assumes `model(batch, epoch, current_step)` returns a dict
  - Sums every returned key containing `loss`
  - Logs to TensorBoard
  - Logs to W&B only if enabled via config

### Eval / retrieval pipeline

- `utils/metrics.py`
  - `class Evaluator`
  - `_compute_embedding(...)`
  - `_compute_embedding_grab(...)`
  - `eval(...)`
- Current retrieval behavior:
  - Calls `model.encode_text(...)` and `model.encode_image(...)`
  - L2 normalizes both sides
  - Uses cosine similarity
  - Optionally mixes `global` and `grab` scores using the existing fixed-weight fusion table
- Retrieval metrics exposed for logging:
  - `R1`
  - `R5`
  - `R10`
  - `mAP`
  - `mINP`
  - `rSum`

### Checkpoint save/load flow

- `utils/checkpoint.py`
  - `Checkpointer.save(name, **kwargs)`
  - `Checkpointer.load(f=None)`
  - `Checkpointer.resume(f=None)`
  - `load_state_dict(model, loaded_state_dict, except_keys=None)`
- Train path:
  - `train.py` creates `Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)`
- Eval path:
  - `test.py` creates `Checkpointer(model)` then loads a checkpoint file
- Compatibility note:
  - checkpoint loading strips `module.` prefixes and aligns keys by suffix match

## Config And Logging Mapping

### Current argument/config system

- `utils/options.py`
  - `build_parser()`
  - `get_args(argv=None)`
- `utils/config.py`
  - `load_yaml_config(default_path=None, override_path=None)`
  - `apply_config_to_args(parser, args, config_data, argv=None)`
  - `build_runtime_config(args)`
  - `flatten_config_dict(config_data)`
  - `dump_yaml_config(path, config_data)`
- Default config files:
  - `configs/default.yaml`
  - `configs/baseline_legacy.yaml`
- Backward-compatible persistence:
  - `utils/iotools.py:save_train_configs(...)`
    - writes `configs.yaml` (flat legacy-style resolved args)
    - writes `resolved_config.yaml` (structured nested config)
  - `utils/iotools.py:load_train_configs(...)`
    - reads either nested or flat YAML

### Current logging system

- `utils/logger.py`
  - `setup_logger(...)`
  - console + file logging
- TensorBoard:
  - `processor/processor.py` uses `SummaryWriter`
- W&B integration:
  - `utils/experiment.py`
    - `ExperimentTracker`
    - `build_iteration_metrics(...)`
    - `build_epoch_metrics(...)`
    - `build_validation_metrics(...)`
  - Activated only if `logging.use_wandb: true` or `--use_wandb`
- Standard W&B training metrics currently supported:
  - `train/loss`
  - `train/lr`
  - `train/epoch`
  - `train/step`
- Validation/retrieval metrics currently supported:
  - `val/top1`
  - `val/<retrieval_head>/R1`
  - `val/<retrieval_head>/R5`
  - `val/<retrieval_head>/R10`
  - `val/<retrieval_head>/mAP`
  - `val/<retrieval_head>/mINP`
  - `val/<retrieval_head>/rSum`
- Future debug placeholders already structured in W&B logging:
  - `debug/prototype_usage_entropy`
  - `debug/prototype_dead_count`
  - `debug/token_pool_entropy`
  - `debug/token_special_mass`
  - `debug/routing_max_prob`
  - `debug/logit_scale`

## Reuse / Refactor / Do-Not-Touch Map

### Reuse as-is

- `datasets/cuhkpedes.py`
- `datasets/icfgpedes.py`
- `datasets/rstpreid.py`
- `datasets/sampler.py`
- `datasets/sampler_ddp.py`
- `datasets/build.py`
- `datasets/bases.py`
- `model/clip_model.py`
- `model/grab.py`
- `model/simple_tokenizer.py`
- `utils/simple_tokenizer.py`
- `utils/checkpoint.py`
- `utils/comm.py`
- `solver/lr_scheduler.py`

Reason:
These modules already provide stable dataset, backbone, tokenizer, scheduler, and checkpoint behavior and should remain the inheritance base.

### Need light refactor only

- `train.py`
- `test.py`
- `utils/options.py`
- `utils/config.py`
- `utils/iotools.py`
- `utils/logger.py`
- `utils/experiment.py`
- `processor/processor.py`
- `utils/metrics.py`

Reason:
These are orchestration and infrastructure files. They are the correct place for config, experiment tracking, and future additive logging.

### Should not be touched unless necessary

- `model/clip_model.py`
- `utils/checkpoint.py`
- `datasets/*` record parsing and tokenization flow
- `solver/lr_scheduler.py`

Reason:
These are compatibility-sensitive foundations. Changing them would create broad regression risk.

## Current Public Interfaces

### Data

- `datasets.build_dataloader(args, tranforms=None)`
  - train mode returns `(train_loader, val_img_loader, val_txt_loader, num_classes)`
  - eval mode returns `(test_img_loader, test_txt_loader, num_classes)`

### Model

- `model.build_model(args, num_classes=11003)`
- `ITSELF.forward(batch, epoch=None, current_step=None)`
- `ITSELF.encode_image(image)`
- `ITSELF.encode_text(text)`
- `ITSELF.encode_image_grab(image)`
- `ITSELF.encode_text_grab(text)`
- `ITSELF.extract_image_outputs(image)`
- `ITSELF.extract_text_outputs(text)`

### Training / eval

- `processor.processor.do_train(...)`
- `processor.processor.do_inference(...)`
- `utils.metrics.Evaluator.eval(model, i2t_metric=False)`

### Config / logging

- `utils.options.get_args(argv=None)`
- `utils.config.load_yaml_config(...)`
- `utils.config.apply_config_to_args(...)`
- `utils.config.build_runtime_config(args)`
- `utils.experiment.ExperimentTracker`

## Where The Future Method Should Attach Later

### Primary attachment point

- `model/build.py`
  - This is the correct insertion point for future interaction logic because it already owns:
    - wrapper-level image feature extraction
    - wrapper-level text feature extraction
    - text pooling
    - loss assembly

### Secondary attachment points

- `ITSELF.extract_image_outputs(...)`
  - Good place to consume image global embeddings and token sequences
- `ITSELF.extract_text_outputs(...)`
  - Good place to consume text pooled embeddings and token states
- `utils/metrics.py`
  - Good place to add future retrieval-side logging or scoring variants
- `processor/processor.py`
  - Good place to log future auxiliary metrics

## What Already Exists vs What Is Missing

### Already exposed

- Image global embedding:
  - Yes
  - via `ITSELF.encode_image(...)` and `ITSELF.extract_image_outputs(...)`
- Text token hidden states:
  - Yes
  - via `CLIP.encode_text(...)` and `ITSELF.extract_text_outputs(...)`
- Text pooled embedding:
  - Yes
  - via `ITSELF.encode_text(...)`
- Projection outputs:
  - Partially
  - backbone projections exist, but there is no clean standalone retrieval projector interface
- Similarity computation:
  - Yes
  - via `utils.metrics.py` and `model/objectives.py:cosine_similarity_matrix(...)`

### Missing for future integration

- Dedicated retrieval projector module with a clean public interface
- Structured model output object instead of a raw dict
- Future interaction module namespace and implementation files
- Validation loss computation in the eval pipeline
- CPU-safe runtime defaults in training/eval entrypoints

## Baseline-Specific Codepaths And Risks

### Baseline-specific codepaths

- GRAB branch in `model/grab.py` and `model/build.py`
- Fixed global/grab fusion table in `utils/metrics.py`
- `loss_names` controls `tal` and `cid` composition

### Hardcoded assumptions that may block future integration

- `train.py`
  - assumes CUDA training device
  - rewrites `args.output_dir` using timestamp and dataset name
- `test.py`
  - defaults checkpoint to `best.pth` inside `args.output_dir`
  - defaults device to `cuda`
- `processor/processor.py`
  - assumes every output key containing `loss` should be summed
- `datasets/build.py`
  - returns different tuple shapes depending on `args.training`
  - distributed identity sampler path is still incomplete

## Reserved Future Files Likely Needed Later

- `model/projectors.py`
- `model/outputs.py`
- `utils/metrics_prototype.py` or expansion of `utils/metrics.py`
- `configs/prototype_*.yaml`
- `tests/test_config_loading.py`
- `tests/test_training_entrypoint.py`

## Phase A Constraint

The repository now includes reserved config fields for prototype-guided interaction, but no prototype-bank method logic is active. If those future-facing flags are enabled in Phase A, the model raises `NotImplementedError` by design.

