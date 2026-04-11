# Modular Checkpoint Refactor Report

## Summary
This refactor introduces a modular checkpoint system that saves/loads smaller named parameter groups independently instead of relying only on full-model checkpoint flows.

Primary checkpoint groups are now:
- `host`
- `prototype_bank`
- `prototype_projector`
- `fusion`

The previous coarse `prototype_best` checkpoint-selection flow in training was replaced by a centralized modular flow driven by validation `R1`.

## New Config Structure
New unified section:

```yaml
checkpointing:
  metric:
    name: R1
    mode: max
  groups:
    host: { enabled: true }
    prototype_bank: { enabled: true }
    prototype_projector: { enabled: true }
    fusion: { enabled: true }
  save:
    dir: null
    save_latest: true
    save_best: true
    keep_last_n: 1
    artifacts:
      host:
        enabled: true
        filename_latest: checkpoint_host_latest.pth
        filename_best: checkpoint_host_best.pth
      prototype_bank:
        enabled: true
        filename_latest: checkpoint_prototype_bank_latest.pth
        filename_best: checkpoint_prototype_bank_best.pth
      prototype_projector:
        enabled: true
        filename_latest: checkpoint_prototype_projector_latest.pth
        filename_best: checkpoint_prototype_projector_best.pth
      fusion:
        enabled: true
        filename_latest: checkpoint_fusion_latest.pth
        filename_best: checkpoint_fusion_best.pth
  load:
    enabled: false
    strict: true
    sources:
      host: { enabled: false, path: null }
      prototype_bank: { enabled: false, path: null }
      prototype_projector: { enabled: false, path: null }
      fusion: { enabled: false, path: null }
```

## Single Source of Truth: Group Mapping
Implemented in:
- `utils/module_group_registry.py`

Key APIs:
- `get_group_state_dict(model, group_name)`
- `load_group_state_dict(model, group_name, state_dict, strict=True)`

Mapping:
- `host` = `host_backbone` + `host_retrieval`
  - `host_backbone`: `base_model.visual`, `base_model.transformer`, `base_model.token_embedding`, `base_model.positional_embedding`, `base_model.ln_final`, `base_model.text_projection`
  - `host_retrieval`: `host_head.*`
- `prototype_bank`: `prototype_head.prototype_bank.*`
- `prototype_projector`: `prototype_head.image_projector.*`, `prototype_head.text_projector.*`, `prototype_head.image_adapter.*`, `prototype_head.text_adapter.*`, `prototype_head.losses.class_proxies`
- `fusion`: `prototype_head.text_pool_query.*`, `prototype_head.token_pooler.*`, `prototype_head.token_scorer.*`, `prototype_head.token_mask_builder.*`, `prototype_head.aggregator.*`, `fusion_module.*`

## Host Abstraction (`host.type`)
Checkpointing is backend-agnostic and does not branch on `itself` vs `clip` for group extraction.  
It relies on actual parameter names in the active model:
- if `host.type=itself`, `host_head` points to ITSELF host implementation parameters
- if `host.type=clip`, `host_head` points to CLIP host implementation parameters

Thus the same `host` checkpoint group works across host backends via module structure.

## Save Flow
Implemented in:
- `utils/modular_checkpoint.py` (`ModularCheckpointManager`)
- wired from `processor/processor.py`

Behavior:
- at validation (`R1` available), saves `latest` per enabled group when `checkpointing.save.save_latest=true`
- tracks best by validation `R1` (centralized), and when improved, saves `best` per enabled group when `checkpointing.save.save_best=true`

Each saved payload includes:
- `group_name`
- `state_dict` (only group keys)
- `epoch`
- `global_step`
- `metric` (`name`, `value`, `mode`)
- `metadata` (`host_type`, `run_name`, `training_stage`)

## Load Flow
Implemented in:
- `utils/modular_checkpoint.py` + `utils/module_group_registry.py`
- wired from `train.py` and `test.py`

Behavior:
- load independently per group from `checkpointing.load.sources.<group>.path`
- strictness controlled by `checkpointing.load.strict`
- logs for each group:
  - group name
  - path
  - strict mode
  - loaded keys
  - missing keys
  - unexpected keys
  - shape mismatches

## Best-by-R1 Logic
Centralized in `ModularCheckpointManager`:
- selection metric is validation `R1`
- best improvement updates all enabled `filename_best` artifacts across groups

`training.prototype_selection_metric` is now deprecated in training flow (warning emitted).

## Old Config Path Status
Old coarse load path `training.finetune` is no longer the primary path:
- primary path is `checkpointing.load.*`
- `training.finetune` remains as compatibility fallback with deprecation warning

## Updated Configs
Updated examples:
- `configs/base.yaml`
- `configs/head_type/itself/direction2_stage1_prototype_stability.yaml`
- `configs/head_type/itself/direction2_stage2_optionA_frozen_transfer.yaml`
- `configs/head_type/itself/direction2_stage2_optionB_lowlr_adaptation.yaml`

## Caveats
- Existing full-model `best.pth` / `last.pth` saving via `Checkpointer` is still preserved for compatibility and resume workflows.
- `checkpointing.metric.name` currently falls back to `R1` selection if set to other values.
- Runtime schema validation now accepts and validates the hierarchical `checkpointing` section.
- Routing/contextualizer modules remain in training/freezing controls, but were removed from checkpoint save/load groups because they are parameter-free in the current implementation.
