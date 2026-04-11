# Module Name Summary

This document is the quick reference for module/group names used by:
- `training.freeze_schedule.*`
- `checkpointing.*`
- runtime optimizer and freeze logs

Source of truth in code:
- `utils/module_group_registry.py`
- `utils/freeze_schedule.py`
- `model/pas_model.py` (`named_optimizer_groups`)

## 1) Logical Groups (freeze schedule names)

These are the names accepted in:
- `training.freeze_schedule.trainable_groups`
- `training.freeze_schedule.frozen_groups`
- `training.freeze_schedule.lr_overrides`

| Logical group | What it contains (parameter prefix match) |
| --- | --- |
| `host_backbone` | `base_model.visual.*`, `base_model.transformer.*`, `base_model.token_embedding.*`, `base_model.positional_embedding*`, `base_model.ln_final.*`, `base_model.text_projection*` |
| `host_retrieval` | `host_head.*` |
| `prototype_bank` | `prototype_head.prototype_bank.*` |
| `prototype_projector` | `prototype_head.image_projector.*`, `prototype_head.text_projector.*`, `prototype_head.image_adapter.*`, `prototype_head.text_adapter.*`, `prototype_head.losses.class_proxies*` |
| `routing` | `prototype_head.router.*`, `prototype_head.contextualizer.*` |
| `fusion` | `prototype_head.text_pool_query*`, `prototype_head.token_pooler.*`, `prototype_head.token_scorer.*`, `prototype_head.token_mask_builder.*`, `prototype_head.aggregator.*`, `fusion_module.*` |

## 2) Checkpoint Groups (save/load artifact names)

These are used in:
- `checkpointing.groups`
- `checkpointing.save.artifacts`
- `checkpointing.load.sources`

| Checkpoint group | Logical groups included |
| --- | --- |
| `host` | `host_backbone` + `host_retrieval` |
| `prototype_bank` | `prototype_bank` |
| `prototype_projector` | `prototype_projector` |
| `fusion` | `fusion` |

Note:
- `routing` is not a checkpoint group now. It was removed from save/load artifacts because current routing path is parameter-free.

## 3) Optimizer Group Names (runtime log names)

These are produced by `PASModel.named_optimizer_groups()`:

| Optimizer group | What it contains |
| --- | --- |
| `prototype_bank` | `prototype_head.prototype_bank.*` |
| `prototype_projectors` | `prototype_head.image_projector.*`, `prototype_head.text_projector.*`, `prototype_head.image_adapter.*`, `prototype_head.text_adapter.*` |
| `prototype_routing` | `prototype_head.router.*` |
| `prototype_contextualization` | `prototype_head.contextualizer.*` |
| `prototype_pooling` | `prototype_head.text_pool_query*`, `prototype_head.token_pooler.*`, `prototype_head.token_scorer.*`, `prototype_head.token_mask_builder.*` |
| `class_proxies` | `prototype_head.losses.class_proxies*` |
| `host_projectors` | `host_head.*` |
| `image_backbone` | `base_model.visual.*` |
| `text_backbone` | `base_model.transformer.*`, `base_model.token_embedding.*`, `base_model.positional_embedding*`, `base_model.ln_final.*`, `base_model.text_projection*` |
| `other` | any trainable parameter not matched above |

## 4) Crosswalk: Logical Group -> Optimizer Groups

This is the mapping used by freeze-schedule LR overrides:

| Logical group | Optimizer group(s) affected |
| --- | --- |
| `host_backbone` | `image_backbone`, `text_backbone` |
| `host_retrieval` | `host_projectors` |
| `prototype_bank` | `prototype_bank` |
| `prototype_projector` | `prototype_projectors`, `class_proxies` |
| `routing` | `prototype_routing`, `prototype_contextualization` |
| `fusion` | `prototype_pooling` |

## 5) Practical Notes

- In summary logs, `trainable_groups` and `frozen_groups` are logical names (section 1), while optimizer LR lines use optimizer names (section 3).
- If `routing` shows `0` tensors/params, that is expected for parameter-free routing implementations.
