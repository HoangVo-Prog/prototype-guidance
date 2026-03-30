# Phase B Report

## Files Inspected

- `INTEGRATION_PLAN.md`
- `LEGACY_CODE_MAPPING.md`
- `PHASE_A_REPORT.md`
- `train.py`
- `test.py`
- `configs/default.yaml`
- `configs/baseline_legacy.yaml`
- `utils/config.py`
- `utils/options.py`
- `utils/iotools.py`
- `utils/experiment.py`
- `utils/metrics.py`
- `processor/processor.py`
- `solver/build.py`
- `model/build.py`
- `model/clip_model.py`
- `model/objectives.py`
- `model/grab.py`

## Files Added

- `INTEGRATION_PLAN.md`
- `MODEL_INTERFACE_CONTRACT.md`
- `PHASE_B_REPORT.md`
- `configs/prototype_v1_draft.yaml`
- `model/interfaces.py`
- `model/pooling.py`

## Files Modified

- `configs/default.yaml`
- `configs/baseline_legacy.yaml`
- `train.py`
- `model/build.py`
- `model/clip_model.py`
- `model/objectives.py`
- `processor/processor.py`
- `solver/build.py`
- `utils/config.py`
- `utils/experiment.py`
- `utils/options.py`

## Phase A Repairs Applied During Phase B

- The repository root did not contain `INTEGRATION_PLAN.md` even though the Phase A expectations referenced it. A minimal in-repo integration plan was added so the Phase A artifact set is now complete inside `PAS` itself.
- The previous YAML schema used a partial prototype/config surface. The Phase B schema was expanded and normalized while preserving backward-compatible alias loading for older Phase A-style keys.

## Summary Of Config Schema Updates

The config system is now future-ready and still backward-compatible.

### Major schema additions

- `model`
  - `name`
  - `variant`
  - `image_backbone`
  - `text_backbone`
  - `embedding_dim`
  - `projection_dim`
  - `pooling_mode`
  - `freeze_image_backbone`
  - `freeze_text_backbone`
  - `return_debug_outputs`
  - existing feature flags preserved
- `prototype`
  - `prototype_dim`
  - `prototype_init`
  - `prototype_init_path`
  - `routing_type`
  - `routing_temperature`
  - `contextualization_enabled`
  - `contextualization_type`
  - `contextualization_residual`
  - `contextualization_num_layers`
  - `prototype_normalize`
  - `assignment_sparse`
  - `assignment_topk`
  - `balance_loss_weight`
- `text_pooling`
  - `token_policy`
  - `scoring_type`
  - `token_temperature`
  - `exclude_special_tokens`
  - `eos_as_only_token`
  - `mask_padding_tokens`
- `training`
  - `epochs`
  - `grad_clip`
  - `amp`
  - `seed`
  - explicit freeze flags
- `optimizer`
  - `type`
  - `scheduler`
  - `lr_overrides.prototypes`
  - `lr_overrides.projectors`
  - `lr_overrides.logit_scale`
  - `lr_overrides.backbones`
- `logging`
  - `save_interval`
  - `log_debug_metrics`
- `evaluation`
  - `checkpoint_path`
  - `eval_frequency`
  - `retrieval_metrics`
  - `batch_size`

### Compatibility notes

- `utils/config.py` now distinguishes between the primary Phase B schema and read-only legacy aliases.
- Older keys such as `training.num_epoch`, `optimizer.optimizer`, `prototype.init`, `prototype.temperature`, and `model.text_pooling_type` are still loadable.
- `configs/prototype_v1_draft.yaml` documents the future prototype config surface, but intentionally still activates codepaths that raise `NotImplementedError` until Phase C.

## Summary Of Image Interface Status

The image encoder interface is now explicitly locked.

### New image exposure points

- `model/clip_model.py:CLIP.encode_image_intermediates(...)`
  - exposes pre-projection image tokens
  - exposes projected image tokens
  - preserves the legacy `encode_image(...)` output contract
- `model/build.py:ITSELF.extract_image_features(...)`
  - wraps image outputs in `model/interfaces.py:EncoderOutput`
  - exposes both pooled and token-level representations
  - keeps `extract_image_outputs(...)` and `encode_image(...)` backward-compatible

### Current availability

- Global image embedding `V`: available
- Projected image token sequence: available
- Pre-projection image token sequence: available
- Projected global embedding: available
- Pre-projection global embedding: available
- Attention tensors: available

### Baseline behavior status

- Baseline numerical behavior is unchanged when the new interfaces are unused.
- The public evaluation path still reads `model.encode_image(...)` and therefore still receives the same global embedding type as before.

## Summary Of Text Interface Status

The text encoder interface is now explicitly locked.

### New text exposure points

- `model/clip_model.py:CLIP.encode_text_intermediates(...)`
  - exposes pre-projection token states
  - exposes projected token states
  - preserves the legacy `encode_text(...)` output contract
- `model/build.py:ITSELF.extract_text_features(...)`
  - wraps text outputs in `EncoderOutput`
  - exposes token masks
  - exposes special-token positions
  - exposes pooled pre-projection and post-projection text embeddings

### Current availability

- Token-level hidden states `H`: available
- Pooled text embedding: available
- Attention mask / validity mask: available through `token_mask`
- Special token positions: available through `special_token_positions`
- Projected text embedding: available
- Pre-projection pooled text representation: available

### Baseline behavior status

- Default text pooling remains EOS pooling.
- `encode_text(...)` still returns the same baseline pooled text embedding type used by retrieval evaluation.

## Summary Of Pooling And Projection Boundaries

### Pooling boundary

- New module: `model/pooling.py`
- Locked class: `TextPooler`
- Default path: `eos`
- Additional non-default path exposed: `mean`
- Explicitly reserved and still blocked:
  - `attention`
  - `image_conditioned`

This isolates text pooling from the crowded `ITSELF.forward(...)` block and gives Phase C a clean insertion point for image-conditioned token pooling.

### Projection boundary

Projection exposure now exists at both backbone and wrapper levels.

- Backbone level:
  - `CLIP.encode_image_intermediates(...)`
  - `CLIP.encode_text_intermediates(...)`
- Wrapper level:
  - `ITSELF.extract_image_features(...)`
  - `ITSELF.extract_text_features(...)`

Both sides now expose:

- pre-projection tokens
- projected tokens
- pre-projection pooled embedding
- projected pooled embedding

## Summary Of Similarity / Loss Boundary Status

The similarity and loss surface is now explicitly locked.

### Similarity

- New helper: `model/objectives.py:compute_similarity_logits(...)`
- Wrapper access point: `model/build.py:ITSELF.compute_similarity_matrix(...)`

### Loss assembly

Existing baseline loss math remains in place.

- TAL remains in `model/objectives.py:compute_TAL(...)`
- CID / ID remain in:
  - `compute_cid(...)`
  - `compute_id(...)`
- Loss assembly remains in `model/build.py:ITSELF.forward(...)`

No prototype-specific losses were added in Phase B.

## Summary Of Debug Output Plumbing

Debug output plumbing now exists and is optional.

### Activation

- Config: `model.return_debug_outputs`
- Runtime call: `ITSELF.forward(..., return_debug=True)`

### Output shape

`ITSELF.forward(...)` may now return `ret['debug']` containing:

- image-side pooled tensors
- text token tensors
- text pooled tensors
- token masks
- pooling mode metadata
- logit scale
- text token count
- padding fraction
- special token positions

The training loop remains backward-compatible because only keys containing `loss` are summed.

## Summary Of Logging Readiness

The logging surface is ready for later prototype metrics.

### W&B / metric plumbing

- `utils/experiment.py` now supports nested debug dictionaries.
- Scalar-like entries inside `ret['debug']` are promoted automatically to `debug/...` metrics.
- New Phase B-ready debug metrics include:
  - `debug/logit_scale`
  - `debug/text_token_count`
  - `debug/padding_fraction`
  - `debug/pooling_mode`
- Existing future placeholders remain supported:
  - `debug/prototype_usage_entropy`
  - `debug/prototype_dead_count`
  - `debug/token_pool_entropy`
  - `debug/token_special_mass`
  - `debug/routing_max_prob`

### Training-loop readiness

- `processor/processor.py` now supports:
  - `grad_clip`
  - `save_interval`
  - a warning when `training.amp` is configured but explicit AMP remains intentionally inactive
- `solver/build.py` now supports future LR overrides for:
  - prototypes
  - projectors
  - logit scale
  - backbones

## Known Blockers For Phase C

- Prototype bank modules are still not implemented.
- Image-conditioned pooling is still intentionally blocked.
- Prototype contextualization is still intentionally blocked.
- Prototype diversity and balance losses are still intentionally blocked.
- `logit_scale` is still a wrapper-owned tensor rather than an actively optimized parameter in the baseline path.
- The current environment does not include `PyYAML`, so dynamic `get_args()` verification could not be executed even though static compilation passed.
- The repository still effectively assumes the ViT-style CLIP path for token-level image features; non-ViT image backbones remain outside the intended research path.

## Recommended Exact Next Implementation Order For Phase C

1. Add a dedicated prototype-bank module that consumes `image_output.projected_pooled` (and optionally `pre_projection_pooled`) from `ITSELF.extract_image_features(...)`.
2. Add prototype routing outputs and debug stats while keeping them behind `use_prototype_bank`.
3. Add prototype contextualization behind `use_prototype_contextualization`, reusing the now-locked prototype config section.
4. Extend `TextPooler` with image-conditioned token scoring and weighted pooling behind `use_image_conditioned_pooling` / `pooling_mode: image_conditioned`.
5. Thread the new pooled text embedding back through `ITSELF.compute_similarity_matrix(...)` and the existing TAL/CID objective path.
6. Add prototype-specific auxiliary losses and scalar debug metrics, returning them in the existing `ret` dict and nested `ret['debug']` surface.
7. Add focused tests for config loading, encoder interface shapes, and optional debug output keys.

## Verification Performed

- `python -m compileall PAS`
  - passed
- live `get_args()` smoke test
  - blocked by missing `PyYAML` in the current environment (`ModuleNotFoundError: No module named 'yaml'`)

## Phase B Outcome

Phase B is complete. The repository now has a concrete future-ready config schema, locked image/text/pooling/projection/similarity/debug interfaces, logging and optimizer surfaces ready for prototype metrics, and explicit documentation for Phase C implementation. The active default path remains baseline-compatible and no prototype-method logic has been implemented yet.

