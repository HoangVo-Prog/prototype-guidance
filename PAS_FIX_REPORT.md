# PAS Fix Report

## 1. Summary of fixed issues
- Hardcoded special-token handling: fixed by implementation. Runtime masking now uses explicit `text_pooling.special_token_ids` metadata and validated special-token positions instead of assuming `cls=position 0`, `eos=argmax(token_ids)`, or `pad=0`.
- Fake masking config knobs: fixed by removal and explicit rejection. `exclude_special_tokens`, `eos_as_only_token`, and `mask_padding_tokens` were removed from the live config surface, and `token_policy` is now the only masking behavior control.
- Spec-runtime config mismatch for method knobs: fixed by implementation and explicit rejection. `projector_type`, `learn_logit_scale`, `use_balancing_loss`, `special_token_ids`, and `error_on_empty_kept_tokens` are now real runtime surfaces; unsupported normalization override knobs now fail loudly.
- Dead `pooling_mode` config surface: fixed by removal and explicit rejection. PAS v1 now exposes only the real image-conditioned pooling path.
- Misleading contextualizer optimizer knobs: fixed by removal and explicit rejection. No contextualizer optimizer group or contextualizer-specific optimizer config remains because the contextualizer is parameter-free.
- AMP config inconsistency: fixed by removal and explicit rejection. `training.amp` no longer exists as a no-op setting.

## 2. Files changed
- `PAS/model/prototype/token_mask.py`: replaced positional CLIP assumptions with explicit special-token-id driven masking and strict per-sample validation.
- `PAS/model/prototype/projector.py`: added real `projector_type` support for `mlp2` and `linear`.
- `PAS/model/prototype/losses.py`: wired `learn_logit_scale` / `use_balancing_loss` behavior into the live loss module and added consistency checks.
- `PAS/model/prototype/build.py`: passed new runtime config surfaces into the prototype head and validated balance-loss settings.
- `PAS/model/prototype/head.py`: threaded the new masking and projector controls through the active PAS head without changing the core pipeline.
- `PAS/model/build.py`: removed hardcoded text-token assumptions, required explicit `special_token_ids`, reused the mask builder for text feature extraction, and removed the contextualizer optimizer bucket.
- `PAS/solver/build.py`: removed contextualizer-specific optimizer group mappings.
- `PAS/utils/options.py`: removed dead parser surfaces and added the live PAS method knobs that now affect runtime.
- `PAS/utils/config.py`: aligned YAML-to-runtime mapping with the real method surface and added loud rejection for removed/unsupported keys.
- `PAS/configs/default.yaml`: synchronized the default config with the real runtime surface.
- `PAS/configs/train_pas_v1.yaml`: synchronized the training config with the real runtime surface.
- `PAS/configs/debug_pas_v1.yaml`: synchronized the debug config with the real runtime surface.
- `PAS/configs/ablation_pas_no_context.yaml`: synchronized the ablation config with the real runtime surface.
- `PAS/configs/ablation_pas_no_diversity.yaml`: synchronized the ablation config with the real runtime surface.
- `PAS/configs/kaggle_pas_quicktrain.yaml`: synchronized the quick-train config with the real runtime surface.
- `PAS/configs/schema_pas_reference.yaml`: updated the commented schema to document only supported runtime/config behavior.
- `PAS/MODEL_INTERFACE_CONTRACT.md`: updated the model contract to reflect explicit token metadata and current optimizer groups.
- `PAS/MODULE_CONTRACTS.md`: updated module contracts for the new masking, projector, and loss configuration surfaces.
- `PAS/ARCHITECTURE.md`: synchronized the architecture notes with the active optimizer/runtime surface.
- `PAS/tests/test_prototype_modules.py`: added focused masking, projector, and loss-surface tests.
- `PAS/tests/test_phase_e_integration.py`: added integration checks for prototype participation in inference and train/inference representation consistency.
- `PAS/tests/test_config_surface.py`: added config-honesty tests for removed, supported, and loudly rejected surfaces.
- `PAS/scripts/phase_e_smoke.py`: synchronized the smoke script with the real runtime args/config surface.

## 3. Behavior changes
- Token masking is now driven by `text_pooling.special_token_ids` plus optional explicit `special_token_positions`, with loud failure when a token policy requires metadata that is missing for any sample.
- `text_pooling.token_policy` is now the only masking behavior switch. The removed masking booleans no longer appear as usable runtime options.
- `model.projector_type` now changes projector behavior for real. Supported values are `mlp2` and `linear`.
- `model.learn_logit_scale` now controls whether the contrastive logit scale is a trainable parameter or a fixed buffer.
- `prototype.use_balancing_loss` must agree with `prototype.balance_loss_weight`; inconsistent settings now raise errors instead of silently behaving like the default path.
- `model.pooling_mode`, `training.amp`, `optimizer.lr_contextualizer`, and `optimizer.weight_decay_contextualizer` were removed from the live config surface and now raise clear config errors if supplied.
- Unsupported normalization override knobs are no longer silently ignored; they now fail loudly because PAS v1 keeps normalization fixed at the current method-preserving sites.

## 4. Preserved method logic
- The active PAS runtime still follows the same core method: image encoding -> prototype bank/contextualization -> image-to-prototype routing -> summary-conditioned token scoring/masking -> image-conditioned text pooling -> projector heads -> retrieval scoring.
- The one-way interaction direction remains unchanged: image features condition text pooling, but text features do not update image features.
- The prototype bank still participates in both training and inference scoring.
- Train and inference still use the same representation family for retrieval scoring.
- No legacy retrieval or optimizer path was reintroduced.

## 5. Tests added or updated
- `PAS/tests/test_prototype_modules.py`
  Verifies token masking uses configured token ids instead of positional assumptions, fails loudly when required metadata is missing, validates per-sample special-token requirements, exercises `linear` projector mode, and checks `learn_logit_scale` / balance-loss config behavior.
- `PAS/tests/test_phase_e_integration.py`
  Verifies the contextualizer optimizer group is absent, inference similarity changes when the prototype bank changes, and retrieval text features reuse the same token-state family as the training path.
- `PAS/tests/test_config_surface.py`
  Verifies removed fake surfaces are rejected, unsupported method knobs fail loudly, and supported new surfaces load through the config/parser path.

## 6. Remaining limitations
- PAS v1 still supports only the image-conditioned pooling path; `model.pooling_mode` is intentionally unsupported and now rejected.
- PAS v1 keeps normalization behavior fixed at the current runtime sites; config overrides such as `model.normalize_projector_outputs`, `prototype.normalize_for_self_interaction`, `prototype.normalize_for_routing`, and `text_pooling.normalize_for_token_scoring` remain intentionally unsupported and now fail loudly.
- The contextualizer remains parameter-free in PAS v1, so contextualizer-specific optimizer surfaces are intentionally unsupported.
- AMP is intentionally unsupported in the PAS runtime until a real train/eval AMP path is added; `training.amp` now fails loudly instead of acting like a no-op.
