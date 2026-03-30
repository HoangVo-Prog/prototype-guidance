# PATCH_SUMMARY

## Modified files
| File | What changed | Affects |
|---|---|---|
| `PAS/utils/options.py` | Aligned parser defaults to canonical v1, added canonical CLI aliases, added per-group weight-decay options | default config |
| `PAS/utils/config.py` | Added config mappings and aliases for the repaired optimizer/model fields | default config |
| `PAS/solver/build.py` | Added per-group weight-decay handling | default config |
| `PAS/model/prototype/build.py` | Fixed prototype-head defaults (`projection_dim=256`, `normalized_random`, canonical aliases) | logic, default config |
| `PAS/model/prototype/prototype_bank.py` | Added canonical init aliases and external sampled/k-means init handling | default config |
| `PAS/model/prototype/router.py` | Exposed raw routing similarity and explicit temperature-scaled logits | normalization, logging |
| `PAS/model/prototype/token_scorer.py` | Exposed raw token similarity and scaled scores explicitly | normalization, logging |
| `PAS/model/prototype/token_mask.py` | Split padding-only vs keep-policy masks and tightened token-policy validation | masking |
| `PAS/model/prototype/token_pooler.py` | Implemented explicit masked-softmax surface with `beta_logits_masked`, zero beta on invalid tokens, and `compute_weights()/pool()` helpers | masking |
| `PAS/model/prototype/projector.py` | Exposed raw and normalized projector outputs separately | normalization, logging |
| `PAS/model/prototype/losses.py` | Repaired diversity formulation, returned raw sub-losses, and exposed weighted totals separately | loss |
| `PAS/model/prototype/head.py` | Added canonical tensor aliases, always-on scalar diagnostics, explicit masks/logits, raw projector outputs, and richer debug payloads | logic, masking, normalization, logging |
| `PAS/model/build.py` | Enforced hidden-state path consistency, validated legacy token flags, exposed repaired debug outputs, and limited fp16 conversion to the backbone | logic, default config, logging |
| `PAS/utils/metric_logging.py` | Added mappings for new diagnostics and weighted loss reporting | logging |
| `PAS/configs/default.yaml` | Repaired default v1 settings | default config |
| `PAS/configs/train_pas_v1.yaml` | Repaired default train recipe | default config |
| `PAS/configs/debug_pas_v1.yaml` | Kept debug-only training knobs while restoring canonical method defaults | default config |
| `PAS/configs/ablation_pas_no_context.yaml` | Isolated the no-context ablation while restoring other defaults | default config |
| `PAS/configs/ablation_pas_no_diversity.yaml` | Isolated the no-diversity ablation while restoring other defaults | default config |
| `PAS/configs/kaggle_pas_quicktrain.yaml` | Kept quicktrain training knobs while restoring canonical method defaults | default config |
| `PAS/tests/test_prototype_modules.py` | Added coverage for raw-vs-weighted losses, explicit masks, and repaired outputs | tests |
| `PAS/tests/test_phase_e_integration.py` | Added coverage for lightweight diagnostics, full debug tensors, optimizer groups, and backbone-only fp16 conversion | tests |
| `PAS/scripts/phase_e_smoke.py` | Added alpha/beta, mask, and prototype-gradient smoke checks | tests, logging |

## Remaining optional ablations not touched
- Dot-product routing and token scoring remain optional ablations only.
- Contextualization-off and overwrite-mode branches remain available as non-default ablations.
- Prototype-count sweeps remain optional (`16`, `32`, `64`).
- Balancing loss remains implemented as off-by-default.
- Backbone unfreezing remains optional and is still off by default.
- Pooling-policy ablations such as `content_plus_special` and `eos_only` remain available but non-default.
