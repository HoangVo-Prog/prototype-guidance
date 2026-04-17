# Runtime Mode Config Recipes

## 1. Executive summary
- After the structural split, `model.runtime_mode` is now the primary semantic switch (`utils/config.py:21`, `model/runtime_modes.py:32-42`).
- Training entrypoint selection is mode-routed in `processor/processor.py` (`do_train` at `:692+`), not by legacy stage names.
- `training.freeze_schedule` is only applied in `joint_training`; in other modes it is explicitly ignored with a warning (`processor/processor.py:399-410`).
- `fused_external` is a first-class inference mode, but `train.py` with `runtime_mode=fused_external` is intentionally routed to prototype-external training path (`processor/processor.py:720-722`).
- `calibration_only` is real composer-only optimization, but only meaningful when `fusion.composer_calibration_enabled=true` (`model/plug_and_play.py:408`, `:451`).
- Legacy keys still parse (for compatibility), but should not define semantics; `training.runtime_mode` is an alias to canonical `model.runtime_mode` (`utils/config.py:229`).

## 2. Config surface audit

### 2.1 Canonical runtime-mode key and supported values
- Canonical YAML key: `model.runtime_mode` (`utils/config.py:21`).
- Supported values: `auto`, `host_only`, `prototype_only`, `fused_external`, `joint_training`, `calibration_only` (`model/runtime_modes.py:6-19`, `utils/config.py:444`).
- CLI equivalent: `--runtime_mode` (`utils/options.py:61`).

### 2.2 Runtime-mode resolution behavior
- `auto` resolves as:
  - no prototype branch -> `host_only`
  - training + prototype branch -> `joint_training`
  - eval + prototype branch -> `fused_external`
  - Source: `model/runtime_modes.py:32-42`.

### 2.3 Builder path by mode
- Host-only model can bypass split only when `runtime_mode=host_only` and prototype path is disabled (`model/build.py:42-43`).
- If prototype branch is active, builder routes to split runtime model (`model/build.py:45-48`).

### 2.4 Trainer path by mode
- `host_only` -> `train_host_core` (`processor/processor.py:706-718`, `:672-673`).
- `prototype_only` and `fused_external` -> `train_prototype_external` (`processor/processor.py:720-733`, `:677-678`).
- `calibration_only` -> `train_composer_calibration` (`processor/processor.py:734-746`, `:682-683`).
- otherwise -> `train_joint` (`processor/processor.py:748-759`, `:687-688`).

### 2.5 Freeze schedule semantics
- Joint mode only: schedule parsed/applied in `joint_training` (`processor/processor.py:399-424`).
- Non-joint modes: schedule ignored with warning (`processor/processor.py:405-410`).

### 2.6 New/important config keys
- `fusion.composer_calibration_enabled` (canonical) -> composer calibration gate (`utils/config.py:90`, `model/plug_and_play.py:408`, `:451`).
- `evaluation.retrieval_scorer`: `exact`/`approximate` (`utils/config.py:222`, `:471`, `utils/metrics.py:85-95`).
- `checkpointing.authority_validation.*` keys are validated and used in modular checkpoint manager (`utils/config.py:395`, `:775-787`; `utils/modular_checkpoint.py:332-338`, `:579+`).

### 2.7 Legacy keys and their status
- `training.runtime_mode` is accepted alias for compatibility (`utils/config.py:229`) but use `model.runtime_mode`.
- `training.stage` / `training_stage` still parse (`utils/config.py:461`, `:493`) but do not define runtime mode routing.
- `training.freeze_prototype_side` and other freeze booleans still parse (`utils/options.py:218-221`) but explicit runtime-mode trainability is applied first in non-joint modes (`processor/processor.py:288-329`, `:366-372`).

## 3. Legacy-to-new config mapping

| Old behavior / intention | New runtime mode | Old flags still relevant? | Old flags now ignored/secondary | Notes |
|---|---|---|---|---|
| CLIP host-only baseline | `host_only` + `use_prototype_branch=false` | `retrieval_mode=clip_bidirectional`, `token_policy=eos_only`, `retrieval_scorer=exact` still mandatory (`utils/config.py:896-897`) | Prototype-loss knobs should be disabled | Use host-only checkpoint authority. |
| Prototype branch training as add-on | `prototype_only` | prototype/objective knobs still relevant | `freeze_schedule` ignored | Trainer routes to prototype-external path. |
| Fusion-heavy evaluation | `fused_external` (usually `test.py`) | `fusion.lambda_*`, `fusion.eval_subsets`, `retrieval_scorer` relevant | Stage naming is secondary | Do not rely on `train.py` fused mode as unique training semantics. |
| Old schedule-driven staged training | `joint_training` + optional `training.freeze_schedule` | freeze schedule relevant | n/a | Only mode where schedule is semantic. |
| Composer calibration pass | `calibration_only` | `fusion.composer_calibration_enabled=true` required | host/prototype freeze flags secondary to mode routing | Use fusion-authority checkpoint saving. |

## 4. Summary matrix

| Mode | Intended use | Primary trainer path | Freeze schedule role | Composer role | Recommended first or later? |
|---|---|---|---|---|---|
| `host_only` | Host baseline training/eval | `train_host_core` | Ignored | Not trainable | First |
| `prototype_only` | External prototype optimization | `train_prototype_external` | Ignored | Frozen | Second |
| `fused_external` | Composed-system eval | Train routes to `train_prototype_external`; eval is native fused | Ignored in training route | Used for score fusion | Third (as eval) |
| `joint_training` | Coupled co-training | `train_joint` | Active | Jointly active | Later |
| `calibration_only` | Composer-only calibration | `train_composer_calibration` | Ignored | Only trainable component | Later (after host/prototype checkpoints) |

## 5. Mode-by-mode recipes

### 5.1 `host_only`

Config files:
- `configs/runtime_modes/host_only_smoke.yaml`
- `configs/runtime_modes/host_only_realistic.yaml`

Exact run commands:

```bash
python train.py --config_file configs/runtime_modes/host_only_smoke.yaml --name host_only_smoke --output_dir runs --use_wandb false
python test.py --config_file configs/runtime_modes/host_only_smoke.yaml --output_dir <run_dir_from_train> --checkpoint <run_dir_from_train>/best.pth
```

```bash
python train.py --config_file configs/runtime_modes/host_only_realistic.yaml --name host_only_realistic --output_dir runs --use_wandb false
python test.py --config_file configs/runtime_modes/host_only_realistic.yaml --output_dir <run_dir_from_train> --checkpoint <run_dir_from_train>/best.pth
```

Required flags/keys:
- `model.runtime_mode=host_only`
- `model.use_prototype_branch=false`
- `text_pooling.special_token_ids` must be explicitly set (required by model config validation).
- For `host.type=clip`: `text_pooling.token_policy=eos_only`, `objectives.objectives.retrieval_mode=clip_bidirectional`, `evaluation.retrieval_scorer=exact` (`utils/config.py:896-897`, `:987`).

Optional:
- optimizer scale and training length.

Ignored/secondary in this mode:
- `training.freeze_schedule` (ignored outside joint mode).

Dangerous/misleading:
- Enabling prototype losses while `use_prototype_branch=false` fails validation (`utils/config.py:899-914`).

---

### 5.2 `prototype_only`

Config files:
- `configs/runtime_modes/prototype_only_smoke.yaml`
- `configs/runtime_modes/prototype_only_realistic.yaml`

Exact run commands:

```bash
python train.py --config_file configs/runtime_modes/prototype_only_smoke.yaml --name prototype_only_smoke --output_dir runs --use_wandb false
python test.py --config_file configs/runtime_modes/prototype_only_smoke.yaml --output_dir <run_dir_from_train> --checkpoint <run_dir_from_train>/best.pth
```

```bash
python train.py --config_file configs/runtime_modes/prototype_only_realistic.yaml --name prototype_only_realistic --output_dir runs --use_wandb false
python test.py --config_file configs/runtime_modes/prototype_only_realistic.yaml --output_dir <run_dir_from_train> --checkpoint <run_dir_from_train>/best.pth
```

Required:
- `model.runtime_mode=prototype_only`
- `model.use_prototype_branch=true`, `model.use_prototype_bank=true`, `model.use_image_conditioned_pooling=true` (`utils/config.py:925-930`).
- `text_pooling.special_token_ids` must be explicitly set.
- `objectives.objectives.retrieval_mode=surrogate_i2t` for prototype path.

Optional:
- `evaluation.retrieval_scorer=approximate` (realistic recipe).

Ignored/secondary:
- `training.freeze_schedule` ignored in non-joint modes (`processor/processor.py:405-410`).

Dangerous/misleading:
- Setting `use_prototype_bank=true` with `use_image_conditioned_pooling=false` is invalid (`utils/config.py:925-930`).

---

### 5.3 `fused_external`

Config files:
- `configs/runtime_modes/fused_external_smoke.yaml`
- `configs/runtime_modes/fused_external_realistic.yaml`

Exact run commands:

```bash
# Eval using a full checkpoint (best.pth) from a previous prototype/joint run.
python test.py --config_file configs/runtime_modes/fused_external_smoke.yaml --output_dir <run_dir> --checkpoint <run_dir>/best.pth
```

```bash
# Eval using modular mixed-source loading (paths must be edited in config first).
python test.py --config_file configs/runtime_modes/fused_external_realistic.yaml --output_dir runs/fused_external_eval
```

Required:
- `model.runtime_mode=fused_external`
- prototype branch enabled
- `text_pooling.special_token_ids` must be explicitly set.
- meaningful `fusion.lambda_host` / `fusion.lambda_prototype`.

Optional:
- `fusion.eval_subsets` for sweep rows.
- `evaluation.retrieval_scorer=approximate` when prototype bank is present (`utils/metrics.py:91-95`).

Ignored/secondary:
- If you run `train.py` with `fused_external`, trainer dispatch still uses prototype-external training path (`processor/processor.py:720-722`).

Dangerous/misleading:
- Treating `fused_external` as a distinct training objective in `train.py` is incorrect in current router behavior.

---

### 5.4 `joint_training`

Config files:
- `configs/runtime_modes/joint_training_smoke.yaml`
- `configs/runtime_modes/joint_training_realistic.yaml`

Exact run commands:

```bash
python train.py --config_file configs/runtime_modes/joint_training_smoke.yaml --name joint_training_smoke --output_dir runs --use_wandb false
python test.py --config_file configs/runtime_modes/joint_training_smoke.yaml --output_dir <run_dir_from_train> --checkpoint <run_dir_from_train>/best.pth
```

```bash
python train.py --config_file configs/runtime_modes/joint_training_realistic.yaml --name joint_training_realistic --output_dir runs --use_wandb false
python test.py --config_file configs/runtime_modes/joint_training_realistic.yaml --output_dir <run_dir_from_train> --checkpoint <run_dir_from_train>/best.pth
```

Required:
- `model.runtime_mode=joint_training`
- prototype branch enabled.
- `text_pooling.special_token_ids` must be explicitly set.

Optional:
- `training.freeze_schedule` (only mode where it is active).

Ignored/secondary:
- None; this is the only schedule-active training mode.

Dangerous/misleading:
- Assuming stage labels (`training.stage`) define runtime semantics; mode router controls semantics.

---

### 5.5 `calibration_only`

Config files:
- `configs/runtime_modes/calibration_only_smoke.yaml`
- `configs/runtime_modes/calibration_only_realistic.yaml`

Exact run commands:

```bash
# Smoke: composer-only calibration pass from current initialized model state.
python train.py --config_file configs/runtime_modes/calibration_only_smoke.yaml --name calibration_only_smoke --output_dir runs --use_wandb false
python test.py --config_file configs/runtime_modes/calibration_only_smoke.yaml --output_dir <run_dir_from_train> --checkpoint <run_dir_from_train>/best.pth
```

```bash
# Realistic: edit checkpointing.load.sources paths first (host/prototype artifacts),
# then calibrate composer and evaluate.
python train.py --config_file configs/runtime_modes/calibration_only_realistic.yaml --name calibration_only_realistic --output_dir runs --use_wandb false
python test.py --config_file configs/runtime_modes/calibration_only_realistic.yaml --output_dir <run_dir_from_train> --checkpoint <run_dir_from_train>/best.pth
```

Required:
- `model.runtime_mode=calibration_only`
- `text_pooling.special_token_ids` must be explicitly set.
- `fusion.composer_calibration_enabled=true` (`model/plug_and_play.py:408`, `:451`).
- `checkpointing.groups.fusion.enabled=true` (recommended authority target for this mode).

Optional:
- Modular load sources for host/prototype components in realistic flow.

Ignored/secondary:
- `training.freeze_schedule` ignored.

Dangerous/misleading:
- `composer_calibration_enabled=false` makes calibration params unused in fusion path.

## 6. Full-flag explanation by config

| Config | `runtime_mode` | Host/Prototype/Fusion switches | Checkpointing authority intent | Eval scorer | Freeze schedule role |
|---|---|---|---|---|---|
| `host_only_smoke.yaml` | `host_only` | prototype off, fusion off | host-only group enabled | exact | ignored |
| `host_only_realistic.yaml` | `host_only` | prototype off, fusion off | host-only group enabled | exact | ignored |
| `prototype_only_smoke.yaml` | `prototype_only` | prototype on, fusion on (proto-weighted) | prototype groups enabled | exact | ignored |
| `prototype_only_realistic.yaml` | `prototype_only` | prototype on, fusion on | prototype groups enabled | approximate | ignored |
| `fused_external_smoke.yaml` | `fused_external` | prototype on, fusion on (fixed blend + sweep) | fusion group focus | exact | ignored |
| `fused_external_realistic.yaml` | `fused_external` | prototype on, fusion on, modular load enabled | all groups load-compatible | approximate | ignored |
| `joint_training_smoke.yaml` | `joint_training` | all active | all groups enabled | exact | active if provided |
| `joint_training_realistic.yaml` | `joint_training` | all active + explicit schedule | all groups enabled | exact | active (configured) |
| `calibration_only_smoke.yaml` | `calibration_only` | prototype on, composer calibration on | fusion-only save authority | exact | ignored |
| `calibration_only_realistic.yaml` | `calibration_only` | prototype on, composer calibration on, modular load | fusion-only save authority, host/proto load required | exact | ignored |

## 7. Recommended first runs
1. `configs/runtime_modes/host_only_smoke.yaml`
Reason: validates baseline host path and host-authority checkpoint behavior with minimal risk.
2. `configs/runtime_modes/prototype_only_smoke.yaml`
Reason: validates external prototype training path and prototype-authority artifact flow.
3. `configs/runtime_modes/fused_external_smoke.yaml` (via `test.py`)
Reason: validates composed scoring/eval path without introducing joint training confounds.

## 8. Common mistakes
- Using `training.freeze_schedule` expecting it to define semantics in `host_only`, `prototype_only`, `fused_external`, or `calibration_only`.
- Expecting `train.py` with `runtime_mode=fused_external` to run a unique fused trainer path (current routing maps to prototype-external training path).
- Using `host.type=clip` + `use_prototype_branch=false` but forgetting `retrieval_mode=clip_bidirectional`, `token_policy=eos_only`, and `retrieval_scorer=exact`.
- Setting `evaluation.retrieval_scorer=approximate` without active prototype bank (evaluator falls back to exact).
- Running calibration mode with `fusion.composer_calibration_enabled=false`.
- Leaving placeholder modular checkpoint paths in realistic fused/calibration configs.

## 9. Notes on legacy flags
- Still active compatibility alias: `training.runtime_mode` -> `runtime_mode` (`utils/config.py:229`).
- Secondary legacy surface: `training.stage` still parses but is not the mode router.
- Freeze booleans (`freeze_*`) still parse, but explicit runtime mode trainability is enforced at trainer start in non-joint modes (`processor/processor.py:288-329`, `:366-372`).
