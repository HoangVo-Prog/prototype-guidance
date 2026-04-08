# Prototype Integration Run Recipes

## Canonical Launcher
All stage/mode runs MUST use:
```powershell
python train.py --config configs/<stage_mode>.yaml
```

## Training Loop Surface
- `train.py` runs a full epoch loop with `torch.utils.data.DataLoader`.
- Checkpoints are saved under `runtime.checkpoint.checkpoint_dir` every `runtime.checkpoint.save_every_epochs`.
- Resume is config-driven via `runtime.checkpoint.resume_from`.
- Each config under `configs/` is standalone and includes:
  - `runtime.epochs`
  - `runtime.batch.*`
  - `runtime.data.*`
  - `runtime.optimizer.*`
  - `runtime.checkpoint.*`
  - `runtime.output.log_dir`

## Preconditions
- Install launcher dependencies: `pyyaml`, `torch`.
- Do not modify `prototype/adapter/WACV2026-Oral-ITSELF/**`.
- Use only the root launcher `train.py` with config files under `configs/`.

## Stage 0 ITSELF Baseline
- Config: `configs/stage0_itself.yaml`
- Purpose: host-only baseline in `train_mode=itself`.
- Required binding:
  - `train_mode: itself`
  - `training.stage: stage0`
  - `prototype.enabled: false`
  - `host.lambda_s` set (non-null)
- Command:
```powershell
python train.py --config configs/stage0_itself.yaml
```

## Stage 0 CLIP Baseline
- Config: `configs/stage0_clip.yaml`
- Purpose: host-only baseline in `train_mode=clip`.
- Required binding:
  - `train_mode: clip`
  - `training.stage: stage0`
  - `prototype.enabled: false`
  - `host.lambda_s: null`
- Command:
```powershell
python train.py --config configs/stage0_clip.yaml
```

## Stage 1 Prototype Stabilization
- Configs:
  - `configs/stage1_itself.yaml`
  - `configs/stage1_clip.yaml`
- Purpose: prototype training with host frozen.
- Required binding:
  - `training.stage: stage1`
  - `prototype.enabled: true`
  - `training.freeze.host: true`
  - `loss.host_enabled: false`
  - `loss.prototype_ret_enabled: true`
  - `loss.prototype_diag_enabled: true`
  - `loss.prototype_div_enabled: true`
- Commands:
```powershell
python train.py --config configs/stage1_itself.yaml
python train.py --config configs/stage1_clip.yaml
```

## Stage 2 Prototype-enabled Retraining
- Configs:
  - `configs/stage2_itself.yaml`
  - `configs/stage2_clip.yaml`
- Purpose: integrated retraining with host unfrozen and explicit CLIP init source.
- Required binding:
  - `training.stage: stage2`
  - `prototype.enabled: true`
  - `training.freeze.host: false`
  - `training.initialization.clip_backbone_source` non-null
- Commands:
```powershell
python train.py --config configs/stage2_itself.yaml
python train.py --config configs/stage2_clip.yaml
```

## Stage 3 Fusion Calibration
- Configs:
  - `configs/stage3_itself.yaml`
  - `configs/stage3_clip.yaml`
- Purpose: calibration-only mode with representation frozen.
- Required binding:
  - `training.stage: stage3`
  - `training.calibration_only: true`
  - `training.freeze.host: true`
  - `training.freeze.prototype: true`
  - representation-learning losses disabled
- Commands:
```powershell
python train.py --config configs/stage3_itself.yaml
python train.py --config configs/stage3_clip.yaml
```

## Exact Canonical Command Set (8)
```powershell
python train.py --config configs/stage0_itself.yaml
python train.py --config configs/stage0_clip.yaml
python train.py --config configs/stage1_itself.yaml
python train.py --config configs/stage1_clip.yaml
python train.py --config configs/stage2_itself.yaml
python train.py --config configs/stage2_clip.yaml
python train.py --config configs/stage3_itself.yaml
python train.py --config configs/stage3_clip.yaml
```

## Notes
- Fusion remains score-level only:
  - `s_total = s_host + lambda_f * s_proto`
- `s_host` remains mode-bound (`s_host^itself` vs `s_host^clip`).
- CLIP mode has no GRAB/local branch assumption.
- To resume any stage, set:
  - `runtime.checkpoint.resume_from: outputs/<stage_mode>/checkpoints/latest.pt`
