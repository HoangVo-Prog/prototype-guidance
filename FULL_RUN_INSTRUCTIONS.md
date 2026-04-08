# FULL_RUN_INSTRUCTIONS

## 1. Scope

This document defines how to execute full PAS experiment runs for paper-ready evidence collection.

A full run means:
- standard dataset split
- full training schedule from the selected config
- automatic validation during training
- standalone evaluation with `test.py`
- checkpoint retention and artifact recording
- enough recorded outputs that a table row can be filled without guesswork

Out of scope for this document:
- debug runs
- smoke runs
- tiny-overfit runs
- module-level tests
- implementation changes

## 2. Preconditions

Before launching any full run, confirm all of the following:
- dependencies are installed in the training environment, including `torch`, `yaml`, `prettytable`, TensorBoard support, and optional `wandb`
- dataset root exists at the `--root_dir` location
- dataset subdirectories expected by the repo are available:
  - `data/CUHK-PEDES`
  - `data/ICFG-PEDES`
  - `data/RSTPReid`
- the intended config file exists under `configs/`
- the output root exists or is writable; default root is `runs`
- if wandb is required for the run, create a repo-local `.env` file with `WANDB_API_KEY=...` before launch and run with `--use_wandb true`

## 3. Standard Run Naming Convention

Use this base run-name format for all full runs:

```text
pas_<group>_<dataset_or_source_to_target>_seed<seed>
```

Examples:
- `pas_main_cuhk_seed1`
- `pas_main_icfg_seed2`
- `pas_main_rstp_seed3`
- `pas_ablate_no_context_cuhk_seed1`
- `pas_ablate_no_diversity_cuhk_seed2`
- `pas_xdomain_cuhk_to_rstp_seed3`

Dataset short names used in run names:
- `cuhk` for `CUHK-PEDES`
- `icfg` for `ICFG-PEDES`
- `rstp` for `RSTPReid`

Important path note:
- `train.py` prepends a timestamp automatically
- the actual run directory becomes:

```text
<output_dir>/<dataset>/<timestamp>_<name>
```

Example:

```text
runs/CUHK-PEDES/20260330_153000_pas_main_cuhk_seed1
```

## 4. Full Training Command Pattern

Canonical training command:

```bash
python train.py \
  --config_file <config_path> \
  --dataset_name <dataset_name> \
  --seed <seed> \
  --name <run_id> \
  --output_dir runs \
  --use_wandb true \
  --wandb_project PAS \
  --wandb_run_name <run_id>
```

Notes:
- `--name` controls the stable run ID suffix inside the timestamped output folder.
- `--wandb_run_name` should match `--name` exactly for clean bookkeeping.
- The canonical wandb setup is a repo-local `.env` containing `WANDB_API_KEY=...`.
- If you intentionally do not use wandb, set `--use_wandb false`, but that run should be treated as missing one of the recommended paper-traceability artifacts.

## 5. Full Evaluation Command Pattern

After training completes, run a standalone evaluation against the saved best checkpoint:

```bash
python test.py \
  --config_file <config_path> \
  --output_dir <run_dir> \
  --checkpoint <run_dir>/best.pth
```

Why this is required even though training already evaluates:
- the standalone eval creates a dedicated `test_log.txt`
- it makes the final reported metrics easier to archive and audit
- it avoids relying only on training-time log parsing

## 5.1 Staged Host Checkpoint Policy

For both supported host families, Stage 0 is the checkpoint-creation stage and later stages must load that checkpoint chain explicitly.

CLIP host path:
- Stage 0 creates the preserved CLIP host-only checkpoint.
- Stage 1 must load the Stage 0 CLIP checkpoint through `training.finetune`.
- Stage 2 must load the Stage 1 CLIP+prototype checkpoint through `training.finetune`.
- Stage 3 must load the Stage 2 CLIP+prototype checkpoint through `training.finetune` and `evaluation.checkpoint_path`.

ITSELF host path:
- Stage 0 creates the PAS ITSELF host-only checkpoint.
- Stage 0 is also the reproduction/verification run that should be checked against the original ITSELF codebase metrics before prototype training starts.
- Stage 1 must load the Stage 0 ITSELF checkpoint through `training.finetune`.
- Stage 2 must load the Stage 1 ITSELF+prototype checkpoint through `training.finetune`.
- Stage 3 must load the Stage 2 ITSELF+prototype checkpoint through `training.finetune` and `evaluation.checkpoint_path`.

Practical rule:
- do not start Stage 1 for a host family until the corresponding Stage 0 checkpoint exists and its host-only metrics have been verified
- treat Stage 0 as baseline reproduction, not as an optional warmup

## 5.2 ITSELF Staged Run Sequence

Dataset-specific Stage 0 reproduction reference:
- use `configs/itself_dataset_reference.yaml` to look up the original ITSELF baseline settings, the dataset-specific `host.itself_score_weight_global` value, and the intended checkpoint chain for later stages
- for `RSTPReid`, the current verified reference row is `global+grab(0.32)-t2i`, so Stage 0 parity checks should use `host.itself_score_weight_global: 0.32`

Recommended ITSELF execution order:

```bash
python train.py --config_file configs/stage0/stage0_itself_host_only.yaml
python train.py --config_file configs/stage1/stage1_itself_host_plus_prototype.yaml
python train.py --config_file configs/stage2/stage2_itself_host_plus_prototype.yaml
python test.py --config_file configs/stage3/stage3_itself_fusion_calibration.yaml --output_dir <run_dir>
```

Before launching Stage 1, replace:
- `training.finetune` in `configs/stage1/stage1_itself_host_plus_prototype.yaml` with the real Stage 0 `best.pth`

Before launching Stage 2, replace:
- `training.finetune` in `configs/stage2/stage2_itself_host_plus_prototype.yaml` with the real Stage 1 `best.pth`

Before launching Stage 3, replace:
- `training.finetune` in `configs/stage3/stage3_itself_fusion_calibration.yaml` with the real Stage 2 `best.pth`
- `evaluation.checkpoint_path` in `configs/stage3/stage3_itself_fusion_calibration.yaml` with the same Stage 2 `best.pth`

## 6. Per-Table Full Run Instructions

### 6.1 Table 1 and PAS Rows Of The Main Comparison Table

Use config:
- `configs/train_pas_v1.yaml`

Required datasets:
- `CUHK-PEDES`
- `ICFG-PEDES`
- `RSTPReid`

Required seeds:
- `1`
- `2`
- `3`

Train command template:

```bash
python train.py \
  --config_file configs/train_pas_v1.yaml \
  --dataset_name <CUHK-PEDES|ICFG-PEDES|RSTPReid> \
  --seed <1|2|3> \
  --name <pas_main_<dataset_short>_seedX> \
  --output_dir runs \
  --use_wandb true \
  --wandb_project PAS \
  --wandb_run_name <pas_main_<dataset_short>_seedX>
```

Eval command template:

```bash
python test.py \
  --config_file configs/train_pas_v1.yaml \
  --output_dir <run_dir> \
  --checkpoint <run_dir>/best.pth
```

Expected outputs:
- `<run_dir>/resolved_config.yaml`
- `<run_dir>/configs.yaml`
- `<run_dir>/train_log.txt`
- `<run_dir>/test_log.txt`
- `<run_dir>/best.pth`
- TensorBoard event files in `<run_dir>`
- wandb run page if enabled

Completion rule:
- training reaches the configured final epoch
- `best.pth` exists
- standalone eval finishes
- final PAS metrics are copied into the result sheet

Report destination:
- main benchmark table sheet
- PAS row of the main comparison table

### 6.2 Table 3 Essential PAS Ablations

Primary benchmark:
- `CUHK-PEDES`

Required configs:
- `configs/train_pas_v1.yaml`
- `configs/ablation_pas_no_context.yaml`
- `configs/ablation_pas_no_diversity.yaml`

Required seeds:
- `1`
- `2`
- `3`

Run IDs:
- `pas_ablation_ref_cuhk_seed1`
- `pas_ablation_ref_cuhk_seed2`
- `pas_ablation_ref_cuhk_seed3`
- `pas_ablate_no_context_cuhk_seed1`
- `pas_ablate_no_context_cuhk_seed2`
- `pas_ablate_no_context_cuhk_seed3`
- `pas_ablate_no_diversity_cuhk_seed1`
- `pas_ablate_no_diversity_cuhk_seed2`
- `pas_ablate_no_diversity_cuhk_seed3`

Train command templates:

```bash
python train.py \
  --config_file configs/train_pas_v1.yaml \
  --dataset_name CUHK-PEDES \
  --seed <1|2|3> \
  --name <pas_ablation_ref_cuhk_seedX> \
  --output_dir runs \
  --use_wandb true \
  --wandb_project PAS \
  --wandb_run_name <pas_ablation_ref_cuhk_seedX>
```

```bash
python train.py \
  --config_file configs/ablation_pas_no_context.yaml \
  --dataset_name CUHK-PEDES \
  --seed <1|2|3> \
  --name <pas_ablate_no_context_cuhk_seedX> \
  --output_dir runs \
  --use_wandb true \
  --wandb_project PAS \
  --wandb_run_name <pas_ablate_no_context_cuhk_seedX>
```

```bash
python train.py \
  --config_file configs/ablation_pas_no_diversity.yaml \
  --dataset_name CUHK-PEDES \
  --seed <1|2|3> \
  --name <pas_ablate_no_diversity_cuhk_seedX> \
  --output_dir runs \
  --use_wandb true \
  --wandb_project PAS \
  --wandb_run_name <pas_ablate_no_diversity_cuhk_seedX>
```

Eval command pattern:

```bash
python test.py \
  --config_file <same_config_used_for_training> \
  --output_dir <run_dir> \
  --checkpoint <run_dir>/best.pth
```

Expected outputs:
- same artifact set as the main benchmark runs

Completion rule:
- same as the main benchmark runs

Report destination:
- essential ablation table sheet

### 6.3 Table 4 Cross-Dataset Generalization / Robustness

This block is eval-only and depends on completed Table 1 source checkpoints.

Source checkpoints:
- all `best.pth` files from the main PAS runs

Target datasets per source:
- `CUHK-PEDES` source -> `ICFG-PEDES`, `RSTPReid`
- `ICFG-PEDES` source -> `CUHK-PEDES`, `RSTPReid`
- `RSTPReid` source -> `CUHK-PEDES`, `ICFG-PEDES`

Eval command template:

```bash
python test.py \
  --config_file configs/train_pas_v1.yaml \
  --output_dir <run_dir> \
  --checkpoint <run_dir>/best.pth \
  --cross_domain_generalization true \
  --target_domain <target_dataset>
```

Recommended eval naming convention in the tracker sheet:
- `pas_xdomain_<source_short>_to_<target_short>_seedX`

Expected outputs:
- `test_log.txt` showing cross-domain metrics
- recorded source run ID, source dataset, target dataset, seed, and checkpoint path

Completion rule:
- eval finishes without error
- the source checkpoint and target dataset are recorded
- PAS metrics are copied into the cross-domain results sheet

Report destination:
- robustness / cross-domain table sheet

## 7. Seed Policy

Standard policy for full paper-facing PAS runs:
- use exactly `3` seeds for all main benchmark runs
- use exactly `3` seeds for the shipped essential ablations
- use the same `3` seeds for cross-domain evals by reusing the corresponding source checkpoints

Single-seed is acceptable only when:
- the run is exploratory and not destined for a paper table
- or the table block is explicitly marked as provisional

Aggregation rule:
- report `mean ± std` over the `3` seeds for every multi-seed PAS result block
- never report the single best seed as the final paper number for a multi-seed group

## 8. Checkpoint Policy

Checkpoint selection rule:
- the repository promotes `best.pth` using `val/top1`, which equals `val/pas/R1`

Keep at minimum:
- `best.pth`
- `resolved_config.yaml`
- `configs.yaml`
- `train_log.txt`
- `test_log.txt`

Retention policy:
- `epoch_*.pth` checkpoints may be deleted after verification if storage is tight
- keep `best.pth` until the paper tables are finalized
- if a run is part of the final table set, do not delete its `best.pth` or logs

## 9. Output Artifact Policy

Every full PAS run must preserve:
- config snapshot:
  - `resolved_config.yaml`
  - `configs.yaml`
- logs:
  - `train_log.txt`
  - `test_log.txt`
- checkpoint:
  - `best.pth`
- tracking:
  - wandb run URL if enabled
  - run directory path
  - seed
  - dataset
  - config file name

Current repo limitation:
- the repo does not automatically export eval metrics to JSON or CSV
- the repo does not automatically write a table-ready summary row
- therefore the researcher must manually copy the PAS metric row from `test_log.txt` or wandb into the aggregation sheet

## 10. Result Aggregation Notes

For PAS paper tables:
- use `best.pth` selected by `val/pas/R1`
- run standalone eval on `best.pth`
- aggregate `R1`, `R5`, `R10`, `mAP`, `mINP`, and `rSum` over seeds with `mean ± std`

Handling failed seeds:
- if a seed crashes before producing `best.pth`, rerun the same seed from scratch
- if a seed finishes training but standalone eval fails, rerun eval first; only retrain if the checkpoint is invalid
- if one seed remains unusable after one clean retry, mark the table block as incomplete rather than mixing two-seed and three-seed aggregates silently

## 11. Failure / Restart Policy

A run counts as failed if any of the following is true:
- training exits before the configured final epoch
- `best.pth` was never written
- standalone eval does not finish
- the final metric row cannot be recovered from logs or wandb
- the saved config snapshot does not match the intended run

Restart rules:
- restart from scratch if the run directory is incomplete or corrupted
- only use `--resume` when you are intentionally continuing the exact same run after an interruption and the checkpoint is valid
- if a run was launched with the wrong dataset, seed, or config, discard it from the paper table set and relaunch with the correct metadata

Partial-completion marking:
- record partial runs as `FAILED_TRAIN`, `FAILED_EVAL`, or `FAILED_ARTIFACTS`
- do not treat them as table-complete

## 12. Completion Checklist

A full PAS run is complete only if all of the following are true:
- training finished
- standalone evaluation finished
- `best.pth` exists
- `resolved_config.yaml` and `configs.yaml` exist
- `train_log.txt` and `test_log.txt` exist
- the wandb run is recorded if wandb was required
- final PAS metrics are copied into the result aggregation sheet
- the run can be traced back to a dataset, config, seed, and checkpoint without manual guesswork

## 13. Known Full-Run Blockers

The following full-run groups are mentioned in the planning documents but are not runnable from the current repo without additional config or code work:
- no-bank baseline full runs
- EOS-only pooled baseline full runs
- mean-pooling baseline full runs
- prototype-count sensitivity full runs (`N=16`, `N=64`)
- temperature sensitivity full runs
- automatic eval JSON/CSV export
- automatic table-row export

These should be treated as planned but blocked, not silently approximated.


