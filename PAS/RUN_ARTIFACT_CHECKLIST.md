# RUN_ARTIFACT_CHECKLIST

Use this checklist for every full PAS run.

## Before Launch

- [ ] Config file recorded
- [ ] Dataset recorded
- [ ] Seed recorded
- [ ] Run ID recorded
- [ ] wandb mode decided
- [ ] Output root confirmed writable
- [ ] Dataset version / split confirmed

## After Training

- [ ] Run directory path recorded
- [ ] `resolved_config.yaml` exists
- [ ] `configs.yaml` exists
- [ ] `train_log.txt` exists
- [ ] `best.pth` exists
- [ ] Final epoch reached
- [ ] wandb run link recorded if enabled

## After Standalone Eval

- [ ] `test_log.txt` exists
- [ ] Checkpoint path used for eval recorded
- [ ] Final PAS metrics copied from logs or wandb:
- [ ] `val/pas/R1`
- [ ] `val/pas/R5`
- [ ] `val/pas/R10`
- [ ] `val/pas/mAP`
- [ ] `val/pas/mINP`
- [ ] `val/pas/rSum`

## Table Aggregation

- [ ] Result row copied into the correct table sheet
- [ ] Seed aggregate updated
- [ ] Mean and std updated if this is part of a 3-seed block
- [ ] Any failed seed noted explicitly
- [ ] Keep or archive `best.pth` according to retention policy
