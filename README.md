# PAS Retrieval

This repository is the active research codebase for PAS image-text retrieval.

PAS now supports two explicit host families:
- `host.type=clip`: the preserved CLIP-centered host path, with or without the prototype branch
- `host.type=itself`: a self-contained ITSELF-style host path re-implemented inside PAS, with or without the prototype branch

The integrated retrieval design is host-plus-prototype:
- the selected host path remains independently trainable, scorable, and reportable
- the prototype branch is optional and additive
- fused evaluation uses `host + lambda_f * prototype`
- prototype losses remain separated from host losses so host-only and host-plus-prototype ablations stay clean

## Main entrypoints

- `train.py`: PAS training entrypoint
- `test.py`: retrieval evaluation entrypoint
- `configs/baselines/vanilla_clip.yaml`: CLIP host-only baseline
- `configs/baselines/itself_host_only.yaml`: ITSELF host-only baseline
- `configs/stage0/stage0_host_only_reproduction.yaml`: CLIP Stage 0 host-only reproduction
- `configs/stage0/stage0_itself_host_only.yaml`: ITSELF Stage 0 host-only reproduction
- `configs/stage1/stage1_retrieval_from_start.yaml`: CLIP host + prototype Stage 1
- `configs/stage1/stage1_itself_host_plus_prototype.yaml`: ITSELF host + prototype Stage 1
- `configs/stage2/stage2_backbone_optimization.yaml`: CLIP host + prototype Stage 2
- `configs/stage2/stage2_itself_host_plus_prototype.yaml`: ITSELF host + prototype Stage 2
- `configs/stage3/stage3_fusion_calibration.yaml`: CLIP host + prototype Stage 3
- `configs/stage3/stage3_itself_fusion_calibration.yaml`: ITSELF host + prototype Stage 3
- `scripts/phase_e_smoke.py`: synthetic end-to-end smoke harness

## Repository layout

- `datasets/`: dataset parsing, tokenization, transforms, samplers, dataloader builder
- `model/`: CLIP wrapper plus PAS retrieval model and modules
- `processor/`: train and inference loops
- `solver/`: optimizer and LR scheduler construction
- `utils/`: config IO, logging, metrics, checkpointing, distributed helpers
- `tests/`: module and integration tests

## Wandb setup

Create a local `.env` file in the repo root before any tracked run:

```bash
WANDB_API_KEY=your_wandb_api_key_here
```

`train.py` and `test.py` load `.env` automatically at startup. If `WANDB_API_KEY` is still unset, they also try to read a Kaggle Secret named `WANDB_API_KEY`, so local runs and Kaggle notebook runs both work without an interactive `wandb login` step.

## Typical commands

```bash
python train.py --config_file configs/baselines/vanilla_clip.yaml
python train.py --config_file configs/baselines/itself_host_only.yaml
python train.py --config_file configs/stage1/stage1_itself_host_plus_prototype.yaml
python train.py --config_file configs/stage2/stage2_itself_host_plus_prototype.yaml
python test.py --config_file configs/stage3/stage3_itself_fusion_calibration.yaml --output_dir <run_dir>
```

## Notes

- `README.upstream.md` is kept as a historical snapshot of the inherited upstream project.
- Phase reports document the staged integration history and cleanup decisions.








