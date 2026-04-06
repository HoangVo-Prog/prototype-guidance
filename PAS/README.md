# PAS Retrieval

This repository is the active research codebase for the PAS image-text retrieval model built on top of an inherited CLIP-based retrieval project.

The primary training path is now an integrated PAS host-plus-prototype method:
- preserved CLIP host retrieval path with its own trainable projectors and standalone host score
- optional prototype enhancement branch with routing, basis-bank construction, and image-conditioned surrogate text
- residual score fusion `host + lambda_f * prototype` at evaluation time
- host retrieval loss plus prototype row-wise retrieval, diagonal fidelity, and optional prototype regularization during training

## Main entrypoints

- `train.py`: PAS training entrypoint
- `test.py`: retrieval evaluation entrypoint
- `configs/train_pas_v1.yaml`: main experiment config
- `configs/debug_pas_v1.yaml`: short debug config
- `configs/ablation_pas_no_context.yaml`: contextualization ablation
- `configs/ablation_pas_no_diversity.yaml`: diversity-loss ablation
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
python train.py --config_file configs/train_pas_v1.yaml
python train.py --config_file configs/debug_pas_v1.yaml
python train.py --config_file configs/kaggle_pas_quicktrain.yaml
python test.py --config_file configs/train_pas_v1.yaml --output_dir <run_dir>
```

## Notes

- `README.upstream.md` is kept as a historical snapshot of the inherited upstream project.
- Phase reports document the staged integration history and cleanup decisions.








