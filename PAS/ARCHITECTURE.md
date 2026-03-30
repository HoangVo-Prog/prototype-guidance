# Architecture Notes

## End-to-End Flow

1. `train.py` loads YAML-backed runtime arguments from `utils/options.py` and `utils/config.py`.
2. `datasets/build.py` creates the train loader plus image/text retrieval loaders.
3. `model/build.py` constructs `PASModel` around the CLIP backbone from `model/clip_model.py`.
4. `solver/build.py` creates explicit optimizer param groups for the prototype bank, contextualizer, projectors, logit scale, and optional unfrozen backbones.
5. `processor/processor.py` runs training, logging, checkpointing, and periodic retrieval evaluation.
6. `utils/metrics.py` encodes image and text retrieval features through the model’s public retrieval interface and computes retrieval metrics.

## Main Components

### Dataset Layer

- `datasets/cuhkpedes.py`, `datasets/icfgpedes.py`, and `datasets/rstpreid.py` parse dataset annotations.
- `datasets/bases.py` defines `ImageDataset`, `TextDataset`, and `ImageTextDataset`.
- `datasets/build.py` applies CLIP-style transforms, identity/random sampling, and dataloader construction.

### Model Layer

- `model/clip_model.py` provides the CLIP image and text encoders plus intermediate-state extraction.
- `model/build.py` defines `PASModel`, the primary model wrapper.
- `model/prototype/` contains the prototype bank, contextualizer, router, aggregator, token scorer, token mask builder, token pooler, projector modules, and loss helpers.
- `model/prototype/head.py` composes the PAS branch into a reusable head shared by training and retrieval evaluation.

### Training And Evaluation

- `processor/processor.py` expects `forward(...)` to return `loss_total`, loss breakdown terms, and optional debug metrics.
- `utils/metric_logging.py` extracts scalar losses and diagnostics for TensorBoard and Weights & Biases.
- `utils/metrics.py` evaluates retrieval using `encode_image_for_retrieval(...)`, `encode_text_for_retrieval(...)`, and `compute_retrieval_similarity(...)`.

## Current Defaults

- The PAS method is the default path.
- Both CLIP backbones are frozen by default for v1 experiments.
- Image-conditioned token pooling is always enabled in the active model path.
- Prototype contextualization is enabled by default but can be disabled in ablations.

## Key Active Extension Points

- `model/prototype/head.py`: change prototype routing, contextualization, token scoring, or pooling logic.
- `solver/build.py`: adjust param-group learning rates and freeze policy behavior.
- `utils/metric_logging.py`: add new scalar diagnostics without touching the training loop.
- `configs/*.yaml`: define new experiments and ablations without code changes.

