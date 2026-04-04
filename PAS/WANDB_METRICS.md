# W&B Metrics Reference

This document describes the metrics that PAS sends to Weights & Biases during training and evaluation.

The logging path is implemented in:
- `utils/experiment.py`
- `utils/metric_logging.py`
- `processor/processor.py`
- `utils/metrics.py`
- `model/prototype/head.py`
- `model/prototype/losses.py`

## Logging Behavior

### Metric namespaces
- `train/*`: training metrics.
- `debug/*`: per-batch diagnostic metrics.
- `val/*`: retrieval metrics and proxy-disabled loss metrics for the evaluation split selected by `dataset.val_dataset`.
- `plots/*`: W&B custom comparison charts that overlay train and val series for the same metric inside one run.

### X-axes in W&B
- `train/*` uses `train/step` as the step axis.
- `debug/*` uses `train/step` as the step axis.
- `val/*` uses `val/epoch` as the step axis.
- `plots/*` are custom charts keyed by epoch history inside the tracker.

### When metrics are logged
- Training metrics are logged every `wandb_log_interval` steps.
- Training metrics are also logged once at the end of each epoch.
- Validation metrics are logged every evaluation epoch.
- `plots/*` comparison charts are refreshed once per evaluation epoch.

### Important runtime toggles
- `logging.use_wandb=true`: enables W&B.
- `logging.log_debug_metrics=true`: enables the `debug/*` metrics below.
- `val/*` loss monitoring disables proxy-classification terms because the selected eval split may contain unseen identities.
- `evaluation.retrieval_metrics`: controls which retrieval metrics appear under `val/pas/*`.

### Run files on W&B
- Each run uploads `configs.yaml` and `resolved_config.yaml` to W&B as a `run_config` artifact.
- If you launched with `--config_file`, that source config is also uploaded as `source_config.yaml`.

### Notes on interpretation
- `train/*` metrics are computed from the current batch at log time.
- Most `debug/*` metrics are computed from the current batch at log time.
- Rolling-window `debug/*` coverage metrics summarize recent training batches rather than only the current batch.
- Some losses can be intentionally zero if their corresponding loss branch is disabled.
- Prototype-bank initialization diagnostics are logged to the Python logger, not to W&B.
- `plots/*` overlay train and val in one chart, but the line style itself is controlled by the W&B UI rather than PAS runtime code.

## Always Logged Training Metrics

### Core schedule metrics
- `train/epoch`: current training epoch.
- `train/step`: global optimization step.
- `train/lr`: learning rate used for the logged step.

### Total and component losses
- `train/loss_total`: full optimized objective.
- `train/loss_proxy`: sum of all enabled proxy-classification losses.
- `train/loss_proxy_image`: proxy loss on the image embedding branch.
- `train/loss_proxy_text`: proxy loss on the surrogate text branch.
- `train/loss_proxy_text_exact`: proxy loss on the exact pooled text branch.
- `train/loss_ret_exact`: in-batch exact image-to-text retrieval cross-entropy in deployed scorer space.
- `train/loss_align`: cosine-alignment loss between image and surrogate text embeddings.
- `train/loss_diag`: diagonal fidelity loss between surrogate and exact text embeddings.
- `train/loss_support`: low-support routing penalty based on inverse participation ratio.
- `train/loss_diversity`: prototype diversity regularizer.
- `train/loss_balance`: routing-usage balance regularizer.

### Weighted objective terms
- `train/loss_proxy_weighted`: weighted proxy objective.
- `train/loss_ret_exact_weighted`: weighted exact retrieval objective.
- `train/loss_align_weighted`: weighted alignment objective.
- `train/loss_diag_weighted`: weighted diagonal fidelity objective.
- `train/loss_support_weighted`: weighted support penalty.
- `train/loss_diversity_weighted`: weighted diversity regularizer.
- `train/loss_balance_weighted`: weighted balance regularizer.

## Validation Metrics

These are logged during evaluation on the split selected by `dataset.val_dataset`.

- `val/epoch`: epoch at which validation was run.
- `val/loss_total`: proxy-disabled evaluation objective on the selected eval split.
- `val/loss_ret_exact`: exact retrieval loss on the selected eval split.
- `val/loss_align`: image/surrogate alignment loss on the selected eval split.
- `val/loss_diag`: surrogate/exact fidelity loss on the selected eval split.
- `val/top1`: alias for `R1`, used as the primary model-selection metric.
- `val/pas/R1`: rank-1 retrieval accuracy.
- `val/pas/R5`: rank-5 retrieval accuracy.
- `val/pas/R10`: rank-10 retrieval accuracy.
- `val/pas/mAP`: mean average precision for retrieval.

## Debug Metrics

These are logged only when `logging.log_debug_metrics=true`.

### Temperatures and retrieval scaling
- `debug/logit_scale`: multiplicative retrieval scale used for exact similarity scoring.
- `debug/proxy_temperature`: temperature used for proxy-classification logits.
- `debug/retrieval_temperature`: reciprocal of `logit_scale`, shown as the effective retrieval temperature.
- `debug/ret_exact_temperature`: temperature used by the exact retrieval CE; equals `debug/retrieval_temperature` when unset in config.

### Embedding and pooled-feature norms
- `debug/q_norm`: norm of the prototype summary vector `Q` used for token scoring.
- `debug/surrogate_t_pool_norm`: norm of the surrogate pooled text feature.
- `debug/exact_t_pool_norm`: norm of the exact pooled text feature.
- `debug/image_feature_norm`: norm of the image feature entering the image projector.
- `debug/image_embed_norm_raw`: mean norm of raw projected image embeddings before optional output normalization.
- `debug/image_embed_unit_norm`: mean norm of the final image embeddings after projector normalization.
- `debug/surrogate_text_embed_norm_raw`: mean norm of surrogate text projector outputs before normalization.
- `debug/surrogate_text_embed_unit_norm`: mean norm of final surrogate text embeddings after normalization.
- `debug/exact_text_embed_norm_raw`: mean norm of exact text embeddings before normalization.
- `debug/exact_text_embed_unit_norm`: mean norm of final exact text embeddings after normalization.

### Exact Retrieval Diagnostics
- `debug/image_surrogate_positive_cosine_mean`: mean positive image-vs-surrogate cosine on the current batch.
- `debug/image_surrogate_hardest_negative_cosine_mean`: mean hardest-negative image-vs-surrogate cosine on the current batch.
- `debug/image_surrogate_margin_mean`: mean positive-minus-hardest-negative surrogate margin.
- `debug/image_exact_positive_cosine_mean`: mean positive exact image-to-text cosine from the in-batch deployed scorer.
- `debug/image_exact_hardest_negative_cosine_mean`: mean hardest-negative exact cosine from the in-batch deployed scorer.
- `debug/image_exact_margin_mean`: mean positive-minus-hardest-negative exact margin.
- `debug/exact_pairwise_logit_mean`: mean of the in-batch exact pairwise logits used by `L_ret_exact`.
- `debug/exact_pairwise_logit_std`: standard deviation of the in-batch exact pairwise logits.
- `debug/exact_pairwise_logit_scale_or_norm`: effective logit scale used for `L_ret_exact`.
- `debug/exact_branch_grad_norm`: gradient norm observed at the exact pairwise retrieval logits used by `L_ret_exact`.
