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
- `val/*`: validation and retrieval metrics.

### X-axes in W&B
- `train/*` uses `train/step` as the step axis.
- `debug/*` uses `train/step` as the step axis.
- `val/*` uses `val/epoch` as the step axis.

### When metrics are logged
- Training metrics are logged every `wandb_log_interval` steps.
- Training metrics are also logged once at the end of each epoch.
- Validation metrics are logged every evaluation epoch.

### Important runtime toggles
- `logging.use_wandb=true`: enables W&B.
- `logging.log_debug_metrics=true`: enables the `debug/*` metrics below.
- `evaluation.retrieval_metrics`: controls which retrieval metrics appear under `val/pas/*`.

### Notes on interpretation
- `train/*` metrics are computed from the current batch at log time.
- Most `debug/*` metrics are computed from the current batch at log time.
- Rolling-window `debug/*` coverage metrics summarize recent training batches rather than only the current batch.
- Some losses can be intentionally zero if their corresponding loss branch is disabled.
- Prototype-bank initialization diagnostics are logged to the Python logger, not to W&B.

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
- `train/loss_proxy_weighted`: `lambda_proxy * loss_proxy`.
- `train/loss_ret_exact_weighted`: `lambda_ret_exact * loss_ret_exact`.
- `train/loss_align_weighted`: `lambda_align * loss_align`.
- `train/loss_diag_weighted`: `lambda_diag * loss_diag`.
- `train/loss_support_weighted`: `lambda_support * loss_support`.
- `train/loss_diversity_weighted`: `lambda_div * loss_diversity`.
- `train/loss_balance_weighted`: `lambda_bal * loss_balance`.

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
- `debug/image_embed_norm_std`: standard deviation of raw image embedding norms.
- `debug/image_embed_norm_min`: minimum raw image embedding norm in the logged batch.
- `debug/image_embed_norm_max`: maximum raw image embedding norm in the logged batch.
- `debug/surrogate_text_embed_norm_raw`: mean norm of surrogate text projector outputs before normalization.
- `debug/surrogate_text_embed_unit_norm`: mean norm of final surrogate text embeddings after normalization.
- `debug/exact_text_embed_norm_raw`: mean norm of exact text embeddings before normalization.
- `debug/exact_text_embed_unit_norm`: mean norm of final exact text embeddings after normalization.

### Routing and prototype-usage behavior
- `debug/prototype_usage_entropy`: entropy of average prototype usage over the batch. Higher means usage is more evenly spread.
- `debug/prototype_usage_max`: largest mean routing weight assigned to any prototype.
- `debug/prototype_dead_count`: number of prototypes whose mean usage falls below `prototype.dead_prototype_threshold`.
- `debug/routing_entropy`: mean entropy of per-sample routing distributions.
- `debug/routing_max_prob`: mean top routing probability across samples.
- `debug/routing_top1_usage_entropy`: entropy of the histogram of top-1 routed prototypes.
- `debug/routing_top1_usage_max`: largest top-1 assignment fraction for any prototype.
- `debug/routing_top1_dead_count`: number of prototypes that were never selected as top-1 in the batch.
- `debug/prototype_assignment_entropy`: mean entropy of routing assignments, as reported by the router.
- `debug/routing_effective_support`: exponentiated routing entropy, interpretable as the effective number of active prototypes per sample.
- `debug/routing_effective_support_ipr`: inverse-participation-ratio effective support, matching the `L_support` definition.
- `debug/routing_effective_support_ipr_p10`: 10th percentile of inverse-participation-ratio support across the batch.
- `debug/routing_effective_support_ipr_p50`: median inverse-participation-ratio support across the batch.
- `debug/routing_effective_support_ipr_p90`: 90th percentile of inverse-participation-ratio support across the batch.
- `debug/routing_support_below_2_frac`: fraction of samples whose IPR support is below 2.
- `debug/routing_support_below_3_frac`: fraction of samples whose IPR support is below 3.
- `debug/routing_support_below_min_frac`: fraction of samples whose IPR support is below the configured `support_min`.
- `debug/routing_top1_minus_top2`: mean gap between the largest and second-largest routing weights.
- `debug/routing_top2_mass`: mean mass carried by the top-2 routing weights.
- `debug/routing_top4_mass`: mean mass carried by the top-4 routing weights.
- `debug/prototype_active_count_eps_1e-3`: number of prototypes whose batch-mean usage exceeds `1e-3`.
- `debug/prototype_active_count_eps_1e-2`: number of prototypes whose batch-mean usage exceeds `1e-2`.
- `debug/routing_top1_active_count_window_100`: number of prototypes that appeared as a top-1 winner in the last 100 training batches.
- `debug/routing_top1_active_count_window_500`: number of prototypes that appeared as a top-1 winner in the last 500 training batches.
- `debug/routing_top1_dead_count_window_100`: number of prototypes that never appeared as a top-1 winner in the last 100 training batches.
- `debug/routing_top1_dead_count_window_500`: number of prototypes that never appeared as a top-1 winner in the last 500 training batches.
- `debug/routing_top1_usage_entropy_window_100`: entropy of the normalized cumulative top-1 histogram over the last 100 training batches.
- `debug/routing_top1_usage_entropy_window_500`: entropy of the normalized cumulative top-1 histogram over the last 500 training batches.
- `debug/routing_top1_usage_max_window_100`: largest normalized top-1 winner fraction over the last 100 training batches.
- `debug/routing_top1_usage_max_window_500`: largest normalized top-1 winner fraction over the last 500 training batches.
- `debug/prototype_active_count_eps_1e-3_window_100`: number of prototypes whose rolling-window average usage exceeds `1e-3` over the last 100 training batches.
- `debug/prototype_active_count_eps_1e-3_window_500`: number of prototypes whose rolling-window average usage exceeds `1e-3` over the last 500 training batches.
- `debug/prototype_active_count_eps_1e-2_window_100`: number of prototypes whose rolling-window average usage exceeds `1e-2` over the last 100 training batches.
- `debug/prototype_active_count_eps_1e-2_window_500`: number of prototypes whose rolling-window average usage exceeds `1e-2` over the last 500 training batches.
- `debug/prototype_usage_entropy_window_100`: entropy of the normalized rolling-window average usage distribution over the last 100 training batches.
- `debug/prototype_usage_entropy_window_500`: entropy of the normalized rolling-window average usage distribution over the last 500 training batches.
- `debug/prototype_usage_max_window_100`: maximum entry of the normalized rolling-window average usage distribution over the last 100 training batches.
- `debug/prototype_usage_max_window_500`: maximum entry of the normalized rolling-window average usage distribution over the last 500 training batches.

### Top-k reconstruction certification
- `debug/diag_cos_full`: mean diagonal cosine between the full surrogate embedding and exact text embedding.
- `debug/diag_cos_top1`: mean diagonal cosine after truncating routing to top-1 and renormalizing.
- `debug/diag_cos_top2`: mean diagonal cosine after truncating routing to top-2 and renormalizing.
- `debug/diag_cos_top4`: mean diagonal cosine after truncating routing to top-4 and renormalizing.
- `debug/loss_diag_full`: current diagonal fidelity loss for the full surrogate path.
- `debug/loss_diag_top1`: diagonal fidelity loss for the top-1 truncated surrogate.
- `debug/loss_diag_top2`: diagonal fidelity loss for the top-2 truncated surrogate.
- `debug/loss_diag_top4`: diagonal fidelity loss for the top-4 truncated surrogate.

### Prototype geometry and contextualization
- `debug/prototype_pairwise_cosine_mean`: mean off-diagonal cosine similarity among base prototypes.
- `debug/prototype_pairwise_cosine_std`: standard deviation of off-diagonal cosine similarity among base prototypes.
- `debug/prototype_pairwise_cosine_max`: maximum off-diagonal cosine similarity among base prototypes.
- `debug/contextualized_prototype_pairwise_cosine_mean`: mean off-diagonal cosine similarity among contextualized prototypes.
- `debug/contextualized_prototype_pairwise_cosine_std`: standard deviation of off-diagonal cosine similarity among contextualized prototypes.
- `debug/contextualized_prototype_pairwise_cosine_max`: maximum off-diagonal cosine similarity among contextualized prototypes.
- `debug/prototype_contextualization_entropy`: entropy of the prototype contextualization attention weights. Higher means contextualization is more diffuse.

### Token-pooling diagnostics
- `debug/token_pool_entropy`: entropy of token weights used by the exact text pooler.
- `debug/beta_max_prob`: mean maximum token weight assigned by the exact text pooler.
- `debug/token_special_mass`: average total attention mass assigned to special tokens such as CLS and EOS.
- `debug/token_valid_fraction`: fraction of token positions that are valid before keep-policy filtering.
- `debug/valid_token_fraction`: fraction of token positions that remain after the keep-policy mask.

### Proxy-logit diagnostics
- `debug/image_proxy_logit_mean`: mean image proxy logit.
- `debug/image_proxy_logit_std`: standard deviation of image proxy logits.
- `debug/image_proxy_logit_min`: minimum image proxy logit.
- `debug/image_proxy_logit_max`: maximum image proxy logit.
- `debug/image_positive_proxy_cosine_mean`: mean cosine similarity between image embeddings and their correct class proxy.
- `debug/image_positive_proxy_cosine_std`: standard deviation of positive image-proxy cosine similarity.
- `debug/image_hardest_negative_proxy_cosine_mean`: mean cosine similarity to the hardest negative proxy for image embeddings.
- `debug/image_hardest_negative_proxy_cosine_std`: standard deviation of hardest-negative image-proxy cosine similarity.
- `debug/image_proxy_margin_mean`: mean positive-minus-hardest-negative margin for image embeddings.
- `debug/image_proxy_margin_min`: minimum positive-minus-hardest-negative margin for image embeddings.
- `debug/text_proxy_logit_mean`: mean surrogate-text proxy logit.
- `debug/text_proxy_logit_std`: standard deviation of surrogate-text proxy logits.
- `debug/text_proxy_logit_min`: minimum surrogate-text proxy logit.
- `debug/text_proxy_logit_max`: maximum surrogate-text proxy logit.
- `debug/text_positive_proxy_cosine_mean`: mean cosine similarity between surrogate text embeddings and their correct class proxy.
- `debug/text_positive_proxy_cosine_std`: standard deviation of positive surrogate-text proxy cosine similarity.
- `debug/text_hardest_negative_proxy_cosine_mean`: mean cosine similarity to the hardest negative proxy for surrogate text embeddings.
- `debug/text_hardest_negative_proxy_cosine_std`: standard deviation of hardest-negative surrogate-text proxy cosine similarity.
- `debug/text_proxy_margin_mean`: mean positive-minus-hardest-negative margin for surrogate text embeddings.
- `debug/text_proxy_margin_min`: minimum positive-minus-hardest-negative margin for surrogate text embeddings.
- `debug/text_exact_proxy_logit_mean`: mean exact-text proxy logit.
- `debug/text_exact_proxy_logit_std`: standard deviation of exact-text proxy logits.
- `debug/text_exact_proxy_logit_min`: minimum exact-text proxy logit.
- `debug/text_exact_proxy_logit_max`: maximum exact-text proxy logit.
- `debug/text_exact_positive_proxy_cosine_mean`: mean cosine similarity between exact text embeddings and their correct class proxy.
- `debug/text_exact_positive_proxy_cosine_std`: standard deviation of positive exact-text proxy cosine similarity.
- `debug/text_exact_hardest_negative_proxy_cosine_mean`: mean cosine similarity to the hardest negative proxy for exact text embeddings.
- `debug/text_exact_hardest_negative_proxy_cosine_std`: standard deviation of hardest-negative exact-text proxy cosine similarity.
- `debug/text_exact_proxy_margin_mean`: mean positive-minus-hardest-negative margin for exact text embeddings.
- `debug/text_exact_proxy_margin_min`: minimum positive-minus-hardest-negative margin for exact text embeddings.

### Exact Retrieval Diagnostics
- `debug/image_surrogate_positive_cosine_mean`: mean positive image-vs-surrogate cosine on the current batch.
- `debug/image_surrogate_hardest_negative_cosine_mean`: mean hardest-negative image-vs-surrogate cosine on the current batch.
- `debug/image_surrogate_margin_mean`: mean positive-minus-hardest-negative surrogate margin.
- `debug/image_exact_positive_cosine_mean`: mean positive exact image-to-text cosine from the in-batch deployed scorer.
- `debug/image_exact_hardest_negative_cosine_mean`: mean hardest-negative exact cosine from the in-batch deployed scorer.
- `debug/image_exact_margin_mean`: mean positive-minus-hardest-negative exact margin.
- `debug/image_exact_positive_logit_mean`: mean scaled positive exact score on the training batch.
- `debug/image_exact_hardest_negative_logit_mean`: mean scaled hardest-negative exact score on the training batch.
- `debug/exact_pairwise_logit_mean`: mean of the in-batch exact pairwise logits used by `L_ret_exact`.
- `debug/exact_pairwise_logit_std`: standard deviation of the in-batch exact pairwise logits.
- `debug/exact_pairwise_logit_scale_or_norm`: effective logit scale used for `L_ret_exact`.

### Class-proxy parameter norms
- `debug/class_proxy_norm_mean`: mean L2 norm of raw class-proxy parameters.
- `debug/class_proxy_norm_std`: standard deviation of raw class-proxy norms.
- `debug/class_proxy_norm_min`: minimum raw class-proxy norm.
- `debug/class_proxy_norm_max`: maximum raw class-proxy norm.
- `debug/class_proxy_norm_normalized_mean`: mean norm after explicit L2 normalization of class proxies.
- `debug/class_proxy_norm_normalized_std`: standard deviation of normalized class-proxy norms.
- `debug/class_proxy_norm_normalized_min`: minimum normalized class-proxy norm.
- `debug/class_proxy_norm_normalized_max`: maximum normalized class-proxy norm.

### Gradient-norm diagnostics
- `debug/grad_norm_class_proxies`: gradient norm of the class-proxy parameters.
- `debug/grad_norm_image_projector`: gradient norm of the image projector and image adapter.
- `debug/grad_norm_text_projector`: gradient norm of the text projector and text adapter.
- `debug/grad_norm_prototype_bank`: gradient norm of the prototype-bank parameters.
- `debug/grad_norm_image_backbone`: gradient norm of the image backbone.
- `debug/grad_norm_text_backbone`: gradient norm of the text backbone.
- `debug/grad_norm_image_projected_output`: gradient norm observed at the image projected output tensor.
- `debug/grad_norm_surrogate_text_projected_output`: gradient norm observed at the surrogate text projected output tensor.
- `debug/grad_norm_exact_text_projected_output`: gradient norm observed at the exact text projected output tensor.
- `debug/exact_branch_grad_norm`: gradient norm observed at the exact pairwise retrieval logits used by `L_ret_exact`.
- `debug/grad_norm_total`: total gradient norm across all parameters.

## Epoch Coverage Metrics

These are logged once per epoch when `logging.log_debug_metrics=true`. They summarize cumulative routing coverage across the completed training epoch.


## Validation Metrics

These are logged during evaluation.

- `val/epoch`: epoch at which validation was run.
- `val/top1`: alias for `R1`, used as the primary model-selection metric.
- `val/pas/R1`: rank-1 retrieval accuracy.
- `val/pas/R5`: rank-5 retrieval accuracy.
- `val/pas/R10`: rank-10 retrieval accuracy.
- `val/pas/mAP`: mean average precision for retrieval.
- `val/pas/mINP`: mean inverse negative penalty.
- `val/pas/rSum`: `R1 + R5 + R10`.

The `val/pas/*` metrics that appear depend on `evaluation.retrieval_metrics`. By default the repo requests all six.

## Metrics Not Currently Sent to W&B

These exist in code or helper interfaces but are not currently emitted as time-series W&B metrics in the normal training path:
- prototype initialization diagnostics such as init mode, source path, clustering iterations, empty-cluster reseeds, and row-norm summaries
- `val/loss_total` in `build_validation_metrics(...)`, because validation currently logs evaluator metrics only and does not pass a validation loss
- full tensors such as routing weights, token weights, similarity matrices, basis banks, token masks, and logits arrays

## Quick Reading Guide

If you want a compact subset to watch first, start with:
- optimization: `train/loss_total`, `train/lr`
- retrieval quality: `val/top1`, `val/pas/mAP`, `val/pas/rSum`
- routing health: `debug/prototype_usage_entropy`, `debug/prototype_dead_count`, `debug/routing_effective_support`
- cross-batch coverage: `debug/routing_top1_active_count_window_500`, `debug/prototype_usage_entropy_window_500`
- geometry health: `debug/prototype_pairwise_cosine_max`, `debug/contextualized_prototype_pairwise_cosine_max`
- token pooling: `debug/token_pool_entropy`, `debug/token_special_mass`
- optimization stability: `debug/grad_norm_total`, `debug/logit_scale`
