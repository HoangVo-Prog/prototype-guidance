# W&B Metrics Reference

This document is the current metric surface for W&B logging in this repo.

Code sources:
- `processor/processor.py`
- `utils/metric_logging.py`
- `utils/metrics.py`

## Logging Behavior

- Training console logs now print only loss keys (plus `Base Lr`).
- W&B still logs all non-loss diagnostics under `debug/*`.
- Validation logs are written under `val/*`.

## W&B Namespaces

- `train/*`: loss and schedule metrics.
- `debug/*`: diagnostic metrics (routing, gradients, geometry, similarity stats, norms).
- `val/*`: validation loss metrics, retrieval metrics, and validation diagnostics.

## Train Metrics (`train/*`)

Always logged schedule keys:
- `train/epoch`
- `train/step`
- `train/lr`

Loss keys (from `TRAIN_LOSS_KEYS`):

```text
train/loss_total
train/loss_host
train/loss_host_ret
train/loss_host_ret_i2t
train/loss_host_ret_t2i
train/loss_host_cid
train/loss_proto_total
train/loss_host_weighted
train/lambda_host
train/loss_proxy
train/loss_proxy_image
train/loss_proxy_text
train/loss_proxy_text_exact
train/loss_ret
train/loss_align
train/loss_diag
train/loss_support
train/loss_diversity
train/loss_balance
train/loss_proxy_image_weighted
train/loss_proxy_text_weighted
train/loss_proxy_text_exact_weighted
train/loss_proxy_weighted
train/loss_ret_weighted
train/loss_align_weighted
train/loss_diag_weighted
train/loss_support_weighted
train/loss_diversity_weighted
train/loss_balance_weighted
```

## Debug Metrics (`debug/*`)

Full debug key inventory (from `DEBUG_METRIC_MAP`):

```text
debug/logit_scale
debug/host_logit_scale
debug/host_retrieval_temperature
debug/fusion_coefficient
debug/host_loss_total
debug/host_loss_ret
debug/proxy_temperature
debug/retrieval_temperature
debug/image_embed_norm_std
debug/image_embed_norm_min
debug/image_embed_norm_max
debug/text_embed_norm_std
debug/text_embed_norm_min
debug/text_embed_norm_max
debug/prototype_usage_entropy
debug/prototype_usage_max
debug/prototype_dead_count
debug/routing_max_prob
debug/routing_entropy
debug/routing_top1_usage_entropy
debug/routing_top1_usage_max
debug/routing_top1_dead_count
debug/routing_top1_active_count_window_100
debug/routing_top1_active_count_window_500
debug/routing_top1_dead_count_window_100
debug/routing_top1_dead_count_window_500
debug/routing_top1_usage_entropy_window_100
debug/routing_top1_usage_entropy_window_500
debug/routing_top1_usage_max_window_100
debug/routing_top1_usage_max_window_500
debug/prototype_assignment_entropy
debug/routing_effective_support
debug/routing_effective_support_ipr
debug/routing_effective_support_ipr_p10
debug/routing_effective_support_ipr_p50
debug/routing_effective_support_ipr_p90
debug/routing_support_below_2_frac
debug/routing_support_below_3_frac
debug/routing_support_below_min_frac
debug/routing_top1_minus_top2
debug/routing_top2_mass
debug/routing_top4_mass
debug/diag_cos_full
debug/diag_cos_top1
debug/diag_cos_top2
debug/diag_cos_top4
debug/loss_diag_full
debug/loss_diag_top1
debug/loss_diag_top2
debug/loss_diag_top4
debug/prototype_active_count_eps_1e-3
debug/prototype_active_count_eps_1e-2
debug/prototype_active_count_eps_1e-3_window_100
debug/prototype_active_count_eps_1e-3_window_500
debug/prototype_active_count_eps_1e-2_window_100
debug/prototype_active_count_eps_1e-2_window_500
debug/prototype_usage_entropy_window_100
debug/prototype_usage_entropy_window_500
debug/prototype_usage_max_window_100
debug/prototype_usage_max_window_500
debug/image_proxy_logit_mean
debug/image_proxy_logit_std
debug/image_proxy_logit_min
debug/image_proxy_logit_max
debug/text_proxy_logit_mean
debug/text_proxy_logit_std
debug/text_proxy_logit_min
debug/text_proxy_logit_max
debug/image_positive_proxy_cosine_mean
debug/image_positive_proxy_cosine_std
debug/image_hardest_negative_proxy_cosine_mean
debug/image_hardest_negative_proxy_cosine_std
debug/image_proxy_margin_mean
debug/image_proxy_margin_min
debug/text_positive_proxy_cosine_mean
debug/text_positive_proxy_cosine_std
debug/text_hardest_negative_proxy_cosine_mean
debug/text_hardest_negative_proxy_cosine_std
debug/text_proxy_margin_mean
debug/text_proxy_margin_min
debug/text_exact_positive_proxy_cosine_mean
debug/text_exact_positive_proxy_cosine_std
debug/text_exact_hardest_negative_proxy_cosine_mean
debug/text_exact_hardest_negative_proxy_cosine_std
debug/text_exact_proxy_margin_mean
debug/text_exact_proxy_margin_min
debug/image_surrogate_positive_cosine_mean
debug/image_surrogate_positive_cosine_std
debug/image_surrogate_hardest_negative_cosine_mean
debug/image_surrogate_hardest_negative_cosine_std
debug/image_surrogate_margin_mean
debug/image_surrogate_margin_min
debug/image_surrogate_positive_logit_mean
debug/image_surrogate_hardest_negative_logit_mean
debug/image_exact_positive_cosine_mean
debug/image_exact_positive_cosine_std
debug/image_exact_hardest_negative_cosine_mean
debug/image_exact_hardest_negative_cosine_std
debug/image_exact_margin_mean
debug/image_exact_margin_min
debug/image_exact_positive_logit_mean
debug/image_exact_hardest_negative_logit_mean
debug/surrogate_pairwise_positive_cosine_mean
debug/surrogate_pairwise_positive_cosine_std
debug/surrogate_pairwise_hardest_negative_cosine_mean
debug/surrogate_pairwise_hardest_negative_cosine_std
debug/surrogate_pairwise_margin_mean
debug/surrogate_pairwise_margin_min
debug/surrogate_pairwise_positive_logit_mean
debug/surrogate_pairwise_hardest_negative_logit_mean
debug/surrogate_pairwise_logit_mean
debug/surrogate_pairwise_logit_std
debug/class_proxy_norm_mean
debug/class_proxy_norm_std
debug/class_proxy_norm_min
debug/class_proxy_norm_max
debug/class_proxy_norm_normalized_mean
debug/class_proxy_norm_normalized_std
debug/class_proxy_norm_normalized_min
debug/class_proxy_norm_normalized_max
debug/grad_norm_class_proxies
debug/grad_norm_image_projector
debug/grad_norm_text_projector
debug/grad_norm_prototype_bank
debug/grad_norm_image_backbone
debug/grad_norm_text_backbone
debug/grad_norm_image_projected_output
debug/grad_norm_surrogate_text_projected_output
debug/grad_norm_exact_text_projected_output
debug/surrogate_retrieval_grad_norm
debug/grad_norm_total
debug/token_pool_entropy
debug/beta_max_prob
debug/token_special_mass
debug/token_valid_fraction
debug/valid_token_fraction
debug/prototype_pairwise_cosine_mean
debug/prototype_pairwise_cosine_std
debug/prototype_pairwise_cosine_max
debug/contextualized_prototype_pairwise_cosine_mean
debug/contextualized_prototype_pairwise_cosine_std
debug/contextualized_prototype_pairwise_cosine_max
debug/prototype_contextualization_entropy
debug/q_norm
debug/surrogate_t_pool_norm
debug/exact_t_pool_norm
debug/image_feature_norm
debug/image_embed_norm_raw
debug/image_embed_unit_norm
debug/surrogate_text_embed_norm_raw
debug/surrogate_text_embed_unit_norm
debug/exact_text_embed_norm_raw
debug/exact_text_embed_unit_norm
```

## Validation Metrics (`val/*`)

Always possible:
- `val/epoch`
- `val/top1`

Validation loss keys (same set as train losses, produced by `collect_loss_metrics`):

```text
val/loss_total
val/loss_host
val/loss_host_ret
val/loss_host_ret_i2t
val/loss_host_ret_t2i
val/loss_host_cid
val/loss_proto_total
val/loss_host_weighted
val/lambda_host
val/loss_proxy
val/loss_proxy_image
val/loss_proxy_text
val/loss_proxy_text_exact
val/loss_ret
val/loss_align
val/loss_diag
val/loss_support
val/loss_diversity
val/loss_balance
val/loss_proxy_image_weighted
val/loss_proxy_text_weighted
val/loss_proxy_text_exact_weighted
val/loss_proxy_weighted
val/loss_ret_weighted
val/loss_align_weighted
val/loss_diag_weighted
val/loss_support_weighted
val/loss_diversity_weighted
val/loss_balance_weighted
```

Retrieval metrics (`evaluation.retrieval_metrics` subset of this list):

```text
val/pas/R1
val/pas/R5
val/pas/R10
val/pas/mAP
val/pas/mINP
val/pas/rSum
```

Validation debug metrics (`Evaluator._compute_eval_debug_metrics`):

```text
val/debug/eval_positive_gallery_count_min
val/debug/eval_positive_gallery_count_mean
val/debug/eval_logit_scale
val/debug/eval_retrieval_temperature
val/debug/eval_positive_exact_cosine_mean
val/debug/eval_hardest_negative_exact_cosine_mean
val/debug/eval_exact_margin_mean
val/debug/eval_image_projected_norm_mean
val/debug/eval_image_projected_norm_std
val/debug/eval_positive_exact_text_embed_norm_mean
val/debug/eval_positive_exact_text_embed_norm_std
val/debug/eval_positive_exact_text_embed_unit_norm_mean
val/debug/eval_positive_exact_pair_cosine_mean
```

## Conditional Notes

- `debug/*` keys are emitted only when `logging.log_debug_metrics=true`.
- Some keys can be consistently zero when corresponding branches/losses are disabled by config.
- `val/pas/*` keys depend on `evaluation.retrieval_metrics` (only requested metrics are logged).
- Some `val/debug/*` keys are conditional on runtime path:
  - `val/debug/eval_logit_scale`, `val/debug/eval_retrieval_temperature` require prototype-loss module exposure.
  - Text-exact debug keys require exact retrieval path data availability.
