# CLI Flag Reference

Generated from `utils/options.py` and config maps in `utils/config.py`.

Notes:
- `itself_reference` support has been removed from runtime/config parsing.
- `Primary Config Keys` are canonical YAML keys.
- `Alias Config Keys` are backward-compatible keys still accepted by loader.

## `config_file`
- Flags: `--config_file`
- Type/Action: -
- Default: `''`
- Primary Config Keys: -
- Alias Config Keys: -
- Help: Optional YAML config override file

## `local_rank`
- Flags: `--local_rank`
- Type/Action: type=int
- Default: `0`
- Primary Config Keys: `experiment.local_rank`
- Alias Config Keys: -

## `output_dir`
- Flags: `--output_dir`
- Type/Action: -
- Default: `'runs'`
- Primary Config Keys: `experiment.output_dir`
- Alias Config Keys: -

## `name`
- Flags: `--name`
- Type/Action: -
- Default: `'pas_v1'`
- Primary Config Keys: `experiment.name`
- Alias Config Keys: -
- Help: Experiment name

## `seed`
- Flags: `--seed`
- Type/Action: type=int
- Default: `1`
- Primary Config Keys: `experiment.seed`
- Alias Config Keys: -

## `model_name`
- Flags: `--model_name`
- Type/Action: -
- Default: `'PAS'`
- Primary Config Keys: `model.name`
- Alias Config Keys: -

## `model_variant`
- Flags: `--model_variant`
- Type/Action: -
- Default: `'pas_v1'`
- Primary Config Keys: `model.variant`
- Alias Config Keys: -

## `training_mode`
- Flags: `--training_mode`
- Type/Action: type=str
- Default: `'pas'`
- Primary Config Keys: `model.training_mode`
- Alias Config Keys: -

## `pretrain_choice`
- Flags: `--pretrain_choice`
- Type/Action: -
- Default: `'ViT-B/16'`
- Primary Config Keys: `model.pretrain_choice`
- Alias Config Keys: -

## `image_backbone`
- Flags: `--image_backbone`
- Type/Action: -
- Default: `'clip_visual'`
- Primary Config Keys: `model.image_backbone`
- Alias Config Keys: -

## `text_backbone`
- Flags: `--text_backbone`
- Type/Action: -
- Default: `'clip_text_transformer'`
- Primary Config Keys: `model.text_backbone`
- Alias Config Keys: -

## `embedding_dim`
- Flags: `--embedding_dim`
- Type/Action: type=int
- Default: `512`
- Primary Config Keys: `model.embedding_dim`
- Alias Config Keys: -

## `projection_dim`
- Flags: `--projector_output_dim`, `--projection_dim`
- Type/Action: type=int
- Default: `256`
- Primary Config Keys: `model.projection_dim`
- Alias Config Keys: `model.projector_output_dim`

## `projector_hidden_dim`
- Flags: `--projector_hidden_dim`
- Type/Action: type=int
- Default: `512`
- Primary Config Keys: `model.projector_hidden_dim`
- Alias Config Keys: -

## `projector_dropout`
- Flags: `--projector_dropout`
- Type/Action: type=float
- Default: `0.0`
- Primary Config Keys: `model.projector_dropout`
- Alias Config Keys: -

## `projector_type`
- Flags: `--projector_type`
- Type/Action: type=str
- Default: `'mlp2'`
- Primary Config Keys: `model.projector_type`
- Alias Config Keys: -

## `normalize_projector_outputs`
- Flags: `--normalize_projector_outputs`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `model.normalize_projector_outputs`
- Alias Config Keys: -

## `use_custom_projector`
- Flags: `--use_custom_projector`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `model.use_custom_projector`
- Alias Config Keys: `host.use_custom_projector`

## `backbone_precision`
- Flags: `--backbone_precision`
- Type/Action: type=str
- Default: `'fp16'`
- Primary Config Keys: `model.backbone_precision`
- Alias Config Keys: -

## `prototype_precision`
- Flags: `--prototype_precision`
- Type/Action: type=str
- Default: `'fp32'`
- Primary Config Keys: `model.prototype_precision`
- Alias Config Keys: -

## `temperature`
- Flags: `--temperature`
- Type/Action: type=float
- Default: `0.07`
- Primary Config Keys: `model.temperature`
- Alias Config Keys: -

## `proxy_temperature`
- Flags: `--proxy_temperature`
- Type/Action: type=float
- Default: `0.07`
- Primary Config Keys: `training.proxy_temperature`
- Alias Config Keys: -

## `host_type`
- Flags: `--host_type`
- Type/Action: type=str
- Default: `'clip'`
- Primary Config Keys: `host.type`
- Alias Config Keys: -

## `use_host_loss`
- Flags: `--use_host_loss`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `objectives.objectives.use_host_loss`
- Alias Config Keys: `host.enabled`, `objectives.use_host_loss`

## `lambda_host`
- Flags: `--lambda_host`
- Type/Action: type=float
- Default: `1.0`
- Primary Config Keys: `objectives.lambda.host`
- Alias Config Keys: `host.loss_weight`, `objectives.lambda_host`

## `itself_loss_names`
- Flags: `--itself_loss_names`, `--loss_names`
- Type/Action: type=str
- Default: `'tal+cid'`
- Primary Config Keys: `host.itself_loss_names`
- Alias Config Keys: `host.loss_names`

## `itself_only_global`
- Flags: `--itself_only_global`, `--only_global`
- Type/Action: type=_str2bool, action='store_true', nargs='?'
- Default: `False`
- Primary Config Keys: `host.itself_only_global`
- Alias Config Keys: `host.only_global`

## `itself_select_ratio`
- Flags: `--itself_select_ratio`, `--select_ratio`
- Type/Action: type=float
- Default: `0.4`
- Primary Config Keys: `host.itself_select_ratio`
- Alias Config Keys: `host.select_ratio`

## `itself_grab_embed_dim`
- Flags: `--itself_grab_embed_dim`
- Type/Action: type=int
- Default: `4096`
- Primary Config Keys: `host.itself_grab_embed_dim`
- Alias Config Keys: -

## `itself_score_weight_global`
- Flags: `--itself_score_weight_global`
- Type/Action: type=float
- Default: `0.68`
- Primary Config Keys: `host.itself_score_weight_global`
- Alias Config Keys: -

## `itself_tau`
- Flags: `--itself_tau`, `--tau`
- Type/Action: type=float
- Default: `0.015`
- Primary Config Keys: `host.itself_tau`
- Alias Config Keys: `host.tau`

## `itself_margin`
- Flags: `--itself_margin`, `--margin`
- Type/Action: type=float
- Default: `0.1`
- Primary Config Keys: `host.itself_margin`
- Alias Config Keys: `host.margin`

## `itself_return_all`
- Flags: `--itself_return_all`, `--return_all`
- Type/Action: type=_str2bool, action='store_true', nargs='?'
- Default: `False`
- Primary Config Keys: `host.itself_return_all`
- Alias Config Keys: `host.return_all`

## `itself_topk_type`
- Flags: `--itself_topk_type`, `--topk_type`
- Type/Action: type=str
- Default: `'mean'`
- Primary Config Keys: `host.itself_topk_type`
- Alias Config Keys: `host.topk_type`

## `itself_layer_index`
- Flags: `--itself_layer_index`, `--layer_index`
- Type/Action: type=int
- Default: `-1`
- Primary Config Keys: `host.itself_layer_index`
- Alias Config Keys: `host.layer_index`

## `itself_average_attn_weights`
- Flags: `--itself_average_attn_weights`, `--average_attn_weights`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `host.itself_average_attn_weights`
- Alias Config Keys: `host.average_attn_weights`

## `itself_modify_k`
- Flags: `--itself_modify_k`, `--modify_k`
- Type/Action: type=_str2bool, action='store_true', nargs='?'
- Default: `False`
- Primary Config Keys: `host.itself_modify_k`
- Alias Config Keys: `host.modify_k`

## `lambda1_weight`
- Flags: `--lambda1_weight`
- Type/Action: type=float
- Default: `0.5`
- Primary Config Keys: `host.itself_lambda1_weight`
- Alias Config Keys: `host.lambda1_weight`

## `lambda2_weight`
- Flags: `--lambda2_weight`
- Type/Action: type=float
- Default: `3.5`
- Primary Config Keys: `host.itself_lambda2_weight`
- Alias Config Keys: `host.lambda2_weight`

## `lambda_proxy`
- Flags: `--lambda_proxy`
- Type/Action: type=float
- Default: `1.0`
- Primary Config Keys: `objectives.lambda.proxy`
- Alias Config Keys: `loss.lambda_proxy`, `training.lambda_proxy`

## `lambda_proxy_image`
- Flags: `--lambda_proxy_image`
- Type/Action: type=float
- Default: `None`
- Primary Config Keys: `objectives.lambda.proxy_image`
- Alias Config Keys: `loss.lambda_proxy_image`, `training.lambda_proxy_image`

## `lambda_proxy_text`
- Flags: `--lambda_proxy_text`
- Type/Action: type=float
- Default: `None`
- Primary Config Keys: `objectives.lambda.proxy_text`
- Alias Config Keys: `loss.lambda_proxy_text`, `training.lambda_proxy_text`

## `lambda_proxy_text_exact`
- Flags: `--lambda_proxy_text_exact`
- Type/Action: type=float
- Default: `None`
- Primary Config Keys: `objectives.lambda.proxy_text_exact`
- Alias Config Keys: `loss.lambda_proxy_text_exact`, `training.lambda_proxy_text_exact`

## `use_loss_proxy_image`
- Flags: `--use_loss_proxy_image`
- Type/Action: type=_str2bool, nargs='?'
- Default: `False`
- Primary Config Keys: `objectives.objectives.use_loss_proxy_image`
- Alias Config Keys: `loss.use_loss_proxy_image`, `training.use_loss_proxy_image`

## `use_loss_proxy_text`
- Flags: `--use_loss_proxy_text`
- Type/Action: type=_str2bool, nargs='?'
- Default: `False`
- Primary Config Keys: `objectives.objectives.use_loss_proxy_text`
- Alias Config Keys: `loss.use_loss_proxy_text`, `training.use_loss_proxy_text`

## `use_loss_proxy_text_exact`
- Flags: `--use_loss_proxy_text_exact`
- Type/Action: type=_str2bool, nargs='?'
- Default: `False`
- Primary Config Keys: `objectives.objectives.use_loss_proxy_text_exact`
- Alias Config Keys: `loss.use_loss_proxy_text_exact`, `training.use_loss_proxy_text_exact`

## `use_loss_align`
- Flags: `--use_loss_align`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `objectives.objectives.use_loss_align`
- Alias Config Keys: `loss.use_loss_align`, `training.use_loss_align`

## `lambda_align`
- Flags: `--lambda_align`
- Type/Action: type=float
- Default: `1.0`
- Primary Config Keys: `objectives.lambda.align`
- Alias Config Keys: `loss.lambda_align`, `training.lambda_align`

## `use_loss_diag`
- Flags: `--use_loss_diag`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `objectives.objectives.use_loss_diag`
- Alias Config Keys: `loss.use_loss_diag`, `objectives.use_diag_fidelity`, `training.use_loss_diag`

## `lambda_diag`
- Flags: `--lambda_diag`
- Type/Action: type=float
- Default: `1.0`
- Primary Config Keys: `objectives.lambda.diag`
- Alias Config Keys: `loss.lambda_diag`, `objectives.lambda_diag`, `training.lambda_diag`

## `use_loss_ret`
- Flags: `--use_loss_ret`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `objectives.objectives.use_loss_ret`
- Alias Config Keys: `loss.use_loss_ret`, `objectives.use_proto_loss_ret`, `training.use_loss_ret`

## `retrieval_mode`
- Flags: `--retrieval_mode`
- Type/Action: type=str
- Default: `'surrogate_i2t'`
- Primary Config Keys: `objectives.objectives.retrieval_mode`
- Alias Config Keys: `loss.retrieval_mode`

## `lambda_ret`
- Flags: `--lambda_ret`
- Type/Action: type=float
- Default: `1.0`
- Primary Config Keys: `objectives.lambda.ret`
- Alias Config Keys: `loss.lambda_ret`, `objectives.lambda_proto_ret`, `training.lambda_ret`

## `use_loss_support`
- Flags: `--use_loss_support`
- Type/Action: type=_str2bool, nargs='?'
- Default: `False`
- Primary Config Keys: `objectives.objectives.use_loss_support`
- Alias Config Keys: `loss.use_loss_support`

## `lambda_support`
- Flags: `--lambda_support`
- Type/Action: type=float
- Default: `0.0`
- Primary Config Keys: `objectives.lambda.support`
- Alias Config Keys: `loss.lambda_support`

## `support_min`
- Flags: `--support_min`
- Type/Action: type=float
- Default: `2.0`
- Primary Config Keys: `objectives.objectives.support_min`
- Alias Config Keys: `loss.support_min`

## `img_size`
- Flags: `--img_size`
- Type/Action: type=int, nargs=2
- Default: `(384, 128)`
- Primary Config Keys: `model.img_size`
- Alias Config Keys: -

## `stride_size`
- Flags: `--stride_size`
- Type/Action: type=int
- Default: `16`
- Primary Config Keys: `model.stride_size`
- Alias Config Keys: -

## `text_length`
- Flags: `--text_length`
- Type/Action: type=int
- Default: `77`
- Primary Config Keys: `model.text_length`
- Alias Config Keys: -

## `vocab_size`
- Flags: `--vocab_size`
- Type/Action: type=int
- Default: `49408`
- Primary Config Keys: `model.vocab_size`
- Alias Config Keys: -

## `use_prototype_branch`
- Flags: `--use_prototype_branch`
- Type/Action: type=_str2bool, nargs='?'
- Default: `None`
- Primary Config Keys: `model.use_prototype_branch`
- Alias Config Keys: -

## `use_prototype_bank`
- Flags: `--use_prototype_bank`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `model.use_prototype_bank`
- Alias Config Keys: -

## `use_image_conditioned_pooling`
- Flags: `--use_image_conditioned_pooling`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `model.use_image_conditioned_pooling`
- Alias Config Keys: -

## `fusion_enabled`
- Flags: `--fusion_enabled`
- Type/Action: type=_str2bool, nargs='?'
- Default: `None`
- Primary Config Keys: `fusion.enabled`
- Alias Config Keys: -

## `fusion_coefficient`
- Flags: `--fusion_coefficient`
- Type/Action: type=float
- Default: `1.0`
- Primary Config Keys: `fusion.coefficient`
- Alias Config Keys: -

## `fusion_coefficient_source`
- Flags: `--fusion_coefficient_source`
- Type/Action: type=str
- Default: `'fixed'`
- Primary Config Keys: `fusion.coefficient_source`
- Alias Config Keys: -

## `use_prototype_contextualization`
- Flags: `--use_prototype_contextualization`
- Type/Action: type=_str2bool, nargs='?'
- Default: `None`
- Primary Config Keys: -
- Alias Config Keys: `model.use_prototype_contextualization`

## `return_debug_outputs`
- Flags: `--return_debug_outputs`
- Type/Action: type=_str2bool, nargs='?'
- Default: `False`
- Primary Config Keys: `model.return_debug_outputs`
- Alias Config Keys: -

## `prototype_num_prototypes`
- Flags: `--prototype_num_prototypes`
- Type/Action: type=int
- Default: `32`
- Primary Config Keys: `prototype.num_prototypes`
- Alias Config Keys: -

## `prototype_dim`
- Flags: `--prototype_dim`
- Type/Action: type=int
- Default: `512`
- Primary Config Keys: `prototype.prototype_dim`
- Alias Config Keys: -

## `prototype_init`
- Flags: `--prototype_init`
- Type/Action: type=str
- Default: `'normalized_random'`
- Primary Config Keys: `prototype.prototype_init`
- Alias Config Keys: -

## `prototype_init_path`
- Flags: `--prototype_init_path`
- Type/Action: type=str
- Default: `''`
- Primary Config Keys: `prototype.prototype_init_path`
- Alias Config Keys: -

## `prototype_init_hybrid_ratio`
- Flags: `--prototype_init_hybrid_ratio`
- Type/Action: type=float
- Default: `0.5`
- Primary Config Keys: `prototype.prototype_init_hybrid_ratio`
- Alias Config Keys: -

## `prototype_init_max_iters`
- Flags: `--prototype_init_max_iters`
- Type/Action: type=int
- Default: `50`
- Primary Config Keys: `prototype.prototype_init_max_iters`
- Alias Config Keys: -

## `prototype_init_tol`
- Flags: `--prototype_init_tol`
- Type/Action: type=float
- Default: `0.0001`
- Primary Config Keys: `prototype.prototype_init_tol`
- Alias Config Keys: -

## `prototype_init_seed`
- Flags: `--prototype_init_seed`
- Type/Action: type=int
- Default: `None`
- Primary Config Keys: `prototype.prototype_init_seed`
- Alias Config Keys: -

## `prototype_routing_type`
- Flags: `--routing_similarity`, `--prototype_routing_type`
- Type/Action: type=str
- Default: `'cosine'`
- Primary Config Keys: `prototype.routing_type`
- Alias Config Keys: `prototype.routing_similarity`

## `prototype_temperature`
- Flags: `--tau_p`, `--prototype_temperature`
- Type/Action: type=float
- Default: `0.07`
- Primary Config Keys: `prototype.routing_temperature`
- Alias Config Keys: `prototype.tau_p`

## `prototype_contextualization_enabled`
- Flags: `--prototype_contextualization_enabled`
- Type/Action: type=_str2bool, nargs='?'
- Default: `None`
- Primary Config Keys: `prototype.contextualization_enabled`
- Alias Config Keys: -

## `prototype_contextualization_type`
- Flags: `--prototype_contextualization_type`
- Type/Action: type=str
- Default: `'self_attention'`
- Primary Config Keys: `prototype.contextualization_type`
- Alias Config Keys: -

## `prototype_contextualization_residual`
- Flags: `--prototype_contextualization_residual`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `prototype.contextualization_residual`
- Alias Config Keys: -

## `normalize_for_self_interaction`
- Flags: `--normalize_for_self_interaction`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `prototype.normalize_for_self_interaction`
- Alias Config Keys: -

## `normalize_for_routing`
- Flags: `--normalize_for_routing`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `prototype.normalize_for_routing`
- Alias Config Keys: -

## `use_balancing_loss`
- Flags: `--use_balancing_loss`
- Type/Action: type=_str2bool, nargs='?'
- Default: `False`
- Primary Config Keys: `objectives.objectives.use_balancing_loss`
- Alias Config Keys: `loss.use_balancing_loss`, `objectives.use_balance`, `prototype.use_balancing_loss`

## `prototype_balance_loss_weight`
- Flags: `--lambda_bal`, `--prototype_balance_loss_weight`
- Type/Action: type=float
- Default: `0.0`
- Primary Config Keys: `objectives.lambda.balance`
- Alias Config Keys: `loss.balance_loss_weight`, `objectives.lambda_balance`, `prototype.balance_loss_weight`, `prototype.lambda_bal`

## `prototype_dead_threshold`
- Flags: `--prototype_dead_threshold`
- Type/Action: type=float
- Default: `0.005`
- Primary Config Keys: `prototype.dead_prototype_threshold`
- Alias Config Keys: -

## `use_diversity_loss`
- Flags: `--use_diversity_loss`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `objectives.objectives.use_diversity_loss`
- Alias Config Keys: `loss.use_diversity_loss`, `objectives.use_diversity`, `prototype.use_diversity_loss`

## `diversity_loss_weight`
- Flags: `--lambda_div`, `--diversity_loss_weight`
- Type/Action: type=float
- Default: `0.01`
- Primary Config Keys: `objectives.lambda.diversity`
- Alias Config Keys: `loss.diversity_loss_weight`, `objectives.lambda_diversity`, `prototype.diversity_loss_weight`, `prototype.lambda_div`

## `token_policy`
- Flags: `--token_policy`
- Type/Action: type=str
- Default: `'content_only'`
- Primary Config Keys: `text_pooling.token_policy`
- Alias Config Keys: -

## `token_scoring_type`
- Flags: `--token_similarity`, `--token_scoring_type`
- Type/Action: type=str
- Default: `'cosine'`
- Primary Config Keys: `text_pooling.scoring_type`
- Alias Config Keys: `text_pooling.token_similarity`

## `normalize_for_token_scoring`
- Flags: `--normalize_for_token_scoring`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `text_pooling.normalize_for_token_scoring`
- Alias Config Keys: -

## `token_pooling_temperature`
- Flags: `--tau_t`, `--token_pooling_temperature`
- Type/Action: type=float
- Default: `0.07`
- Primary Config Keys: `text_pooling.token_temperature`
- Alias Config Keys: `text_pooling.tau_t`

## `error_on_empty_kept_tokens`
- Flags: `--error_on_empty_kept_tokens`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `text_pooling.error_on_empty_kept_tokens`
- Alias Config Keys: -

## `batch_size`
- Flags: `--batch_size`
- Type/Action: type=int
- Default: `32`
- Primary Config Keys: `training.batch_size`
- Alias Config Keys: -

## `training_stage`
- Flags: `--stage`, `--training_stage`
- Type/Action: type=str
- Default: `'joint'`
- Primary Config Keys: `training.stage`
- Alias Config Keys: `training.training_stage`

## `num_epoch`
- Flags: `--epochs`, `--num_epoch`
- Type/Action: type=int
- Default: `60`
- Primary Config Keys: `training.epochs`
- Alias Config Keys: `training.num_epoch`

## `log_period`
- Flags: `--log_period`
- Type/Action: type=int
- Default: `50`
- Primary Config Keys: `training.log_period`
- Alias Config Keys: -

## `eval_period`
- Flags: `--eval_frequency`, `--eval_period`
- Type/Action: type=int
- Default: `1`
- Primary Config Keys: `training.eval_frequency`
- Alias Config Keys: `training.eval_period`

## `save_interval`
- Flags: `--save_interval`
- Type/Action: type=int
- Default: `1`
- Primary Config Keys: `training.save_interval`
- Alias Config Keys: -

## `prototype_selection_metric`
- Flags: `--prototype-selection-metric`
- Type/Action: type=str
- Default: `None`
- Primary Config Keys: `training.prototype_selection_metric`
- Alias Config Keys: -
- Help: Deprecated. Use `checkpointing.metric` + `checkpointing.save.artifacts.*` for modular best/latest checkpointing.

## `grad_clip`
- Flags: `--grad_clip`
- Type/Action: type=float
- Default: `1.0`
- Primary Config Keys: `training.grad_clip`
- Alias Config Keys: -

## `amp`
- Flags: `--amp`
- Type/Action: type=_str2bool, nargs='?'
- Default: `False`
- Primary Config Keys: `training.amp`
- Alias Config Keys: -

## `amp_dtype`
- Flags: `--amp_dtype`
- Type/Action: type=str
- Default: `'fp16'`
- Primary Config Keys: `training.amp_dtype`
- Alias Config Keys: -

## `resume`
- Flags: `--resume`
- Type/Action: type=_str2bool, nargs='?'
- Default: `False`
- Primary Config Keys: `training.resume`
- Alias Config Keys: -

## `resume_ckpt_file`
- Flags: `--resume_ckpt_file`
- Type/Action: -
- Default: `''`
- Primary Config Keys: `training.resume_ckpt_file`
- Alias Config Keys: -
- Help: Resume from checkpoint path

## `finetune`
- Flags: `--finetune`
- Type/Action: type=str
- Default: `''`
- Primary Config Keys: `training.finetune`
- Alias Config Keys: -

## `pretrain`
- Flags: `--pretrain`
- Type/Action: type=str
- Default: `''`
- Primary Config Keys: `training.pretrain`
- Alias Config Keys: -

## `img_aug`
- Flags: `--img_aug`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `training.img_aug`
- Alias Config Keys: -

## `txt_aug`
- Flags: `--txt_aug`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `training.txt_aug`
- Alias Config Keys: -

## `sampler`
- Flags: `--sampler`
- Type/Action: -
- Default: `'identity'`
- Primary Config Keys: `training.sampler`
- Alias Config Keys: -
- Help: choose sampler from [identity, random]

## `num_instance`
- Flags: `--num_instance`
- Type/Action: type=int
- Default: `2`
- Primary Config Keys: `training.num_instance`
- Alias Config Keys: -

## `num_workers`
- Flags: `--num_workers`
- Type/Action: type=int
- Default: `4`
- Primary Config Keys: `dataset.num_workers`, `training.num_workers`
- Alias Config Keys: -

## `freeze_host_projectors`
- Flags: `--freeze_host_projectors`
- Type/Action: type=_str2bool, nargs='?'
- Default: `False`
- Primary Config Keys: `host.freeze_projectors`, `training.freeze_host_projectors`
- Alias Config Keys: -

## `freeze_image_backbone`
- Flags: `--freeze_image_backbone`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `training.freeze_image_backbone`
- Alias Config Keys: -

## `freeze_text_backbone`
- Flags: `--freeze_text_backbone`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `training.freeze_text_backbone`
- Alias Config Keys: -

## `freeze_prototype_side`
- Flags: `--freeze_prototype_side`
- Type/Action: type=_str2bool, nargs='?'
- Default: `False`
- Primary Config Keys: `training.freeze_prototype_side`
- Alias Config Keys: -

## `training`
- Flags: `--test`
- Type/Action: action='store_false'
- Default: `True`
- Primary Config Keys: `training.training`
- Alias Config Keys: -

## `optimizer`
- Flags: `--optimizer_type`, `--optimizer`
- Type/Action: type=str
- Default: `'AdamW'`
- Primary Config Keys: `optimizer.type`
- Alias Config Keys: `optimizer.optimizer`
- Help: [SGD, Adam, AdamW]

## `lr`
- Flags: `--lr`
- Type/Action: type=float
- Default: `0.001`
- Primary Config Keys: `optimizer.lr`
- Alias Config Keys: -

## `lr_prototype_bank`
- Flags: `--lr_prototype_bank`
- Type/Action: type=float
- Default: `0.001`
- Primary Config Keys: `optimizer.lr_prototype_bank`
- Alias Config Keys: -

## `lr_projectors`
- Flags: `--lr_projectors`
- Type/Action: type=float
- Default: `0.001`
- Primary Config Keys: `optimizer.lr_projectors`
- Alias Config Keys: -

## `lr_prototype_routing`
- Flags: `--lr_prototype_routing`
- Type/Action: type=float
- Default: `None`
- Primary Config Keys: `optimizer.lr_prototype_routing`
- Alias Config Keys: -

## `lr_prototype_pooling`
- Flags: `--lr_prototype_pooling`
- Type/Action: type=float
- Default: `None`
- Primary Config Keys: `optimizer.lr_prototype_pooling`
- Alias Config Keys: -

## `lr_prototype_contextualization`
- Flags: `--lr_prototype_contextualization`
- Type/Action: type=float
- Default: `None`
- Primary Config Keys: `optimizer.lr_prototype_contextualization`
- Alias Config Keys: -

## `lr_host_projectors`
- Flags: `--lr_host_projectors`
- Type/Action: type=float
- Default: `None`
- Primary Config Keys: `optimizer.lr_host_projectors`
- Alias Config Keys: -

## `lr_image_backbone`
- Flags: `--lr_image_backbone`
- Type/Action: type=float
- Default: `0.0`
- Primary Config Keys: `optimizer.lr_image_backbone`
- Alias Config Keys: -

## `lr_text_backbone`
- Flags: `--lr_text_backbone`
- Type/Action: type=float
- Default: `0.0`
- Primary Config Keys: `optimizer.lr_text_backbone`
- Alias Config Keys: -

## `momentum`
- Flags: `--momentum`
- Type/Action: type=float
- Default: `0.9`
- Primary Config Keys: `optimizer.momentum`
- Alias Config Keys: -

## `weight_decay`
- Flags: `--weight_decay`
- Type/Action: type=float
- Default: `0.01`
- Primary Config Keys: `optimizer.weight_decay`
- Alias Config Keys: -

## `weight_decay_prototype_bank`
- Flags: `--weight_decay_prototype_bank`
- Type/Action: type=float
- Default: `0.01`
- Primary Config Keys: `optimizer.weight_decay_prototype_bank`
- Alias Config Keys: `optimizer.weight_decay_prototypes`

## `weight_decay_projectors`
- Flags: `--weight_decay_projectors`
- Type/Action: type=float
- Default: `0.05`
- Primary Config Keys: `optimizer.weight_decay_projectors`
- Alias Config Keys: -

## `weight_decay_prototype_routing`
- Flags: `--weight_decay_prototype_routing`
- Type/Action: type=float
- Default: `None`
- Primary Config Keys: `optimizer.weight_decay_prototype_routing`
- Alias Config Keys: -

## `weight_decay_prototype_pooling`
- Flags: `--weight_decay_prototype_pooling`
- Type/Action: type=float
- Default: `None`
- Primary Config Keys: `optimizer.weight_decay_prototype_pooling`
- Alias Config Keys: -

## `weight_decay_prototype_contextualization`
- Flags: `--weight_decay_prototype_contextualization`
- Type/Action: type=float
- Default: `None`
- Primary Config Keys: `optimizer.weight_decay_prototype_contextualization`
- Alias Config Keys: -

## `weight_decay_host_projectors`
- Flags: `--weight_decay_host_projectors`
- Type/Action: type=float
- Default: `None`
- Primary Config Keys: `optimizer.weight_decay_host_projectors`
- Alias Config Keys: -

## `lr_class_proxies`
- Flags: `--lr_class_proxies`
- Type/Action: type=float
- Default: `None`
- Primary Config Keys: `optimizer.lr_class_proxies`
- Alias Config Keys: -

## `weight_decay_class_proxies`
- Flags: `--weight_decay_class_proxies`
- Type/Action: type=float
- Default: `None`
- Primary Config Keys: `optimizer.weight_decay_class_proxies`
- Alias Config Keys: -

## `weight_decay_image_backbone`
- Flags: `--weight_decay_image_backbone`
- Type/Action: type=float
- Default: `0.0`
- Primary Config Keys: `optimizer.weight_decay_image_backbone`
- Alias Config Keys: -

## `weight_decay_text_backbone`
- Flags: `--weight_decay_text_backbone`
- Type/Action: type=float
- Default: `0.0`
- Primary Config Keys: `optimizer.weight_decay_text_backbone`
- Alias Config Keys: -

## `alpha`
- Flags: `--alpha`
- Type/Action: type=float
- Default: `0.9`
- Primary Config Keys: `optimizer.alpha`
- Alias Config Keys: -

## `beta`
- Flags: `--beta`
- Type/Action: type=float
- Default: `0.999`
- Primary Config Keys: `optimizer.beta`
- Alias Config Keys: -

## `lr_factor`
- Flags: `--lr_factor`
- Type/Action: type=float
- Default: `5.0`
- Primary Config Keys: `optimizer.lr_factor`
- Alias Config Keys: -

## `bias_lr_factor`
- Flags: `--bias_lr_factor`
- Type/Action: type=float
- Default: `2.0`
- Primary Config Keys: `optimizer.bias_lr_factor`
- Alias Config Keys: -

## `weight_decay_bias`
- Flags: `--weight_decay_bias`
- Type/Action: type=float
- Default: `0.0`
- Primary Config Keys: `optimizer.weight_decay_bias`
- Alias Config Keys: -

## `optimizer_eps`
- Flags: `--optimizer_eps`
- Type/Action: type=float
- Default: `1e-08`
- Primary Config Keys: `optimizer.optimizer_eps`
- Alias Config Keys: -

## `milestones`
- Flags: `--milestones`
- Type/Action: type=int, nargs='+'
- Default: `(40, 50)`
- Primary Config Keys: `optimizer.milestones`
- Alias Config Keys: -

## `gamma`
- Flags: `--gamma`
- Type/Action: type=float
- Default: `0.1`
- Primary Config Keys: `optimizer.gamma`
- Alias Config Keys: -

## `warmup_factor`
- Flags: `--warmup_factor`
- Type/Action: type=float
- Default: `0.1`
- Primary Config Keys: `optimizer.warmup_factor`
- Alias Config Keys: -

## `warmup_epochs`
- Flags: `--warmup_epochs`
- Type/Action: type=int
- Default: `5`
- Primary Config Keys: `optimizer.warmup_epochs`
- Alias Config Keys: -

## `warmup_method`
- Flags: `--warmup_method`
- Type/Action: type=str
- Default: `'linear'`
- Primary Config Keys: `optimizer.warmup_method`
- Alias Config Keys: -

## `lrscheduler`
- Flags: `--lr_scheduler`, `--lrscheduler`
- Type/Action: type=str
- Default: `'cosine'`
- Primary Config Keys: `optimizer.scheduler`
- Alias Config Keys: `optimizer.lrscheduler`

## `target_lr`
- Flags: `--target_lr`
- Type/Action: type=float
- Default: `0.0`
- Primary Config Keys: `optimizer.target_lr`
- Alias Config Keys: -

## `power`
- Flags: `--power`
- Type/Action: type=float
- Default: `0.9`
- Primary Config Keys: `optimizer.power`
- Alias Config Keys: -

## `dataset_name`
- Flags: `--dataset_name`
- Type/Action: -
- Default: `'CUHK-PEDES'`
- Primary Config Keys: `dataset.dataset_name`
- Alias Config Keys: -
- Help: [CUHK-PEDES, ICFG-PEDES, RSTPReid]

## `root_dir`
- Flags: `--root_dir`
- Type/Action: -
- Default: `'data'`
- Primary Config Keys: `dataset.root_dir`
- Alias Config Keys: -

## `val_dataset`
- Flags: `--val_dataset`
- Type/Action: -
- Default: `'test'`
- Primary Config Keys: `dataset.val_dataset`
- Alias Config Keys: `training.val_dataset`

## `test_batch_size`
- Flags: `--test_batch_size`
- Type/Action: type=int
- Default: `512`
- Primary Config Keys: `evaluation.batch_size`
- Alias Config Keys: `evaluation.test_batch_size`

## `checkpoint`
- Flags: `--checkpoint_path`, `--checkpoint`
- Type/Action: -
- Default: `''`
- Primary Config Keys: `evaluation.checkpoint_path`
- Alias Config Keys: `evaluation.checkpoint`

## `device`
- Flags: `--device`
- Type/Action: -
- Default: `'cuda'`
- Primary Config Keys: `evaluation.device`
- Alias Config Keys: -

## `cross_domain_generalization`
- Flags: `--cross_domain_generalization`
- Type/Action: type=_str2bool, nargs='?'
- Default: `False`
- Primary Config Keys: `evaluation.cross_domain_generalization`
- Alias Config Keys: -

## `target_domain`
- Flags: `--target_domain`
- Type/Action: -
- Default: `'RSTPReid'`
- Primary Config Keys: `evaluation.target_domain`
- Alias Config Keys: -

## `retrieval_metrics`
- Flags: `--retrieval_metrics`
- Type/Action: nargs='+'
- Default: `list(DEFAULT_RETRIEVAL_METRICS)`
- Primary Config Keys: `evaluation.retrieval_metrics`
- Alias Config Keys: -

## `prototype_eval_image_chunk_size`
- Flags: `--prototype_eval_image_chunk_size`
- Type/Action: type=int
- Default: `32`
- Primary Config Keys: `evaluation.prototype_image_chunk_size`
- Alias Config Keys: -

## `prototype_eval_text_chunk_size`
- Flags: `--prototype_eval_text_chunk_size`
- Type/Action: type=int
- Default: `128`
- Primary Config Keys: `evaluation.prototype_text_chunk_size`
- Alias Config Keys: -

## `retrieval_scorer`
- Flags: `--retrieval_scorer`
- Type/Action: type=str
- Default: `'exact'`
- Primary Config Keys: `evaluation.retrieval_scorer`
- Alias Config Keys: -

## `use_wandb`
- Flags: `--use_wandb`
- Type/Action: type=_str2bool, nargs='?'
- Default: `False`
- Primary Config Keys: `logging.use_wandb`
- Alias Config Keys: -

## `wandb_project`
- Flags: `--wandb_project`
- Type/Action: -
- Default: `'PAS'`
- Primary Config Keys: `logging.project`
- Alias Config Keys: -

## `wandb_entity`
- Flags: `--wandb_entity`
- Type/Action: -
- Default: `None`
- Primary Config Keys: `logging.entity`
- Alias Config Keys: -

## `wandb_run_name`
- Flags: `--wandb_run_name`
- Type/Action: -
- Default: `None`
- Primary Config Keys: `logging.run_name`
- Alias Config Keys: -

## `nohup`
- Flags: `--nohup`
- Type/Action: type=_str2bool, nargs='?'
- Default: `False`
- Primary Config Keys: -
- Alias Config Keys: -

## `wandb_group`
- Flags: `--wandb_group`
- Type/Action: -
- Default: `None`
- Primary Config Keys: `logging.group`
- Alias Config Keys: -

## `wandb_mode`
- Flags: `--wandb_mode`
- Type/Action: -
- Default: `'online'`
- Primary Config Keys: `logging.mode`
- Alias Config Keys: -

## `wandb_tags`
- Flags: `--wandb_tags`
- Type/Action: nargs='*'
- Default: `[]`
- Primary Config Keys: `logging.tags`
- Alias Config Keys: -

## `wandb_notes`
- Flags: `--wandb_notes`
- Type/Action: -
- Default: `None`
- Primary Config Keys: `logging.notes`
- Alias Config Keys: -

## `wandb_log_interval`
- Flags: `--wandb_log_interval`
- Type/Action: type=int
- Default: `50`
- Primary Config Keys: `logging.log_interval`
- Alias Config Keys: -

## `wandb_log_code`
- Flags: `--wandb_log_code`
- Type/Action: type=_str2bool, nargs='?'
- Default: `False`
- Primary Config Keys: `logging.log_code`
- Alias Config Keys: -

## `log_debug_metrics`
- Flags: `--log_debug_metrics`
- Type/Action: type=_str2bool, nargs='?'
- Default: `True`
- Primary Config Keys: `logging.log_debug_metrics`
- Alias Config Keys: `training.log_debug_metrics`
