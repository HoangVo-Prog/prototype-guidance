# Prototype Semantic-Structure Method (Additive Path)

This repo now supports two prototype method roles:

- `retrieval_branch` (legacy): prototype branch behaves as retrieval-side scorer.
- `semantic_structure` (new): prototype branch is a semantic-structure regularizer for host/surrogate geometry.

Runtime architecture is unchanged: `HostCore -> PrototypePlugin -> Composer` remains intact.

## What Changes In Semantic-Structure Mode
- Prototype anchors can come from recomputed detached embedding clusters (`prototype.bank_source: recomputed_kmeans`).
- Base anchors and contextualized anchors are separated.
- Diagonal fidelity remains active and central (`loss_diag`).
- New semantic loss is available: `loss_semantic_pbt` (ProtoCLIP-inspired soft PBT regularization).
- Prototype retrieval scoring path is preserved but can be disabled by config.
- Inference can be host-authoritative by default via `model.prototype_inference_mode: host_only`.

## Legacy Retrieval Branch (Unchanged Semantics)
```yaml
model:
  prototype_method_role: retrieval_branch
  prototype_semantic_enabled: false
  prototype_recompute_enabled: false
  prototype_inference_mode: legacy_fused

prototype:
  bank_source: learnable_legacy
  contextualization_enabled: true
  contextualization_residual: true

semantic_structure:
  enabled: false

objectives:
  objectives:
    use_loss_ret: true
    use_loss_diag: true
    use_loss_support: true
    use_loss_semantic_pbt: false
  lambda:
    ret: 1.0
    diag: 1.0
    support: 0.1
    semantic_pbt: 0.0
```

## New Semantic-Structure Mode (Recommended Starting Point)
```yaml
model:
  prototype_method_role: semantic_structure
  prototype_semantic_enabled: true
  prototype_recompute_enabled: true
  prototype_inference_mode: host_only

prototype:
  bank_source: recomputed_kmeans
  contextualization_enabled: true
  contextualization_residual: true
  use_base_for_semantic_targets: true

semantic_structure:
  enabled: true
  feature_space: prototype_projected
  pbt_enabled: true
  soft_target_enabled: true
  target_temperature: 0.01
  pred_temperature: 0.07
  recompute_schedule: epoch
  recompute_interval: 1
  min_cluster_count_for_pbt: 1.0
  empty_cluster_policy: skip
  text_teacher_source: exact_diagonal
  text_student_source: surrogate_diagonal
  image_student_source: image_semantic_feature
  recompute_start_epoch: 0
  recompute_start_step: 0
  loss_ramp_start_epoch: 0
  loss_ramp_start_step: 0
  loss_ramp_epochs: 0
  loss_ramp_steps: 0

objectives:
  objectives:
    use_loss_ret: false
    use_loss_diag: true
    use_loss_support: false
    use_loss_semantic_pbt: true
  lambda:
    ret: 0.0
    diag: 1.0
    support: 0.0
    semantic_pbt: 1.0
```

## Backward Compatibility
- Legacy losses remain available: `loss_ret`, `loss_support`, `loss_diversity`, `loss_balance`.
- Legacy retrieval scoring/fusion paths remain available.
- Existing runtime modes remain available.

## Validation
No training/evaluation was run automatically. Use your own runs to validate behavior and metrics.
