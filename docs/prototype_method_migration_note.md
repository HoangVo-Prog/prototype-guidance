# Migration Note: Prototype Retrieval Branch -> Semantic Structure

## Why legacy losses still exist
Legacy losses and retrieval logic are intentionally retained for:
- backward-compatible checkpoint/config replay,
- side-by-side ablations,
- staged migration from retrieval-branch behavior.

## Minimal migration checklist
1. Set `model.prototype_method_role: semantic_structure`.
2. Enable semantic block (`semantic_structure.enabled: true`).
3. Switch prototype source to recompute (`prototype.bank_source: recomputed_kmeans`).
4. Enable semantic PBT (`objectives.objectives.use_loss_semantic_pbt: true`, `objectives.lambda.semantic_pbt > 0`).
5. Disable legacy retrieval-side prototype loss by default (`use_loss_ret: false`, `ret: 0.0`).
6. Keep diagonal fidelity on (`use_loss_diag: true`, `diag > 0`).
7. Use host-authoritative retrieval inference (`model.prototype_inference_mode: host_only`).

## Diagnostic signals to watch
- `semantic_recompute_count`
- `semantic_active_cluster_count`
- `semantic_empty_cluster_count`
- `semantic_assignment_entropy_image`
- `semantic_assignment_entropy_teacher`
- `semantic_pbt_valid_cluster_count`
- `loss_semantic_pbt`
- `loss_diag`
- `loss_ret` (should stay near 0 if disabled)

## Validation ownership
No automated validation runs were executed in this implementation pass.
