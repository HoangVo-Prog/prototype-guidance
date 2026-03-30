# PHASE_D_REPORT

## Files Modified
- `model/build.py`
- `model/prototype/head.py`
- `processor/processor.py`
- `utils/metrics.py`
- `utils/experiment.py`
- `utils/options.py`
- `utils/config.py`
- `configs/default.yaml`
- `configs/baseline_legacy.yaml`
- `configs/prototype_v1_draft.yaml`

## Files Added
- `tests/test_phase_d_integration.py`
- `scripts/phase_d_smoke.py`
- `PHASE_D_REPORT.md`

## Integration Strategy
- Preserved the legacy baseline implementation by extracting it into `ITSELF.baseline_forward(...)` and leaving its TAL/CID logic unchanged.
- Introduced `ITSELF.prototype_forward(...)` as the Phase D method path, driven by `model.use_prototype_bank` and reusing the Phase B/C encoder interfaces and Phase C prototype modules.
- Kept the integration additive and localized: the backbone encoders are still the single source of image/text states, while prototype routing, pooling, projection, and losses live inside `model/prototype`.
- Avoided training-loop rewrites by returning a compatible loss dictionary and teaching the existing loop to use `loss_total` when present.

## New Forward Path Summary
- `ITSELF.forward(...)` now dispatches between `baseline_forward(...)` and `prototype_forward(...)`.
- The prototype path encodes image globals and text token states through the existing encoder wrappers, runs prototype contextualization and routing, forms summary `Q`, computes token scores and masked pooling, projects image/text features, and returns the Phase C loss breakdown.
- Debug outputs remain optional and are controlled by `model.return_debug_outputs`; when enabled, the returned debug dict now also exposes aliases such as `alpha`, `beta`, `Q`, `Theta_v`, and `Theta_tilde`.

## Baseline Preservation
- Baseline behavior remains on the original path whenever `model.use_prototype_bank: false`.
- Legacy `encode_image(...)`, `encode_text(...)`, TAL loss, CID loss, and GRAB retrieval behavior are unchanged on the baseline branch.
- The prototype branch does not alter baseline math; it is activated only by config.

## Logging Integration
- The training loop now prefers `loss_total` when present, which prevents double-counting prototype sub-losses.
- Meter/logging surfaces were extended to capture `loss_total`, `loss_infonce`, `loss_diversity`, `loss_balance`, and prototype debug scalars.
- W&B/debug logging continues to be optional and tolerant of missing metrics.
- Large prototype tensors are explicitly excluded from scalar logging through `utils/experiment.py` skip lists.

## Evaluation Compatibility
- The public evaluation entrypoint (`test.py` and `processor.do_inference(...)`) is unchanged.
- `Evaluator` now detects prototype retrieval mode automatically and switches to a blockwise pairwise similarity path built on top of `ITSELF.encode_image_for_retrieval(...)`, `ITSELF.encode_text_for_retrieval(...)`, and `ITSELF.compute_retrieval_similarity(...)`.
- Baseline evaluation still uses independently encoded image/text embeddings and the existing optional GRAB fusion table.
- Prototype evaluation is compatible with cached encoder outputs inside a single evaluation run, but it cannot use the old independent text-embedding cache because text pooling is image-conditioned.

## Smoke and Tiny-Overfit Coverage
- Added `scripts/phase_d_smoke.py`, a synthetic dummy-backbone harness that exercises one prototype forward pass, a short optimizer loop, and evaluator execution end-to-end.
- Added `tests/test_phase_d_integration.py`, which patches in the same dummy backbone to test:
  - baseline dispatch preservation
  - prototype dispatch and debug outputs
  - frozen-backbone policy
  - evaluator compatibility for prototype retrieval
  - a tiny repeated-batch overfit signal

## Known Issues and Limitations
- Prototype retrieval uses blockwise pairwise pooling at evaluation time, so it is slower than the baseline independent-embedding path.
- Phase D keeps dense routing only; sparse routing/top-k assignment is still intentionally disabled.
- The prototype path assumes `use_image_conditioned_pooling: true` whenever `use_prototype_bank: true` to avoid silently activating a partial ablation path.
- The synthetic smoke harness validates integration mechanics, not dataset realism or final retrieval quality.

## Phase A/B/C Repairs Carried Forward
- Promoted the prototype branch from an additive side path into the primary model path when `use_prototype_bank` is enabled.
- Added explicit prototype evaluation chunk controls to the YAML/CLI config surface.
- Updated the Phase D draft config to freeze both backbones by default and to use `loss_names: prototype`.

## Next Steps for Phase E
- Tune the prototype retrieval chunking strategy for larger validation sets.
- Add real dataset-level regression checks comparing baseline checkpoints before and after Phase D integration.
- Decide whether the prototype branch should support optional joint training with the legacy TAL/CID objectives.
- Add checkpoint-selection logic based on prototype metrics if prototype training becomes the default experiment path.
