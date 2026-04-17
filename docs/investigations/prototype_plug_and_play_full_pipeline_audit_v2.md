# Prototype Plug-and-Play Full Pipeline Audit V2 (Current System, Claim-Hardened)

## 1. Executive summary
- The current implementation is a **single coupled PAS graph**, not an attachable external plugin: `PASModel` builds `base_model` + `host_head` + `prototype_head` + `fusion_module` in one module (`model/pas_model.py:41-118`).
- Host and prototype consume shared encoder outputs in the same forward pass, with no detach boundary before prototype inputs (`model/pas_model.py:1079-1124`).
- Training uses one scalar objective, `loss_total = lambda_host * host_loss + prototype_loss`, so branch objectives are optimization-coupled by construction (`model/pas_model.py:1135`).
- Phase behavior is mutable at runtime through `training.freeze_schedule` (trainability, loss switches, optimizer/scheduler rebuild), so stage semantics are not fixed by architecture (`processor/processor.py:270-295`; `utils/freeze_schedule.py:301-391`).
- Inference fusion is direct weighted addition, not calibration-only isolation (`model/fusion.py:32-56`).
- Evaluator selection logic can source checkpoint metric from subset rows and still display `pas-t2i`, which weakens mode-reporting stability (`utils/metrics.py:565-610`).
- Current "prototype-only" evaluation row is **not a cleanly isolated runtime path**: PAS component computation still builds host similarity first (`model/pas_model.py:888-906`).
- The audit proves structural coupling and mutable semantics. It does **not** quantify each mechanism's effect size without controlled experiments.

## 2. Full architecture map (current code)

### 2.1 Build-time routing
- Entry: `build_model(args, num_classes)` (`model/build.py:61-64`).
- Routing: `_should_use_pas_model` checks `use_prototype_branch` / prototype flags (`model/build.py:14-21`).
- If enabled: build PAS runtime (`model/build.py:33-38`).
- If disabled: build host-only runtime (`model/build.py:24-30,38`).

### 2.2 PAS construction (where coupling starts)
- Shared CLIP backbone: `self.base_model` (`model/pas_model.py:47`).
- Host module: `self.host_head` (`model/pas_model.py:95-99`).
- Prototype module: `self.prototype_head` (`model/pas_model.py:100-112`).
- Fusion module: `self.fusion_module` (`model/pas_model.py:113-118`).

### 2.3 Major pipeline modules
- Host adapter: `CLIPHostAdapter` wraps `VanillaCLIPHead` (`model/host_heads.py:251-255`).
- Host outputs include `surrogate_pairwise_logits` (`model/vanilla_clip.py:315-333`).
- Prototype head receives image/text shared tensors and optional host logits (`model/prototype/head.py:1007-1111`).
- Fusion computes weighted sum of host/prototype similarities (`model/fusion.py:41-56`).

### 2.4 Checkpoint/load path
- Modular load at train start: `load_configured_groups` (`train.py:299-306`; `utils/modular_checkpoint.py:379-483`).
- Modular save latest/best uses selected validation metric row (`processor/processor.py:479-492`; `utils/modular_checkpoint.py:294-357`).

## 3. Full training pipeline (end-to-end)

### 3.1 Config -> args
- Config flattening maps nested keys (model/prototype/fusion/training) to runtime args (`utils/config.py:38-226`).
- `_finalize_args` derives prototype/fusion defaults and imports `freeze_schedule` (`utils/options.py:303-359,323-324`).

### 3.2 Model build and optional load
- `train.py` builds model and optionally loads modular groups (`train.py:296-306`).

### 3.3 Freeze policy and runtime phase mutation
- Static freeze flags applied in model init (`freeze_image_backbone`, `freeze_text_backbone`, `freeze_host_projectors`, `freeze_prototype_side`) (`model/pas_model.py:327-349`).
- Epoch loop parses and activates `training.freeze_schedule` (`processor/processor.py:270-287`).
- On phase switch:
1. `apply_phase_trainability` toggles `requires_grad` by group (`utils/freeze_schedule.py:301-307`; `utils/module_group_registry.py:8-44`).
2. `apply_loss_weight_overrides` mutates `lambda_host`, prototype lambdas, and loss enable switches (`utils/freeze_schedule.py:310-391`).
3. Optimizer is rebuilt and prior state is copied (`processor/processor.py:289-292`).
4. Scheduler is rebuilt/rewound (`processor/processor.py:294-295`).

### 3.4 Forward/loss/backward
- Shared feature extraction: `extract_image_features`, `extract_text_features` (`model/pas_model.py:1079-1080`).
- Host forward first (`model/pas_model.py:1083-1091`).
- Prototype forward consumes shared features and optional host logits (`model/pas_model.py:1120-1129`).
- Total loss is joint scalar sum (`model/pas_model.py:1135`).
- Backprop on that scalar in trainer loop (`processor/processor.py:365-387`).

### 3.5 Stage naming vs actual control
- `training_stage` is parsed/validated (`model/pas_model.py:84-85,309-316`; `utils/config.py:490`).
- Epoch behavior is actually controlled by `freeze_schedule` phase activation (`processor/processor.py:283-295`), not by a stage-specific branch router in trainer.

## 4. Full usage modes currently supported (as implemented)

### 4.1 Host-only runtime mode
- Achieved by disabling prototype branch in build routing (`model/build.py:14-38`).

### 4.2 Prototype-enabled training mode
- Uses PAS monolith with schedule-driven trainability/loss mutation (`model/pas_model.py:41-118`; `processor/processor.py:270-295`).

### 4.3 Prototype-enabled inference mode
- Exact scorer path: `compute_retrieval_similarity_components` (`utils/metrics.py:525-536`; `model/pas_model.py:888-906`).
- Approximate scorer path: `compute_approximate_retrieval_similarity_components` (`utils/metrics.py:509-523`; `model/pas_model.py:915-936`).

### 4.4 Row-level evaluator modes
- Evaluator emits `pas-t2i`, `host-t2i`, optional `prototype-t2i`, subset rows (`utils/metrics.py:399-461`).
- This is row-level reporting flexibility, not proof of execution-independent architecture.

## 5. Full inference pipeline (current)

### 5.1 Host-only row
- Built by fusing with `(lambda_host=1.0, lambda_prototype=0.0)` in sweep builder (`utils/metrics.py:413-415,443-449`).

### 5.2 Prototype-only row (current meaning)
- Built by evaluator sweep as `(0.0, 1.0)` when prototype similarity exists (`utils/metrics.py:416-418,443-449`).
- Not execution-independent: PAS component computation still computes host similarity first (`model/pas_model.py:893-894`).

### 5.3 Fused row
- Similarity is direct weighted sum (`model/fusion.py:41-56`; `utils/metrics.py:370-397`).

### 5.4 Scoring/ranking/selection
- Text/image features encoded, concatenated, then component similarities computed (`utils/metrics.py:472-538`).
- Ranking metrics computed per row (`utils/metrics.py:556-561`).
- Selection may rewrite `pas-t2i` metrics from subset-best row under specific config condition (`utils/metrics.py:194-203,565-580,607-610`).

## 6. Coupling analysis (current)

### 6.1 Representation coupling
- Prototype receives shared backbone-derived image/text tensors from the same forward (`model/pas_model.py:1121-1124`).
- No detach barrier at this boundary in current code (`model/pas_model.py:1121-1124`).
- Prototype image path mixes adapted image features with prototype summary projection (`model/prototype/head.py:667-670`).

### 6.2 Optimization coupling
- Joint scalar objective couples branch optimization (`model/pas_model.py:1135`).
- If host/backbone params are trainable in active phase, prototype-loss gradients can affect shared parameters through shared upstream tensors.

### 6.3 Loss coupling
- Prototype weighted retrieval can consume host logits (`host_pairwise_logits`) when enabled (`model/prototype/head.py:1107-1109`; `model/prototype/losses.py:523-531`).
- Host margin source can be detached or live based on `weight_ret_detach_host` (`model/prototype/losses.py:467`).

### 6.4 Fusion coupling
- Final score is additive sum of branch similarities (`model/fusion.py:54-56`).
- No separate learned calibration boundary is enforced in this path.

### 6.5 Checkpoint/stage coupling
- Checkpoint payload stores `training_stage` metadata (`utils/modular_checkpoint.py:229-233`), but runtime behavior is phase-driven.
- Modular loading allows mixed source composition by checkpoint groups (`utils/modular_checkpoint.py:379-483`).

### 6.6 Reporting/selection coupling
- Row selection excludes host-only by policy and can source from subset rows (`utils/metrics.py:565-603`).
- Displayed selected row can be normalized back to `pas-t2i` under subset-selection mode (`utils/metrics.py:607-610`).

### 6.7 Terminology contract (strict wording)
- **Not standalone** in this audit means: not a clean runtime-isolated execution path.  
  Example: current prototype row is constructed from component computation that still executes host similarity first (`model/pas_model.py:893-906`).
- **Not semantically independent** in this audit means: prototype behavior/objective or final decisions can still depend on host quantities.  
  Examples: optional host-logit-conditioned prototype loss path (`model/prototype/losses.py:523-531`) and direct host+prototype additive decision fusion (`model/fusion.py:54-56`).
- These are different claims: runtime non-isolation does not automatically prove semantic contamination magnitude; it proves the absence of a hard execution boundary.

## 7. Causal hierarchy of failure mechanisms

### 7.1 Structural root causes (architecture-level)
1. **Single-graph integration**: host and prototype are instantiated and executed inside one `PASModel` graph (`model/pas_model.py:41-118,1079-1131`).
2. **Joint optimization objective**: one scalar loss combines host and prototype losses (`model/pas_model.py:1135`).
3. **No enforced external boundary**: prototype input tensors are shared forward outputs with no detach gate in default path (`model/pas_model.py:1121-1124`).
4. **Decision-level entanglement**: fused score is direct additive combination (`model/fusion.py:54-56`).

### 7.2 Training amplifiers (make impact larger/less stable)
1. **Runtime phase mutation** of trainability, loss switches, and optimizer state (`processor/processor.py:283-295`; `utils/freeze_schedule.py:310-391`).
2. **Host/prototype loss weights can change per phase**, including host loss enable/disable (`utils/freeze_schedule.py:319-337`).
3. **Potential host-logit-conditioned prototype objective** when weighted retrieval is enabled (`model/prototype/losses.py:523-531`).
4. **Mixed-source modular checkpoint loading** can compose artifacts from different runs/groups (`utils/modular_checkpoint.py:379-483`).

### 7.3 Inference/reporting symptoms (observables, not deepest cause)
1. Fused ranking drift when prototype similarity is noisy or mis-scaled due direct additive fusion (`model/fusion.py:54-56`).
2. "Prototype-only" row exists but is not a clean execution-independent mode (`model/pas_model.py:888-906`; `utils/metrics.py:416-418`).
3. Selection/display row behavior can mask which row actually drove `val/top1` (`utils/metrics.py:565-610,623-624`).

## 8. Plug-and-play gap analysis (strict)

### Principle 1: Host preserves original behavior/performance when prototype added
- Status: **Not satisfied by architecture**.
- Evidence: shared graph + shared loss + shared tensors (`model/pas_model.py:1079-1135`).

### Principle 2: Prototype attach/remove without host relearning
- Status: **Not satisfied as a default contract**.
- Evidence: enabling prototype routes to PAS runtime path, not an external plugin boundary (`model/build.py:14-38`).

### Principle 3: Prototype must not silently change host dynamics
- Status: **Not satisfied robustly**.
- Evidence: schedule mutates trainability/loss flags/optimizer during training (`processor/processor.py:283-295`; `utils/freeze_schedule.py:310-391`).

### Principle 4: Unambiguous host/prototype/fused inference
- Status: **Partially satisfied at row-reporting level, not at execution isolation level**.
- Evidence: rows are emitted (`utils/metrics.py:399-461`), but prototype row is not a clean host-free execution path (`model/pas_model.py:893-906`).

### Principle 5: Explicit training semantics (external/partial/joint)
- Status: **Not explicit in architecture contracts**.
- Evidence: semantics emerge from combined flags + schedule, not a dedicated integration-mode contract (`utils/options.py:303-359`; `processor/processor.py:270-295`).

### Principle 6: Clear shared/frozen/trainable/calibration-only/inference-only boundaries
- Status: **Partially operational, not contract-enforced end-to-end**.
- Evidence: logical groups exist (`utils/module_group_registry.py:8-44`), but forward/loss graph remains jointly wired (`model/pas_model.py:1079-1135`).

## 9. Mode semantics are not stable enough yet
- Evaluator selection behavior changes depending on whether fusion lambdas are explicit in `config_data.fusion` (`utils/metrics.py:194-203`).
- Under subset-selection mode, checkpoint source row can be subset-best while displayed row is rewritten as `pas-t2i` (`utils/metrics.py:565-610,623-624`).
- `training_stage` is validated/recorded, but epoch-time optimization semantics are mainly schedule-driven (`model/pas_model.py:309-316`; `processor/processor.py:283-295`).
- Therefore, mode labels are not yet a stable architectural contract; they are partly policy/reporting behavior.

## 10. What the current code does NOT prove
- It does **not** prove the magnitude contribution of each coupling mechanism to host drop; that requires controlled ablations.
- It does **not** prove that host similarity computed first causes semantic contamination by itself; order alone is insufficient without gradient/usage evidence.
- It does **not** prove that evaluator row rewrite changes model semantics; it proves reporting/selection semantics can shift.
- It does **not** prove exact scorer is better/worse than approximate scorer for host safety; both paths exist (`utils/metrics.py:509-538`) but comparative outcomes are empirical.

## 11. File-by-file audit table (coupling relevance)
| File | Symbol(s) | Pipeline role | Why it matters for plug-and-play gap |
|---|---|---|---|
| `model/build.py` | `_should_use_pas_model`, `_build_model_impl` | Runtime path selection | Prototype attach/remove toggles runtime path (`:14-38`). |
| `model/pas_model.py` | `__init__` | Builds host+prototype+fusion together | Structural coupling starts here (`:41-118`). |
| `model/pas_model.py` | `forward` | Shared extraction + host/prototype + total loss | Representation + optimization coupling (`:1079-1135`). |
| `model/pas_model.py` | `compute_retrieval_similarity_components` | Exact component retrieval | Host computed first; prototype row not execution-isolated (`:888-906`). |
| `model/pas_model.py` | `named_optimizer_groups` | Param-group surface | Depends on `requires_grad`; phase mutation affects updates (`:1002-1017`). |
| `model/prototype/head.py` | `forward`, `encode_image_branch` | Prototype runtime path | Consumes shared tensors; image proxy mixes prototype summary (`:1007-1111`, `:667-670`). |
| `model/prototype/losses.py` | `surrogate_retrieval_loss`, `weighted_surrogate_retrieval_loss`, `forward` | Prototype objective | Optional host-logit coupling; stage objective composition (`:419-429`, `:445-482`, `:523-601`). |
| `model/fusion.py` | `ResidualScoreFusion.forward` | Final score combiner | Direct additive decision coupling (`:41-56`). |
| `processor/processor.py` | `do_train` phase loop | Runtime control | Trainability/loss/optimizer mutates by epoch phase (`:270-295`). |
| `utils/freeze_schedule.py` | `apply_phase_trainability`, `apply_loss_weight_overrides` | Phase execution | Switches `requires_grad` and loss behavior (`:301-391`). |
| `utils/module_group_registry.py` | `LOGICAL_MODULE_GROUP_PREFIXES` | Freeze/checkpoint ownership map | Defines practical coupling boundaries (`:8-44`). |
| `solver/build.py` | `build_optimizer` | Optimizer construction | Uses current named groups (already filtered by `requires_grad`) (`:172-205`). |
| `utils/metrics.py` | `_build_similarity_rows`, `eval`, `_should_select_from_eval_subsets` | Eval/fusion/selection | Mode rows, selection policy, display/source row divergence (`:194-203`, `:399-461`, `:565-624`). |
| `utils/modular_checkpoint.py` | `_build_payload`, `save_best_if_improved`, `load_configured_groups` | Save/load policy | Stage metadata + mixed group loading semantics (`:229-233`, `:327-357`, `:379-483`). |
| `configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml` | `freeze_schedule`, `fusion.eval_subsets` | Concrete training/eval policy | Shows warmup-only then joint phase and subset sweep configuration (`:84-96`, `:185-220`). |

## 12. Confirmed facts vs open uncertainties

### 12.1 Confirmed by code
- This is a coupled host+prototype training graph with joint loss (`model/pas_model.py:41-118,1135`).
- Shared representations are reused across both branches in forward (`model/pas_model.py:1079-1124`).
- Fusion is direct additive scoring (`model/fusion.py:54-56`).
- Runtime phase schedule can materially alter trainability and objectives (`processor/processor.py:283-295`; `utils/freeze_schedule.py:310-391`).
- Evaluator row selection policy can differ from displayed row source under subset mode (`utils/metrics.py:565-624`).

### 12.2 Supported by code but needs experiment
- Relative contribution of each coupling mechanism to host performance loss.
- Whether score-scale mismatch between host/prototype contributes materially in this dataset/run distribution.
- Whether approximate scorer exacerbates or dampens coupling symptoms versus exact scorer.

### 12.3 Still unclear from code alone
- Quantitative causal weights (root cause share vs amplifier share).
- Generalization of observed degradation across seeds/datasets/checkpoint combinations.

## 13. Why this audit is sufficient for redesign, but not sufficient for quantitative attribution
- Sufficient for redesign:
1. It identifies concrete structural boundaries where coupling is implemented (`PASModel.forward`, fusion, phase mutation).
2. It identifies exact control levers (`freeze_schedule`, loss-weight overrides, evaluator selection policy).
3. It identifies where contracts are missing (integration mode, metric authority, checkpoint-row policy).
- Not sufficient for quantitative attribution:
1. Code proves pathways, not effect sizes.
2. Relative harm from each pathway is a statistical property requiring controlled experiments.

## 14. Final audit conclusion (hard)
The current system is a **coupled co-training and fused-decision system with mutable stage semantics**. It is not an external plugin architecture and should not be described as true plug-and-play in its current form. The codebase proves structural optimization and decision coupling (`model/pas_model.py:1079-1135`; `model/fusion.py:54-56`) and proves policy-level metric/selection mutability (`utils/metrics.py:565-624`); therefore, only limited host-safe claims are defensible without additional architectural contracts and proof checks.
