# Prototype Plug-and-Play Full Pipeline Audit (Current System)

## 1. Executive summary
- The current prototype branch is not plug-and-play because it is built inside the same `PASModel` graph as the host (`model/pas_model.py:41-120`), consumes the same backbone outputs (`model/pas_model.py:1079-1124`), and is optimized through a shared scalar objective `loss_total = lambda_host * host_loss + prototype_loss` (`model/pas_model.py:1135`).
- The strongest coupling is optimization + representation coupling: prototype losses backpropagate through shared backbone outputs whenever host/backbone groups are trainable (`model/pas_model.py:1121-1124,1135`; `utils/freeze_schedule.py:301-307`).
- Inference fusion is a direct weighted sum, not calibration-only (`model/fusion.py:32-56`), so prototype score errors directly perturb final ranking.
- Stage behavior is controlled by `training.freeze_schedule` (trainability, loss weights, LR overrides, optimizer rebuild), not by `training_stage` routing (`processor/processor.py:270-295`; `model/pas_model.py:309-316`).
- Evaluation can overwrite `pas-t2i` metrics with best subset row when fusion lambdas are omitted and `fusion.eval_subsets` exists (`utils/metrics.py:194-203,565-580`), coupling reporting/selection to sweep policy.
- Host-only/prototype-only/fused rows exist in evaluator sweeps (`utils/metrics.py:413-461`), but retrieval component computation still always builds host similarity first (`model/pas_model.py:888-906`).

## 2. Full architecture map
### 2.1 Model construction path
- Config parsing and flattening map YAML sections to runtime args in `utils/config.py` (`PRIMARY_CONFIG_KEY_MAP`, including `model.*`, `prototype.*`, `fusion.*`, `training.freeze_schedule`) (`utils/config.py:12-221`).
- CLI + config are merged in `get_args()` and `_finalize_args()` (`utils/options.py:423-440,290-420`).
- `build_model()` chooses runtime:
  - PAS runtime when prototype branch is enabled (`model/build.py:_should_use_pas_model`, `_build_model_impl`) (`model/build.py:14-38`).
  - Host-only runtime otherwise (`model/build.py:24-30,38`).
- PAS runtime constructs one graph:
  - Shared CLIP backbone: `self.base_model` (`model/pas_model.py:47`).
  - Host head: `self.host_head` (`model/pas_model.py:95-99`).
  - Prototype head: `self.prototype_head` via `build_prototype_head(...)` (`model/pas_model.py:100-112`; `model/prototype/build.py:10-203`).
  - Fusion module: `self.fusion_module` (`model/pas_model.py:113-118`; `model/fusion.py:7-21`).

### 2.2 Module-level map (current)
```text
train.py -> model/build.py
  -> PASModel
     -> base_model (shared CLIP image/text encoders)
     -> host_head (CLIPHostAdapter or ITSELFHostHead)
     -> prototype_head (PrototypeConditionedTextHead or DirectImageConditionedTextHead)
     -> fusion_module (ResidualScoreFusion)

forward(batch):
  shared features = extract_image_features + extract_text_features
  host_outputs = host_head(shared features)
  prototype_outputs = prototype_head(shared features, optional host logits)
  total_loss = lambda_host * host_loss + prototype_loss
```
Evidence: `model/pas_model.py:730-800,1057-1136`; `model/host_heads.py:251-292,867-887`; `model/prototype/head.py:1007-1111`; `model/fusion.py:32-56`.

### 2.3 Host path
- CLIP host adapter wraps `VanillaCLIPHead` (`model/host_heads.py:251-255`).
- Host retrieval logits come from `VanillaCLIPHead` (`surrogate_pairwise_logits`) (`model/vanilla_clip.py:315-333`).
- Host similarity at retrieval uses `host_head.compute_similarity_matrix(...)` (`model/pas_model.py:865-869`).

### 2.4 Prototype path
- Prototype head is built with routing, token scoring/pooling, projector, and losses (`model/prototype/head.py:136-252`).
- Prototype consumes shared host/backbone-derived tensors: image global/local, text token states, masks, and optional host logits (`model/pas_model.py:1121-1129`; `model/prototype/head.py:1007-1111`).
- Prototype image path injects routed summary into image proxy features: `image_proxy_features = image_features + proto_query_proj(summary)` (`model/prototype/head.py:667-670`).

### 2.5 Fusion path
- Fusion combines branch similarities as linear weighted sum (`model/fusion.py:54-56`).
- PAS inference calls this via `fuse_retrieval_similarity(...)` (`model/pas_model.py:871-886`).

### 2.6 Training control path
- Epoch phase control is parsed from `training.freeze_schedule` (`utils/freeze_schedule.py:145-251`).
- At phase changes, runtime mutates `requires_grad`, loss weights, optimizer, and scheduler (`processor/processor.py:283-295`; `utils/freeze_schedule.py:301-406`).

### 2.7 Evaluation path
- Evaluator encodes text/image features, computes similarity components (exact or approximate), builds sweep rows, and selects `val/top1` row (`utils/metrics.py:463-647`).

### 2.8 Checkpoint/load path
- Training can load modular groups (`host`, `prototype_bank`, `prototype_projector`, `fusion`) via `ModularCheckpointManager` (`train.py:299-306`; `utils/modular_checkpoint.py:165-483`).
- Validation saves latest/best per group with metric row metadata (`processor/processor.py:476-492`; `utils/modular_checkpoint.py:294-357`).

## 3. Full training pipeline
### 3.1 Config load -> args finalization
1. `get_args()` loads YAML config(s), applies aliases, validates schema/runtime constraints (`utils/options.py:423-440`; `utils/config.py:832-999`).
2. `_finalize_args()` derives runtime flags:
   - `use_prototype_branch/use_prototype_bank/use_image_conditioned_pooling` (`utils/options.py:303-316`).
   - fusion defaults and legacy coefficient handling (`utils/options.py:334-359`).
   - `freeze_schedule` copied from config into args (`utils/options.py:320-325`).

### 3.2 Model build -> optional checkpoint load
1. `train.py` builds dataloaders and model (`train.py:296-299`).
2. If modular loading enabled, loads configured groups into model (`train.py:299-306`; `utils/modular_checkpoint.py:379-483`).
3. Else optional deprecated full finetune checkpoint path (`train.py:306-363`).

### 3.3 Optimizer build
- Optimizer is built from `model.named_optimizer_groups()` and includes only parameters with `requires_grad=True` because PAS group builder explicitly skips frozen params (`model/pas_model.py:1015-1017`; `solver/build.py:179-205`).

### 3.4 Freeze schedule application (epoch boundary)
- For each epoch, active phase resolved by epoch range (`processor/processor.py:282-284`; `utils/freeze_schedule.py:254-258`).
- On phase change:
  - apply trainability (`requires_grad`) by logical group prefixes (`processor/processor.py:285-287`; `utils/freeze_schedule.py:301-307`; `utils/module_group_registry.py:8-44`).
  - apply runtime loss-weight overrides, including enabling/disabling host/prototype loss switches (`processor/processor.py:287`; `utils/freeze_schedule.py:310-391`).
  - rebuild optimizer and copy prior optimizer state where params persist (`processor/processor.py:289-292`; `processor/processor.py:178-214`).
  - rebuild scheduler and rewind to current epoch (`processor/processor.py:294-295`).

### 3.5 Forward -> loss -> backward
1. `outputs = model(batch, current_step=...)` (`processor/processor.py:365-367`).
2. PAS forward:
   - shared features (`extract_image_features`, `extract_text_features`) (`model/pas_model.py:1079-1080`).
   - host forward (`model/pas_model.py:1083-1091`).
   - prototype forward (if enabled) with shared features and optional host logits (`model/pas_model.py:1120-1131`).
   - total loss composition (`model/pas_model.py:1133-1136`).
3. backprop on `loss_total` (`processor/processor.py:367-387`).

### 3.6 Evaluation hook
- Every `eval_period`, evaluator computes similarity rows and selected top1; modular checkpoints save latest/best by selected row (`processor/processor.py:458-492`; `utils/metrics.py:556-647`; `utils/modular_checkpoint.py:294-357`).

### 3.7 Stage behavior (actual)
- `training_stage` is validated in model (`model/pas_model.py:309-316`) and persisted in checkpoint metadata (`utils/modular_checkpoint.py:232`), but epoch behavior in training loop is driven by `freeze_schedule` (`processor/processor.py:270-295`).
- Example active config (`direction1_optionA_adaptive_prototype.yaml`):
  - Warmup phase freezes host groups and sets `lambda_host=0.0` (`configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml:187-201`).
  - Joint phase unfreezes host+prototype groups with `lambda_host=0.1`, high prototype weights (`:203-220`).

## 4. Full usage modes currently supported
### 4.1 Host-only
- Runtime path: `model/build.py` routes to host-only when prototype branch disabled (`model/build.py:24-38`).
- `host.type=clip`: `ClipHostModel` enforces no prototype flags (`model/hosts/clip.py:209-218`).
- `host.type=itself`: adapter to original ITSELF runtime when `use_prototype_branch=false` (`model/hosts/itself.py:38-42,316-320`).

### 4.2 Prototype-enabled training
- PAS runtime used when prototype branch enabled (`model/build.py:14-38`).
- Joint/phase-specific optimization controlled by freeze schedule and runtime loss overrides (`processor/processor.py:270-295`; `utils/freeze_schedule.py:310-391`).

### 4.3 Prototype-enabled inference
- Exact scorer path uses full pairwise pooling route (`utils/metrics.py:525-536`; `model/pas_model.py:888-906`; `model/prototype/head.py:894-955`).
- Approximate scorer path uses basis bank + routing route (`utils/metrics.py:509-523`; `model/pas_model.py:915-936`; `model/prototype/head.py:850-892`).
- If approximate requested without active prototype bank, evaluator falls back to exact (`utils/metrics.py:90-94`).

### 4.4 Exact vs approximate scorer
- Controlled by `evaluation.retrieval_scorer` (`utils/options.py:271`; `utils/metrics.py:84-95`).
- Approximate requires `encode_text_basis_for_retrieval` and `compute_approximate_retrieval_similarity_components` support (`utils/metrics.py:476-523`).

### 4.5 Eval subset and fusion sweep behavior
- Evaluator always emits `pas-t2i` plus sweep rows (`host-t2i`, optional `prototype-t2i`, configured subsets, default pair) (`utils/metrics.py:399-461`).
- When fusion lambdas are omitted in config but `fusion.eval_subsets` exists, checkpoint/current-R1 selection is taken from subset rows and can overwrite `pas-t2i` displayed metrics (`utils/metrics.py:194-203,565-580,607-610`).

### 4.6 Config mode switches that alter coupling behavior
- `model.use_prototype_branch`, `model.use_prototype_bank`, `model.use_image_conditioned_pooling` (`utils/options.py:303-316`).
- `evaluation.retrieval_scorer` exact/approximate (`utils/options.py:271`; `utils/metrics.py:84-95,509-538`).
- `training.freeze_schedule` (`utils/options.py:323-324`; `processor/processor.py:270-295`).
- `fusion.lambda_host/lambda_prototype` and `fusion.eval_subsets` (`utils/options.py:337-359`; `utils/metrics.py:137-163,399-461`).

## 5. Full inference pipeline
### 5.1 Host-only retrieval path
1. Encode text/image retrieval features (`utils/metrics.py:472-489`).
2. Compute host similarity via `host_head.compute_similarity_matrix` (`model/pas_model.py:865-869`).
3. Fuse with `lambda_prototype=0` returns host-only score (`model/fusion.py:46-53`).
4. Rank and compute retrieval metrics (`utils/metrics.py:17-66,556-561`).

### 5.2 Prototype-only retrieval path (current implementation)
- Evaluator can emit `prototype-t2i` row by setting `(lambda_host=0, lambda_prototype=1)` when prototype similarity exists (`utils/metrics.py:416-418,430-450`).
- But host similarity is still computed first in PAS component API (`model/pas_model.py:888-894`), so prototype-only row is not an independent host-free execution path.

### 5.3 Fused retrieval path
1. Compute host and prototype component similarities (`utils/metrics.py:525-536` or `509-521`).
2. Fuse with linear weighted sum (`utils/metrics.py:370-397`; `model/fusion.py:54-56`).
3. Build sweep rows and select row for reporting/checkpointing (`utils/metrics.py:399-461,565-647`).

### 5.4 Score construction, normalization, temperature
- Host similarity uses host logit scale in `VanillaClipLosses.compute_logits_i2t` (`model/vanilla_clip.py:69-75`).
- Prototype similarity uses prototype logit scale in `PrototypeLosses.compute_similarity_matrix/compute_paired_similarity` (`model/prototype/losses.py:146-156`).
- Fusion combines already-scaled branch scores linearly (`model/fusion.py:54-56`).

### 5.5 Ranking/metric path
- `rank()` sorts similarity rows and computes CMC/mAP/mINP (`utils/metrics.py:17-42`).
- `get_metrics()` packages per-row metrics (`utils/metrics.py:45-66`).

## 6. Coupling analysis
### 6.1 Representation coupling
- Shared image/text backbone outputs feed both host and prototype branches in same forward (`model/pas_model.py:1079-1124`).
- No detach boundary before prototype inputs (`model/pas_model.py:1121-1124`).
- Prototype image proxy explicitly mixes adapted image features with routed summary (`model/prototype/head.py:667-670`).

### 6.2 Optimization coupling
- Single scalar loss combines host and prototype objectives (`model/pas_model.py:1135`).
- When host groups are trainable, prototype-loss gradients can shape host/backbone parameters due shared upstream features (`model/pas_model.py:1121-1124,1135`; `utils/freeze_schedule.py:301-307`).

### 6.3 Loss coupling
- Runtime can route host logits into prototype weighted retrieval (`host_pairwise_logits` -> `weighted_surrogate_retrieval_loss`) when enabled (`model/pas_model.py:1128`; `model/prototype/losses.py:523-558`).
- Phase loss overrides mutate host/prototype loss switches in-place (`utils/freeze_schedule.py:319-390`).

### 6.4 Fusion coupling
- Final ranking score is direct weighted sum of host/prototype similarities (`model/fusion.py:54-56`).
- No learned/validated calibration module in forward path; weights are fixed inputs (`model/fusion.py:23-30`).

### 6.5 Checkpoint/stage coupling
- `training_stage` naming is mostly metadata/validation (`model/pas_model.py:309-316`; `utils/modular_checkpoint.py:232`), while real stage transitions are `freeze_schedule`-driven runtime mutations (`processor/processor.py:283-295`).
- Modular loading allows mixed host/prototype artifacts from different sources (`utils/modular_checkpoint.py:357-483`).

### 6.6 Calibration coupling
- Branch scales originate from different loss modules (`model/vanilla_clip.py:71-75`; `model/prototype/losses.py:149-156`) but fused directly (`model/fusion.py:54-56`).

### 6.7 Reporting/selection coupling
- Selected checkpoint metric row can come from subset sweeps and overwrite displayed `pas-t2i` row when no explicit fusion lambdas exist (`utils/metrics.py:194-203,565-580,607-610`).

## 7. Plug-and-play gap analysis
| Plug-and-play principle | Current status | Code evidence of pass/fail |
|---|---|---|
| 1. Host preserves original behavior/performance when prototype added | **Fail (by design coupling)** | Shared objective + shared representations (`model/pas_model.py:1121-1124,1135`). |
| 2. Prototype attach/remove without host relearning | **Fail** | Prototype-enabled path uses PAS graph; attach changes model path and objective (`model/build.py:14-38`; `model/pas_model.py:41-120,1135`). |
| 3. Prototype must not silently change host training dynamics | **Fail** | `freeze_schedule` and loss overrides mutate trainability and host-loss switches at runtime (`processor/processor.py:283-295`; `utils/freeze_schedule.py:319-337`). |
| 4. Inference supports host-only/prototype-only/fused unambiguously | **Partial** | Rows exist in evaluator (`utils/metrics.py:413-461`), but prototype-only still depends on computed host components and selection logic can rewrite `pas-t2i` (`model/pas_model.py:888-894`; `utils/metrics.py:565-580`). |
| 5. Training pipeline clearly states external/partial/joint semantics | **Fail** | No explicit integration mode; semantics are emergent from schedule + flags (`utils/options.py:303-324`; `processor/processor.py:270-295`). |
| 6. Architecture clearly separates shared/frozen/trainable/calibration-only/inference-only | **Partial/Fail** | Group prefixes exist (`utils/module_group_registry.py:8-44`) but forward/loss remain jointly wired (`model/pas_model.py:1057-1136`). |

## 8. File-by-file audit table
| File | Symbol(s) | Pipeline role | Coupling relevance |
|---|---|---|---|
| `model/build.py` | `_should_use_pas_model`, `_build_model_impl` | Selects PAS vs host-only runtime | Attach/remove prototype changes runtime graph (`:14-38`). |
| `model/pas_model.py` | `PASModel.__init__` | Builds shared backbone + host + prototype + fusion | Core attachment point (`:47-118`). |
| `model/pas_model.py` | `forward` | Shared feature extraction, host/prototype forward, total loss | Joint optimization coupling (`:1079-1136`). |
| `model/pas_model.py` | `compute_retrieval_similarity_components` | Exact host/prototype component similarity | Host computed first; prototype not standalone (`:888-906`). |
| `model/pas_model.py` | `compute_approximate_retrieval_similarity_components` | Approximate scorer components | Uses routing+basis plus host (`:915-936`). |
| `model/pas_model.py` | `named_optimizer_groups` | Groups trainable params | Freeze/optimizer control surface (`:1002-1055`). |
| `model/prototype/build.py` | `build_prototype_head` | Wires prototype head + loss/routing knobs | Config-level coupling switches and host-logit coupling flags (`:18-203`). |
| `model/prototype/head.py` | `forward`, `encode_image_branch`, `build_text_basis_bank` | Prototype representation and scoring path | Shared input dependency + image proxy injection (`:639-670,783-889,1007-1111`). |
| `model/prototype/losses.py` | `forward`, `surrogate_retrieval_loss`, `weighted_surrogate_retrieval_loss` | Prototype objective composition | Loss coupling + host-conditioned weighting (`:419-482,496-741`). |
| `model/host_heads.py` | `CLIPHostAdapter.forward` | Host branch outputs including pairwise logits | Provides host logits consumed by prototype loss path (`:280-292`). |
| `model/vanilla_clip.py` | `VanillaClipLosses` | Host scaling and retrieval loss | Separate branch scaling before fusion (`:69-75,123-202`). |
| `model/fusion.py` | `ResidualScoreFusion.forward` | Final score fusion | Linear additive coupling (`:32-56`). |
| `processor/processor.py` | `do_train` | Phase control, backward, eval hooks | Runtime mutation of trainability/loss/optimizer (`:270-295,365-387,458-492`). |
| `utils/freeze_schedule.py` | `apply_phase_trainability`, `apply_loss_weight_overrides` | Mutates requires_grad + loss switches | Practical stage semantics and host/prototype loss control (`:301-391`). |
| `solver/build.py` | `build_optimizer` | Builds grouped optimizer from trainable params | Confirms frozen params excluded from updates (`:179-205`). |
| `utils/metrics.py` | `_compute_similarity`, `_build_similarity_rows`, `eval` | Similarity computation, fusion sweep, selection row | Inference/reporting coupling and row overwrite behavior (`:399-647`). |
| `utils/options.py` | `_finalize_args` | Resolves mode/fusion/freeze args | Default fusion and mode semantics (`:303-359`). |
| `utils/modular_checkpoint.py` | `load_configured_groups`, `save_best_if_improved` | Group-wise load/save | Mixed artifact composition and metric-row metadata (`:327-357,379-483`). |
| `configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml` | `freeze_schedule`, `fusion.eval_subsets` | Run-level stage and eval policy | Concrete coupling policy in active recipe (`:185-220`, `:84-96`). |

## 9. Confirmed facts vs open uncertainties
### 9.1 Confirmed by code
- Prototype integration is in-graph and joint-loss optimized (`model/pas_model.py:41-120,1135`).
- Shared representations feed both branches without detach barrier (`model/pas_model.py:1079-1124`).
- Fusion is linear direct score addition (`model/fusion.py:54-56`).
- Stage behavior is schedule-driven runtime mutation, not stage-label routing (`processor/processor.py:270-295`; `model/pas_model.py:309-316`).
- Proxy losses are hard-disabled in current prototype forward path (`model/prototype/losses.py:534-553`); `loss_align` is hard zero there (`:570`).

### 9.2 Supported by code but needs experiment
- Host degradation magnitude from prototype-weighted phases depends on schedule weights (`configs/head_type/clip/direction1_optionA_adaptive_prototype.yaml:217-220`).
- Scale mismatch impact on fusion quality needs empirical calibration ablation (branch-specific logit scales + direct sum) (`model/vanilla_clip.py:71-75`; `model/prototype/losses.py:149-156`; `model/fusion.py:54-56`).
- Frozen-transfer recipes may force host adaptation to fixed prototype artifacts, but effect size needs ablation (`configs/head_type/clip/direction2_stage2_optionA_frozen_transfer.yaml:181-195`; `model/prototype/head.py:1043-1111`).

### 9.3 Still unclear from code
- Exact quantitative contribution of each coupling category to observed host drop (requires controlled runs).
- Whether approximate scorer worsens or improves coupling sensitivity relative to exact scorer on this dataset (code supports both, but no built-in comparative evidence).

## 10. Final audit conclusion
The current system is not plug-and-play because prototype behavior is not externalized: it is attached inside the same PAS graph, consumes shared host/backbone features, and is trained through a joint scalar loss that can update host-relevant parameters depending on phase trainability. Inference also uses direct additive fusion and selection-row policies that can change reported model behavior without changing architecture. Code evidence shows this is a coupled co-training system with configurable freeze phases, not an add-on module that preserves host invariance by default (`model/pas_model.py:1079-1136`; `processor/processor.py:283-295`; `model/fusion.py:54-56`; `utils/metrics.py:565-580`).
