# True Plug-and-Play Target Architecture (Repository-Specific)

Evidence base for this document:
- Current-system coupling audit: `docs/investigations/prototype_plug_and_play_full_pipeline_audit_v2.md`
- Redesign claim ladder and intermediate-vs-final framing: `docs/investigations/prototype_plug_and_play_redesign_proposals_v2.md`
- Current code paths cited inline throughout this document.

## 1. Executive conclusion
- The current system is not truly plug-and-play because prototype is instantiated inside the same runtime graph as host (`PASModel` builds `base_model`, `host_head`, `prototype_head`, and `fusion_module` together in `model/pas_model.py:41-118`).
- Optimization is coupled by design through one scalar objective (`loss_total = lambda_host * host_loss + prototype_loss` in `model/pas_model.py:1135`).
- Decision is coupled by direct weighted score addition (`model/fusion.py:32-56`).
- Training semantics are mutable via schedule-time trainability and loss-switch rewrites (`processor/processor.py:270-295`; `utils/freeze_schedule.py:301-391`) rather than hard component boundaries.
- Existing `external_addon` guardrail direction is an intermediate host-safe mode, not architectural modularity, because it still lives inside a single orchestrator graph (`model/pas_model.py:41-118`, `:1057-1135`).
- Strong plug-and-play claims require structural decoupling into explicit components with interface, metric, and checkpoint contracts.

## 2. Design goal
Operational end-state in this repository:
1. Host runs as a first-class independent system.
2. Prototype is attachable/removable as an independent plugin.
3. Attachment uses an explicit read-only interface contract.
4. No host gradient or optimizer coupling unless `joint_training` mode is explicitly selected.
5. Inference modes are first-class runtime modes, not evaluator row tricks.
6. Metric authority and checkpoint authority are mode-specific and explicit.

Architectural meaning of "true plug-and-play" here:
- Component split is real at runtime, not policy-only inside monolith.
- Interface is versioned and compatibility-checked.
- Removal of prototype does not alter HostCore training path, optimizer membership, or checkpoint semantics.

Experimental meaning of "true plug-and-play" here:
- Host parity and isolation obligations pass in external modes.
- Prototype utility is measurable independently.
- Fused gains are additive value only and cannot mask host degradation.

## 3. Required component split

### A. HostCore
Responsibilities:
- Own image/text encoding and host retrieval scoring.
- Provide stable host-only training and inference behavior.
- Export host artifacts through explicit API only.

Allowed inputs:
- Raw image/text batch inputs used by host tasks.

Allowed outputs:
- Host retrieval scores.
- Host export artifact bundle (versioned) for plugin consumption.

Owns:
- Backbone and host retrieval head currently inside `PASModel` + `host_head` path (`model/pas_model.py:47,95-99`; `model/host_heads.py:251-292`; `model/vanilla_clip.py:304-333`).

Must never depend on:
- Prototype presence for forward, optimizer construction, model selection, or checkpoint save criteria.
- Today, this is violated by PAS monolith integration (`model/pas_model.py:41-118,1057-1135`).

### B. PrototypePlugin
Responsibilities:
- Learn/use prototype-specific routing, basis, and scoring from host-exported artifacts.
- Produce prototype-only score matrix and diagnostics.

Allowed inputs from HostCore:
- Only artifacts defined in interface contract (Section 4).

Read-only interface requirement:
- Input artifacts are detached snapshots in external modes.
- Plugin has no mutable handle to HostCore modules.

Owns:
- Prototype bank/routing/pooling/projectors/losses now in `model/prototype/head.py` and `model/prototype/losses.py`.

Must never modify:
- HostCore parameters/state in external modes.
- Host training objectives or optimizer groups in external modes.

### C. Composer
Responsibilities:
- Combine host and prototype scores into fused score when requested.
- Apply calibration/combination policy.

Inputs:
- Host score matrix from HostCore.
- Prototype score matrix from PrototypePlugin.

Outputs:
- Fused score matrix and composition metadata.

Calibration-only, learned, or both:
- Must support calibration-only composition as default safe path.
- May support learned combiner, but only with explicit train mode and authority rules.

Must never control:
- Host training loss semantics.
- Host checkpoint authority for host claims.

Code motivation:
- Current combiner is direct additive in `ResidualScoreFusion.forward` (`model/fusion.py:32-56`); this should become a distinct Composer component.

### D. Optional JointMode
Recommendation:
- Keep support, but as a separate explicit mode (`joint_training`) with separate claims.

Contract:
- JointMode is not plug-and-play by definition.
- Any result from JointMode cannot be used as evidence for modular external claims.

Code motivation:
- Current default behavior is already joint/coupled (`model/pas_model.py:1135`). The redesign preserves this only as explicit non-modular mode.

## 4. Interface contract (HostCore -> PrototypePlugin)

### 4.1 Export artifact schema (minimum)
Minimum export surface needed by current prototype forward signature (`model/prototype/head.py:1007-1019`):
1. `image_embeddings` (global image embedding).
2. `text_token_states`.
3. `token_ids`.
4. `attention_mask`.
5. `special_token_positions`.

Optional export (feature-complete with current routing behavior):
1. `image_local_tokens` (for local routing path).

### 4.2 Artifact form and gradient policy
- External modes (`prototype_only`, `fused_external`, `calibration_only`) require detached snapshots.
- No gradient-through-interface is allowed in external modes.
- JointMode may allow gradient-through-interface only when explicitly configured.

Code motivation:
- Current monolith passes live tensors into prototype path (`model/pas_model.py:1121-1124`).

### 4.3 Exportability rules
- Text states: exportable (detached) because prototype currently consumes them (`model/prototype/head.py:1028-1042`).
- Patch/local tokens: exportable if prototype routing requires them (`model/prototype/head.py:659-664`).
- Host logits: forbidden by default for true external mode. Current weighted prototype loss can consume host logits (`model/prototype/losses.py:523-531`), which is a semantic coupling path.

### 4.4 Interface scope by phase
- Training-time export: allowed for prototype external training.
- Inference-time export: allowed for prototype/fused inference.
- Calibration-time export: allowed for composer calibration.

### 4.5 Versioning and compatibility
- Host export schema must have explicit `interface_version`.
- Prototype checkpoints must declare compatible `interface_version` range.
- Loader rejects incompatible host/prototype pairings before runtime.

### 4.6 Minimum surface vs forbidden convenience
Minimum required:
- Exactly the artifact keys needed for prototype forward and similarity computation.

Forbidden even if convenient:
- Direct HostCore module references.
- Host loss tensors.
- Host optimizer state.
- Host logits in external mode.
- Undeclared extra tensors outside schema.

## 5. Runtime modes (first-class)

| Mode | Components instantiated | Trainable components | Read-only components | Authoritative metrics | Allowed claims | Forbidden claims |
|---|---|---|---|---|---|---|
| `host_only` | HostCore | HostCore | None | Host-only row | Host capability/performance | Any prototype or fused claim |
| `prototype_only` | HostCore (export service) + PrototypePlugin | PrototypePlugin | HostCore | Prototype-only row | Prototype standalone utility | Host improvement/preservation claim |
| `fused_external` | HostCore + PrototypePlugin + Composer | Composer only (optional) | HostCore + PrototypePlugin (or plugin train-frozen depending run) | Fused row for composed system; Host row still tracked | Additive value of composition | Host parity claim without host row parity proof |
| `joint_training` | HostCore + PrototypePlugin + Composer | Explicitly configured joint trainables | None by default | Joint objective metrics + row provenance | Joint co-training result | Plug-and-play modularity claim |
| `calibration_only` | HostCore + PrototypePlugin + Composer | Composer calibration params only | HostCore + PrototypePlugin | Fused calibrated metric + host/proto companion rows | Safer score composition | Training modularity claim |

Important: evaluator row generation is not mode semantics. Today rows are assembled in evaluator (`utils/metrics.py:399-461`); target design moves mode to runtime orchestrator first.

## 6. Training pipeline (target)

### A. Host training
1. Train HostCore alone.
2. Save HostCore checkpoint only.
3. Validate and lock host baseline metrics.

Current-code motivation:
- Host-only runtime already exists in builder (`model/build.py:24-38`), but not yet elevated to independent architecture lifecycle.

### B. Prototype training in true external mode
1. Load frozen HostCore checkpoint.
2. HostCore generates/exports artifacts via interface contract.
3. Train PrototypePlugin only against those artifacts.
4. HostCore params are excluded from optimizer and remain unchanged.

Forbidden couplings:
- No host loss in optimizer objective.
- No host logits in prototype loss (external mode).
- No schedule-time unfreeze of HostCore.

Current-code motivation:
- Existing phase scheduler can unfreeze host and mutate losses (`processor/processor.py:283-295`; `utils/freeze_schedule.py:319-390`), so this must be mode-guarded or bypassed in external pipeline.

### C. Composer training/calibration
1. Freeze HostCore + PrototypePlugin.
2. Train/calibrate Composer on validation/calibration split.
3. Save Composer checkpoint independently.

Allowed objective:
- Composition objective only (no host/prototype parameter updates).

Current-code motivation:
- Fusion is currently direct weighted sum (`model/fusion.py:54-56`) and not isolated as a lifecycle component.

### D. Optional joint mode
1. Explicitly select `joint_training`.
2. Train HostCore and PrototypePlugin with explicit joint optimizer policy.
3. Track results as joint mode only.

Claim constraint:
- Joint results are not evidence of plug-and-play architecture.

## 7. Inference pipeline (target)

### 7.1 Host-only inference
- Inputs -> HostCore -> host score matrix -> ranking.
- No prototype/composer required.

### 7.2 Prototype-only inference
- Inputs -> HostCore export artifacts (read-only) -> PrototypePlugin score matrix -> ranking.
- Host score may be computed for diagnostics but not required for prototype ranking.

### 7.3 Fused inference
- HostCore score + PrototypePlugin score -> Composer -> fused score -> ranking.

### 7.4 Calibration behavior
- Composer applies calibrated combination policy.
- Calibration parameters are component-owned and versioned.

### 7.5 Caching
- Host export artifacts may be cached with interface version and host checkpoint fingerprint.
- Prototype basis artifacts may be cached with prototype checkpoint fingerprint.

### 7.6 Reporting
- Report rows for host-only, prototype-only, fused.
- Include source provenance and mode label as first-class metadata.

Current-code motivation:
- Existing evaluator already tracks `val/top1_row` and `val/top1_source_row` (`utils/metrics.py:623-624`) but selection logic can still remap source/display (`utils/metrics.py:565-610`), which must be replaced by explicit authority rules.

## 8. Metric authority contract (mandatory)

### 8.1 Authority rules
- Host claims must use host-only authoritative metrics.
- Prototype claims must use prototype-only authoritative metrics.
- Composed-system claims must use fused authoritative metrics.

### 8.2 Forbidden interpretations
- Fused gains cannot hide host degradation.
- Prototype utility cannot be inferred from fused rows alone.
- Host preservation cannot be inferred from non-host rows.

### 8.3 Row provenance requirements
- Every selected metric must include explicit source row and mode.
- Checkpoint selection must reference authoritative row for the component being saved.

Current-code motivation:
- Today selection logic can copy subset-best metrics into `pas-t2i` display path under conditions (`utils/metrics.py:565-610`).

## 9. Checkpoint lifecycle (target)

### 9.1 HostCore checkpoint
Contents:
- Host encoder + host retrieval head + host config hash + interface version.

Best-save authority:
- Host-only metric row only.

### 9.2 PrototypePlugin checkpoint
Contents:
- Prototype bank/routing/pooling/projectors/loss config + required interface version range + host compatibility metadata.

Best-save authority:
- Prototype-only metric row only.

### 9.3 Composer checkpoint
Contents:
- Composition/calibration parameters + expected host/prototype score schema versions.

Best-save authority:
- Fused metric row only (with host row audit logged).

### 9.4 Loading order and missing component behavior
Loading order:
1. HostCore.
2. PrototypePlugin (optional depending mode).
3. Composer (optional depending mode).

If component missing:
- Missing prototype: `host_only` remains valid.
- Missing composer with host+prototype present: allow unfused dual reporting, disallow fused deployment mode.

### 9.5 Mixed-source validation
- Validate version compatibility and schema compatibility for each pair.
- Reject runtime if compatibility checks fail.

Current-code motivation:
- Current modular loader supports group-wise mixed-source loads (`utils/modular_checkpoint.py:379-483`) but needs stronger compatibility contract for true modularity claims.

## 10. Proof obligations for true plug-and-play claim
All must pass before claiming true plug-and-play:
1. Host gradient isolation in external modes.
2. Optimizer isolation (no host params in prototype/composer external training).
3. Host parameter delta isolation (no host drift in external modes).
4. Host output parity vs host-only baseline after attach/remove.
5. Clean attach/remove behavior (no runtime surgery, no host retrain requirement).
6. Metric authority correctness (row provenance + authority rules enforced).
7. Checkpoint authority correctness (component-specific best-save rules).
8. Fused additive value demonstrated without host damage.

Current-code motivation:
- Existing optimizer grouping depends on `requires_grad` and schedule mutation (`model/pas_model.py:1002-1017`; `processor/processor.py:283-292`; `solver/build.py:172-205`), so explicit tests are mandatory.

## 11. Migration plan from current repo

### Phase 0: Current audited state
Code areas:
- `model/pas_model.py`, `processor/processor.py`, `utils/metrics.py`, `utils/modular_checkpoint.py`.

What is true:
- Coupled monolith with schedule-driven behavior.

Claims valid:
- Coupled co-training system, not plug-and-play.

### Phase 1: External add-on guardrails (intermediate)
Code areas:
- PAS forward boundary, trainer phase checks, evaluator authority guards.

What becomes cleaner:
- Better host safety under policy.

Risks remaining:
- Still one runtime graph and policy-enforced boundary.

Claims valid:
- Host-safe external add-on mode (intermediate).

Still invalid:
- True modular plug-and-play claim.

### Phase 2: Architectural split
Code areas:
- Split PAS responsibilities into `HostCore`, `PrototypePlugin`, `Composer` orchestrator; update `model/build.py` routing.

What becomes cleaner:
- Real component boundaries and interface-driven attachment.

Risks remaining:
- Migration complexity and compatibility churn.

Claims valid:
- Architectural modularity basis established.

Still invalid:
- Full plug-and-play claim until obligations in Section 10 pass.

### Phase 3: Composer + metric authority cleanup
Code areas:
- Replace evaluator row-selection-driven semantics with mode-authority policy.
- Update checkpoint manager authority rules.

What becomes cleaner:
- Measurement truth aligns with architectural truth.

Risks remaining:
- Calibration overfit risk.

Claims valid:
- Honest component-level and composed-system claims.

Still invalid:
- Any claim unsupported by proof obligations.

### Phase 4: Optional joint mode reintroduction
Code areas:
- Joint trainer path as explicit mode.

What becomes cleaner:
- Research flexibility without semantic confusion.

Risks remaining:
- Claim leakage from joint results into modular claims.

Claims valid:
- Joint co-training performance claims.

Still invalid:
- Plug-and-play claims from joint mode.

## 12. Concrete code mapping (current -> target)

### 12.1 `model/pas_model.py` mapping
- Target HostCore candidates:
  - `extract_image_features`, `extract_text_features`, host scoring path, retrieval encoders (`encode_image_for_retrieval`, `encode_text_for_retrieval`) (`model/pas_model.py:802-843`).
- Target PrototypePlugin candidates:
  - prototype branch invocation and similarity components now in prototype head (`model/pas_model.py:1120-1131`; `model/prototype/head.py:1007-1111`, `:850-1005`).
- Target Composer candidates:
  - `fuse_retrieval_similarity` and current `fusion_module` usage (`model/pas_model.py:871-886`; `model/fusion.py:32-56`).
- Remove from monolith orchestration:
  - single combined loss assembly (`model/pas_model.py:1135`) should be mode-specific outside component internals.

### 12.2 Trainer mapping
- Current mutable schedule logic in `processor/processor.py:270-295` and `utils/freeze_schedule.py:301-391` should be replaced by explicit mode-specific training pipelines.
- `solver/build.py:172-205` remains reusable if invoked per component with explicit ownership.

### 12.3 Evaluator mapping
- Current `_build_similarity_rows` and selection path (`utils/metrics.py:399-461`, `:565-624`) should be rewritten so runtime mode decides authority; evaluator should report, not redefine, semantics.

### 12.4 Checkpoint mapping
- Current checkpoint groups (`utils/module_group_registry.py:48-55`) and manager (`utils/modular_checkpoint.py:165-483`) are reusable scaffolding.
- Add strict compatibility and per-component metric authority policies.

### 12.5 Build/config mapping
- Current builder switch (`model/build.py:14-38`) should evolve into explicit mode + component composition builder.
- Current config keys (`utils/options.py`, `utils/config.py`) should define first-class runtime mode and interface/composer version constraints.

## 13. Decision rules
- External add-on (Phase 1) is enough when goal is short-term host safety testing under current codebase constraints.
- Move to full structural split (Phase 2+) when goal is any strong modularity/plug-and-play claim.
- Strong plug-and-play claims are allowed only after Section 10 obligations pass on the split architecture.
- Claims remain invalid if architecture is still monolithic/policy-enforced, even if host safety improves.

## 14. Final recommendation
- Build next: Phase 2 architectural split plan (HostCore/PrototypePlugin/Composer) with explicit interface schema and mode-specific training entry points.
- Document next: metric authority and checkpoint authority contracts as repository-level standards.
- Do not claim yet: true plug-and-play modularity until split architecture exists and proof obligations pass.

## Minimum viable path to true plug-and-play
- Smallest honest path:
  - Implement Phase 1 guardrails, then immediately implement Phase 2 component split with explicit HostCore -> PrototypePlugin interface versioning.
- Fastest risky shortcut:
  - Keep monolith and rely on detach/freeze policies plus evaluator discipline. This can improve safety but is not true modularity.
- Non-negotiable architectural step:
  - Split PAS monolith into independently loadable/trainable HostCore, PrototypePlugin, and Composer with explicit mode/metric/checkpoint authority.
