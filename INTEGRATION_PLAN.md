# INTEGRATION PLAN

## 1. Purpose and Non-Goals

### Purpose
- This integration MUST add a prototype-augmented retrieval branch on top of the existing ITSELF host runtime.
- This integration MUST preserve host behavior and add only an additive prototype score.
- This integration MUST produce final retrieval scores with EXACT score-level residual fusion:
  - `s_total = s_host + lambda_f * s_proto`

### Non-Goals
- This integration MUST NOT rewrite ITSELF.
- This integration MUST NOT modify any file under `prototype/adapter/WACV2026-Oral-ITSELF/**`.
- This integration MUST NOT replace host embeddings, host scorer, host losses, or host inference flow.
- This integration MUST NOT introduce embedding-level fusion.
- This integration MUST NOT adopt legacy model heads, legacy host wrappers, or legacy loss definitions.

## 2. Authorities and Conflict Resolution

Authority order (strict):
1. Host source: `prototype/adapter/WACV2026-Oral-ITSELF/**`
2. Method spec: `docs/Prototype4ITSELF.md`
3. Legacy reference: `prototype/legacy/PAS-dropping/**`

Conflict rule (mandatory):
- If conflict exists, host semantics always win.

Operational resolution:
- Host behavior contracts (features, scorer, losses, train/infer flow) MUST follow host source exactly.
- Prototype branch semantics (bank, routing, basis, surrogate, prototype score, fusion form) MUST follow `Prototype4ITSELF.md` where they do not alter host semantics.
- Legacy MAY be used only as skeleton utilities after verification; legacy semantics are FORBIDDEN as authority.
- Host runtime behavior MUST NOT be overridden, monkey-patched, or semantically altered via wrapper logic. Host outputs may be observed and consumed, but not redefined.

## 3. Canonical Host Runtime Contract

### Host Mode Rule (EXACT)
The integration runtime MUST support two host modes:

- `train_mode = itself`
- `train_mode = clip`

The meaning of `s_host` is mode-dependent and MUST always refer to the score surface of the currently active host mode.

### Host Instantiation (EXACT)

For `train_mode = itself`:
- Host model MUST be instantiated through host builder path:
  - `prototype/adapter/WACV2026-Oral-ITSELF/model/build.py::build_model(args, num_classes)`
- Host class contract MUST remain:
  - `prototype/adapter/WACV2026-Oral-ITSELF/model/build.py::ITSELF`

For `train_mode = clip`:
- A dedicated CLIP host runtime MUST be used under the integration project runtime.
- The CLIP host runtime MUST expose a canonical CLIP retrieval score surface.
- The CLIP host runtime MUST NOT be aliased to ITSELF global+local semantics.
- The CLIP host runtime MUST NOT introduce any GRAB/local branch.

### Host Feature Extraction (EXACT)

For `train_mode = itself`:
- Global image feature source MUST be host global CLS-pooled image feature:
  - `v_i_global` from `ITSELF.encode_image(...)` -> `x[:, 0, :]`
- Token-level text states source MUST be taken from the same representation space used by the host immediately before global EOT pooling, including all projection and normalization steps applied by the host text encoder:
  - `H_j` shape `[B, L, D]`
- Global text embedding source MUST be host EOT-pooled global text embedding:
  - `t_j_global` from host text path using EOT index
- Local host feature source MUST be the host GRAB/local branch only

For `train_mode = clip`:
- Global image feature source MUST be the canonical CLIP global image retrieval feature
- Token-level text states source MUST be taken from the same representation space used by the CLIP host immediately before global text pooling
- Global text embedding source MUST be the canonical CLIP pooled global text embedding
- No local or GRAB feature source exists in this mode

Mixing pre-projection and post-projection representations across modules is FORBIDDEN.

### Host Score (EXACT)

For `train_mode = itself`:
- Canonical host score MUST be:
  - `s_host = s_host^itself = lambda_s * s_global^itself + (1 - lambda_s) * s_local^itself`
- `s_global^itself` MUST be host global similarity from host global image/text features
- `s_local^itself` MUST be host local similarity from host GRAB/local branch
- `lambda_s` MUST come from config

For `train_mode = clip`:
- Canonical host score MUST be:
  - `s_host = s_host^clip`
- `s_host^clip` MUST be the canonical CLIP retrieval score from the CLIP host path
- `s_host^clip` MUST NOT be interpreted as ITSELF global+local score
- No GRAB/local term is present in this mode

### Explicit prohibition
- Evaluator grid search over multiple host mixing weights is FORBIDDEN for canonical integration runtime
- Selecting best host score from a weight sweep is FORBIDDEN
- When `train_mode = clip`, any wording, config, or code path that interprets `s_host` as ITSELF global+local score is FORBIDDEN

## 4. Prototype Branch Dataflow (STRICT)

All prototype operations are additive and read-only with respect to host tensors.

### Step 1: Routing Input
- Input tensor: `v_i_global`
- Shape: `[B, D]`
- Source: host global image feature (host)
- Output: routing input tensor for prototype router

### Step 2: Prototype Bank
- Input: learnable bank `P`
- Shape: `[N, D]`
- Source: prototype module (prototype)
- Output: `P` and optional contextualized `P_tilde` `[N, D]`

### Step 3: Routing Weights
- Input tensors: `v_i_global [B, D]`, `P_tilde [N, D]`
- Output tensor: `alpha`
- Shape: `[B, N]`
- Source: prototype router (prototype)
- Constraint: each row MUST sum to 1

### Step 4: Prototype Summary
- Input tensors: `alpha [B, N]`, `P_tilde [N, D]`
- Output tensor: `q_i`
- Shape: `[B, D]`
- Source: prototype aggregator (prototype)

### Step 5: Basis Construction from Token States
- Input tensors: `H_j [B, L, D]`, `P_tilde [N, D]`
- Output tensor: `B_j` (basis bank)
- Shape: `[B, N, D]`
- Source: prototype basis module (prototype)
- Constraint: token substrate MUST be host token states `H_j` (host-derived), not pooled text.

### Step 6: Surrogate Construction
- Input tensors: `alpha_i [B, N]`, `B_j [B, N, D]`
- Output tensor: `t_hat_{j|i}`
- Shape: pairwise surrogate pooled text; diagonal training path `[B, D]`, pairwise scoring path logically `[B, B, D]` (chunked computation allowed)
- The surrogate construction MUST preserve row-wise semantics:
for each image i, the constructed surrogate text objects correspond to all captions j.
Any implicit transpose or symmetric reformulation of the pairwise computation is FORBIDDEN.
- Source: prototype surrogate constructor (prototype)

### Step 7: Prototype Projection
- Input tensor: `t_hat_{j|i}`
- Output tensor: `z_hat_{j|i}^{t,proto}`
- Shape: projection dim `D_r` (typically `[B, D_r]` diagonal, pairwise `[B, B, D_r]` in chunked form)
- Source: prototype text projector (prototype)

### Step 8: Prototype Score
- Inputs: host image retrieval feature `z_i^{v,host}` and surrogate prototype text embedding `z_hat_{j|i}^{t,proto}`
- Output: `s_proto`
- Shape: `[B, B]`
- The image feature used for routing and the image feature used for prototype scoring MUST belong to the same semantic feature space as the host retrieval feature.
Any mismatch in normalization, projection, or scaling between routing and scoring inputs is FORBIDDEN.
- Source split:
  - image side: host-derived feature (host, read-only)
  - text side and scoring path: prototype branch (prototype)

### Final Fusion (STRICT)
- Output MUST be:
  - `s_total = s_host + lambda_f * s_proto`
- Fusion location MUST be score tensor level only.
- Embedding-level fusion is FORBIDDEN.

## 5. Module Boundaries and File Placement

New code MUST be placed only under `prototype/` and MUST NOT touch adapter host files.

Required placement:
- `prototype/integration/`
  - host runtime adapter/wrapper, feature surface extraction, stage control, provenance checks
- `prototype/prototype_branch/`
  - prototype bank, contextualizer, router, basis builder, surrogate builder, prototype scorer
- `prototype/fusion/`
  - residual score-level fusion module
- `prototype/config/`
  - config schema/defaults/validation
- `prototype/tests/`
  - all verification tests

Boundary rule:
- No code is added inside adapter host directory.


## 6. Execution Training Protocol (Stage-wise)

The stage protocol defined here is the authoritative execution schedule for this integration project.
It refines the broader staged-training idea in `Prototype4ITSELF.md` into an implementation-specific protocol required by this repository.
Stage behavior is host-mode dependent and MUST distinguish ITSELF-hosted execution from CLIP-hosted execution.

### Stage 0: Host Reproduction and Host-Only Baselines

**Purpose**
- Establish faithful host baselines before any prototype integration
- Support both:
  - ITSELF host reproduction
  - vanilla CLIP retrieval baseline for standard I2T / T2I training and evaluation

#### Stage 0A: ITSELF host reproduction
- Active host mode: `train_mode = itself`
- Trainable parameters: ITSELF host parameters only
- Prototype branch: disabled
- Active score:
  - `s_host = s_host^itself = lambda_s * s_global^itself + (1 - lambda_s) * s_local^itself`
- Active losses: original ITSELF host losses only
- Goal:
  - reproduced ITSELF performance MUST match original ITSELF behavior as closely as possible
  - no prototype module may participate in scoring, loss, or feature construction

#### Stage 0B: CLIP host baseline
- Active host mode: `train_mode = clip`
- Trainable parameters: CLIP baseline parameters only
- Prototype branch: disabled
- Active score:
  - `s_host = s_host^clip`
- `s_host` in this stage MUST refer to CLIP retrieval score only, not ITSELF global+local score
- Active losses: standard CLIP retrieval loss for conventional I2T / T2I training
- Goal:
  - provide a clean CLIP host baseline under the same project runtime
  - support later prototype-enabled retraining initialized from CLIP

**Mandatory constraints**
- Stage 0 MUST support both host-only runtime modes:
  - `train_mode = itself`
  - `train_mode = clip`
- Stage 0 MUST NOT enable prototype scoring or prototype losses
- Stage 0 outputs MUST be tracked separately for:
  - ITSELF baseline checkpoints and metrics
  - CLIP baseline checkpoints and metrics

---

### Stage 1: Prototype Stabilization Under Frozen Host

**Purpose**
- Stabilize prototype learning under a fixed host feature space

**Supported host modes**
- `train_mode = itself`
- `train_mode = clip`

**Trainable parameters**
- Prototype branch parameters only:
  - prototype bank
  - optional contextualization
  - routing module
  - basis construction module
  - surrogate construction module
  - prototype projector and scorer if used

**Frozen parameters**
- All host parameters frozen
- No host encoder, host backbone, or host retrieval parameter may update

**Prototype branch**
- Enabled

**Active losses**
- `L_div`
- `L_diag`
- `L_ret`

**Host score semantics**
- If `train_mode = itself`, then:
  - `s_host = s_host^itself`
- If `train_mode = clip`, then:
  - `s_host = s_host^clip`

**Mandatory constraints**
- Host forward is used as feature provider only
- No host parameter update is allowed
- No host loss rewrite is allowed
- No embedding-level fusion is allowed
- Host feature extraction MUST follow the active host mode exactly

**Goal**
- learn a stable prototype subsystem
- prevent collapse
- anchor surrogate semantics before backbone adaptation

---

### Stage 2: Prototype-Enabled Retraining with Unfrozen Backbone

**Purpose**
- Train the integrated model with prototype enabled and backbone unfrozen
- Support retraining from CLIP initialization for both architecture families:
  - ITSELF-style architecture
  - CLIP-style architecture

**Supported runtime modes**
- `train_mode = itself`
- `train_mode = clip`

**Initialization**
- Stage 2 MUST support initialization from CLIP checkpoints or configured CLIP starting state
- Stage 1 is used to stabilize the prototype subsystem
- Stage 2 is the main retraining stage with unfrozen backbone

**Trainable parameters**
- Host backbone unfrozen
- Prototype branch either:
  - trainable
  - partially trainable
  - lightly frozen
- Exact prototype freeze policy MUST be config-controlled

**Host score semantics**
- If `train_mode = itself`, then:
  - `s_host = s_host^itself = lambda_s * s_global^itself + (1 - lambda_s) * s_local^itself`
- If `train_mode = clip`, then:
  - `s_host = s_host^clip`

**Active losses**
- Active host losses for the selected runtime mode
- Prototype losses as configured

**Mandatory constraints**
- `s_host` MUST always refer to the score surface of the currently active host mode
- When `train_mode = clip`, implementation MUST NOT assume the existence of GRAB/local score
- When `train_mode = itself`, implementation MUST preserve the declared fixed global/local weighting from config
- No host semantic rewrite is allowed
- No replacement of host scoring path is allowed
- No hidden fallback from CLIP mode to ITSELF score definition is allowed
- No hidden fallback to legacy semantics is allowed

**Goal**
- perform full integrated training under the selected host mode with prototype enabled

---

### Stage 3: Fusion Calibration

**Purpose**
- Tune final fusion behavior after representation learning has completed

**Trainable parameters**
- Calibration target only, typically `lambda_f`

**Frozen parameters**
- Backbone frozen
- Prototype representation parameters frozen
- Host representation parameters frozen

**Host score semantics**
- If `train_mode = itself`, fusion uses:
  - `s_total = s_host^itself + lambda_f * s_proto`
- If `train_mode = clip`, fusion uses:
  - `s_total = s_host^clip + lambda_f * s_proto`

**Mandatory constraints**
- Host score definition MUST remain fixed during calibration
- When `train_mode = itself`, the host score surface is:
  - `s_host^itself = lambda_s * s_global^itself + (1 - lambda_s) * s_local^itself`
- When `train_mode = clip`, the host score surface is:
  - `s_host^clip`
- No feature extraction, normalization, routing semantics, prototype construction, or host score semantics may change in this stage

**Goal**
- calibrate additive prototype contribution on top of the selected host score surface



## 7. Config Design

All behavior MUST be config-driven. Hardcoding is FORBIDDEN.

Required config keys:
- `train_mode` (`itself|clip`)
- `prototype.enabled` (bool)
- `prototype.num_prototypes` (int)
- `prototype.dim` (int, must match selected host feature space contract)
- `prototype.contextualization.enabled` (bool)
- `prototype.contextualization.type` (enum)
- `prototype.routing.temperature` (`tau_p`)
- `prototype.basis.temperature` (`tau_b`)
- `prototype.teacher.temperature` (`tau_t`)
- `prototype.retrieval.temperature` (`tau_r`)
- `prototype.regularization.diversity.enabled` / weight
- `prototype.regularization.balance.enabled` / weight
- `fusion.enabled` (bool)
- `fusion.lambda_f` (float)
- `host.lambda_s` (float, required only for `train_mode = itself`)
- `training.stage` (`stage0|stage1|stage2|stage3`)
- `training.freeze.host` (bool)
- `training.freeze.prototype` (bool)
- `training.unfreeze.host_allowlist` (list, stage2 optional)
- `loss.host.enabled` (bool)
- `loss.prototype.ret.enabled` (bool)
- `loss.prototype.diag.enabled` (bool)

Validation rules:
- `fusion.lambda_f` MUST be present when `prototype.enabled=true`
- `host.lambda_s` MUST be present when `train_mode = itself`
- `host.lambda_s` MUST NOT be used to define `s_host` when `train_mode = clip`
- `training.stage=stage1` MUST enforce host freeze
- `training.stage=stage0` MUST enforce prototype disabled
- `train_mode = clip` MUST disable any assumption of GRAB/local host score

## 8. Verification Plan (MANDATORY)

### Test A: Host Parity Test
- Check: with `prototype.enabled=false` OR `fusion.lambda_f=0`, integrated runtime scores equal the active host-mode score exactly
- Invariant:
  - if `train_mode = itself`, then `s_total == s_host^itself`
  - if `train_mode = clip`, then `s_total == s_host^clip`

### Test B: No-Touch Host Diff Test
- Check: no modified files under `prototype/adapter/WACV2026-Oral-ITSELF/**`.
- Invariant: host directory git diff is empty.

### Test C: Routing Row-Sum Test
- Check: `alpha` produced by router.
- Invariant: each row sum equals 1; no NaN/Inf.

### Test D: Tensor Provenance Test
- Check: routing input and basis inputs.
- Invariant:
  - routing input MUST originate from host global image feature
  - basis construction MUST use host token-level text states `H_j`
  - pooled text-only input to basis is FORBIDDEN

### Test E: Diagonal Fidelity Isolation Test
- Check: exact teacher branch uses only matched `(i,i)` pairs.
- Invariant: no off-diagonal leakage into exact diagonal target.

### Test F: Freeze Gradient Test
- Check: gradients and optimizer updates under stage freeze policy.
- Invariant:
  - stage1: host params unchanged
  - stage3: host and prototype representation params unchanged

### Test G: Shape/Orientation Test
- Check: prototype pairwise score matrix orientation and shape.
- Invariant:
  - pairwise score shape is `[B, B]`
  - row-wise retrieval loss treats row `i` as image `i` against all texts `j`
  - fusion operands have identical shape

### Test H: Fusion-Level Test
- Check: fusion location and arithmetic.
- Invariant: `s_total - s_host == lambda_f * s_proto` at score tensor level.

### Test I: Row-wise Perturbation Test
- Perturb one image feature v_i only
- Check: only row i of s_proto changes significantly
- Invariant:
  - other rows remain unchanged (within tolerance)
  - ensures correct row-wise retrieval semantics

## 9. Explicit Forbidden Implementations

The following are FORBIDDEN:
- Modifying any host file under `prototype/adapter/WACV2026-Oral-ITSELF/**`.
- Embedding-level fusion or concatenation-based fusion.
- Replacing host embeddings with prototype embeddings.
- Replacing or redefining host score computation.
- Using pooled global text embedding as basis input instead of token states `H_j`.
- Using GRAB/local embedding as routing input.
- Copying legacy host wrappers/legacy host heads.
- Reusing legacy prototype loss compositions not defined by `Prototype4ITSELF.md`.
- Recomputing host score in a non-host way.
- Enabling host score grid-search as canonical runtime behavior.
- Hardcoding prototype/fusion hyperparameters in code.
- Using the symbol `s_host` without binding it to the active `train_mode` is FORBIDDEN in implementation.

