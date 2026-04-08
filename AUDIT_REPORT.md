# AUDIT REPORT

## 1. Executive Summary

- The host system is the ITSELF code under `prototype/adapter/WACV2026-Oral-ITSELF`, with model construction in `model/build.py::build_model` and `model/build.py::ITSELF` (`prototype/adapter/WACV2026-Oral-ITSELF/model/build.py:66`, `:171`, `:296`).
- The prototype branch specification authority is `docs/Prototype4ITSELF.md`, especially Sections 4-15 and 19-24 (host preservation, routing, basis bank, surrogate text, row-wise retrieval, diagonal fidelity, and score-level fusion) (`docs/Prototype4ITSELF.md:125`, `:238`, `:277`, `:336`, `:406`, `:426`, `:480`, `:513`, `:648`, `:818`, `:1000`).
- The highest integration risk is silent semantic drift from using the wrong host tensors or host score surface: global text/image pooled embeddings vs token states vs GRAB features (`prototype/adapter/WACV2026-Oral-ITSELF/model/build.py:103`, `:107`, `:111`, `:116`, `:188`, `:189`, `:241`, `:242`).
- A second high risk is host score definition drift: host evaluation currently computes global and GRAB scores and reports the best R1 over multiple fixed weight mixtures (`prototype/adapter/WACV2026-Oral-ITSELF/utils/metrics.py:126`, `:132`, `:139-153`, `:168`).
- Legacy (`prototype/legacy/PAS-dropping`) is useful for scaffolding patterns, but its core model math is non-authoritative and contains alternative objectives/interfaces that can conflict with the method spec and host-preservation requirement (`prototype/legacy/PAS-dropping/model/build.py:31`, `prototype/legacy/PAS-dropping/model/host_heads.py:295`, `prototype/legacy/PAS-dropping/model/prototype/losses.py:8`).

## 2. Source of Truth Hierarchy

1. Host authority (highest): `prototype/adapter/WACV2026-Oral-ITSELF`
- Source of truth for host architecture, features, losses, and training/inference behavior.
- Must be treated read-only for this project.

2. Method authority: `docs/Prototype4ITSELF.md`
- Source of truth for prototype bank, routing, basis construction, surrogate text, prototype scoring, fusion, and staged training semantics.

3. Legacy reference only: `prototype/legacy/PAS-dropping`
- Allowed as reference for structure/config/CLI/logging/scaffolding after verification.
- Not authoritative for host behavior or method math.

## 3. Host Integration Surface

| Integration need | Exact host surface | Why this is the correct surface | Importable/callable without host edits | Semantic risk if wrong nearby tensor/function is used |
|---|---|---|---|---|
| Host model entrypoint | `model/build.py::build_model(args, num_classes)` -> `ITSELF` (`prototype/adapter/WACV2026-Oral-ITSELF/model/build.py:296`) | Canonical host constructor used by `train.py` and `test.py` (`prototype/adapter/WACV2026-Oral-ITSELF/train.py:54`, `test.py:44`) | Yes | Building a parallel reimplementation would violate host-preservation and reuse constraints |
| Image global feature `v_i^global` | `ITSELF.encode_image` returns `x[:,0,:]` from `base_model.encode_image` (`prototype/adapter/WACV2026-Oral-ITSELF/model/build.py:103-105`) | Same extraction path used by host evaluator (`utils/metrics.py:88`, `:126`) | Yes | Using GRAB local embedding (`encode_image_grab`) would change routing semantics and host alignment basis |
| Text token states `H_j` for prototype basis/teacher | `ITSELF.forward` reads `text_feats` from `self.base_model(images, caption_ids)`; token-level tensor retained before EOT pooling (`prototype/adapter/WACV2026-Oral-ITSELF/model/build.py:236-239`) plus CLIP token pipeline (`model/clip_model.py:575-591`) | This is the only direct host token-state stream used before pooled text extraction | Yes (call through `ITSELF.base_model` or `CLIP.encode_text`) | Using pooled text only (`t_feats`) removes token granularity required by basis and diagonal teacher logic |
| Text global pooled feature | `ITSELF.encode_text` uses EOT index pooling over token outputs (`prototype/adapter/WACV2026-Oral-ITSELF/model/build.py:107-109`) | Matches host global retrieval path used in evaluator (`utils/metrics.py:78`, `:123-127`) | Yes | Confusing pooled global with token states will collapse prototype text construction into a degenerate path |
| Host local branch features | `ITSELF.encode_image_grab` / `encode_text_grab` (`prototype/adapter/WACV2026-Oral-ITSELF/model/build.py:111-119`), with GRAB layers in `model/grab.py` | This is host local branch used in host score mixing and TAL/CID local losses | Yes | Replacing token states with GRAB pooled vectors breaks method-spec basis construction (needs token-level states) |
| Host projection surface | CLIP token projections happen inside `CLIP.encode_text` (`x @ text_projection`) and visual projection in `VisionTransformer.forward` (`x @ self.proj`) (`prototype/adapter/WACV2026-Oral-ITSELF/model/clip_model.py:467-470`, `:589-591`) | Defines what “projected host space” is in current host runtime | Yes | Mixing projected and pre-projection tensors silently changes cosine geometry and retrieval temperatures |
| Host score access (inference) | `utils/metrics.py::Evaluator.eval`: normalized global and GRAB similarities, weighted combinations (`prototype/adapter/WACV2026-Oral-ITSELF/utils/metrics.py:122-153`) | This is the active host scoring/evaluation path | Yes | Re-deriving host score differently can change reported host behavior and invalidate parity |
| Host loss/training path reuse | `ITSELF.forward` computes TAL/CID losses (`prototype/adapter/WACV2026-Oral-ITSELF/model/build.py:244-292`), called by `processor/do_train` (`processor/processor.py:51-66`) | Canonical host training objective flow | Yes | Recomputing losses externally risks mismatch in negative sampling, weighting, and dtype behaviors |
| Data contract for integration | Training batch keys from dataset: `images`, `caption_ids`, `pids` (`datasets/bases.py:126-132`), consumed in model/processor (`model/build.py:183-184`, `processor/processor.py:53-60`) | Stable integration boundary for wrapper modules | Yes | Wrong key/shape assumptions create silent pair misalignment and wrong diagonal supervision |
| Host behavior knobs to preserve | Options in `utils/options.py` for `only_global`, `return_all`, `topk_type`, `layer_index`, `modify_k` (`prototype/adapter/WACV2026-Oral-ITSELF/utils/options.py:73-78`) | These alter host feature path; wrapper must not override semantics silently | Yes | Prototype branch may be validated against a different host mode than baseline if knobs drift |

## 4. Method-to-Code Mapping

Method concepts from `docs/Prototype4ITSELF.md` map to implementation units as follows (conceptual only):

- Prototype bank (`docs/Prototype4ITSELF.md:194`): new module under `prototype/` owning learnable `P` in host embedding dimension.
- Prototype contextualization (`docs/Prototype4ITSELF.md:194`): optional parameter-free/self-attention transform from `P` to `P_tilde`.
- Routing (`docs/Prototype4ITSELF.md:238`): consumes host global image feature from ITSELF (`model/build.py:103-105` or `:188/:237`) and outputs `alpha` with row-sum=1.
- Basis construction (`docs/Prototype4ITSELF.md:277`): consumes host token states `H_j` from host text encoder stream (`model/build.py:236-239`, `clip_model.py:575-591`) to build `b_{j,n}` and `B_j`.
- Surrogate construction (`docs/Prototype4ITSELF.md:336`): combines `alpha_i` with `B_j` to build `t_hat_{j|i}`.
- Prototype scorer (`docs/Prototype4ITSELF.md:406`): compares host image retrieval feature with surrogate projected text, producing `s_proto`.
- Final fusion (`docs/Prototype4ITSELF.md:426`): score-level residual only, `s_total = s_host + lambda_f * s_proto`.
- Training schedule mapping (`docs/Prototype4ITSELF.md:648`, `:676`, `:756`): stage 0 host reproduction, stage 1 frozen-host prototype training, optional stage 2 light co-adaptation, stage 3 fusion calibration.

## 5. Legacy Reuse Assessment

| legacy component | reusable / reusable with verification / not reusable | reason |
|---|---|---|
| `prototype/legacy/PAS-dropping/utils/logger.py` | reusable | Pure logging utility, no model semantics (`utils/logger.py`). |
| `prototype/legacy/PAS-dropping/utils/launch.py` | reusable with verification | Useful run naming/nohup utilities; verify CLI assumptions and OS behavior (`utils/launch.py`). |
| `prototype/legacy/PAS-dropping/utils/options.py` | reusable with verification | Rich CLI surface and removed-flag guardrails are useful, but contains PAS-specific flags and defaults that can conflict with strict host-preservation (`utils/options.py:11`, `:69`, `:322`). |
| `prototype/legacy/PAS-dropping/utils/config.py` | reusable with verification | Strong schema/override utilities; must prune PAS-specific constraints and legacy alias behavior before adoption (`utils/config.py:10`, `:270`, `:538`). |
| `prototype/legacy/PAS-dropping/utils/metric_logging.py` | reusable with verification | Good metric extraction scaffolding; many keys are PAS-specific and should be reduced to required host/prototype/fusion diagnostics (`utils/metric_logging.py`). |
| `prototype/legacy/PAS-dropping/processor/processor.py` | reusable with verification | Training loop scaffolding is reusable, but output-key expectations are PAS-model specific (`processor/processor.py`). |
| `prototype/legacy/PAS-dropping/solver/build.py` | reusable with verification | Optimizer grouping helpers are useful if new model exposes matching group API; validate group names and stage logic (`solver/build.py`). |
| `prototype/legacy/PAS-dropping/configs/stage*/*.yaml` | reusable with verification | Stage scheduling/checkpoint chaining is useful reference only; values must be revalidated against host+spec authoritative behavior. |
| `prototype/legacy/PAS-dropping/model/fusion.py` | reusable | Implements exact residual score fusion shape guard and coefficient handling (`model/fusion.py:7-30`). |
| `prototype/legacy/PAS-dropping/model/prototype/router.py` | reusable with verification | Routing softmax over image-prototype similarity matches spec intent; verify temperature and normalization semantics against method doc (`model/prototype/router.py`). |
| `prototype/legacy/PAS-dropping/model/prototype/aggregator.py` | reusable | `summary = alpha @ prototypes` exactly matches spec primitive (`model/prototype/aggregator.py`). |
| `prototype/legacy/PAS-dropping/model/prototype/contextualizer.py` | reusable with verification | Conceptually aligned optional contextualization; must verify exact residual and normalization settings (`model/prototype/contextualizer.py`). |
| `prototype/legacy/PAS-dropping/model/prototype/token_scorer.py` | reusable with verification | Token-query scoring primitive is aligned; confirm scorer temperature and normalization points with method spec. |
| `prototype/legacy/PAS-dropping/model/prototype/token_pooler.py` | reusable with verification | Masked softmax pooling primitive is reusable if token mask policy is aligned with method assumptions. |
| `prototype/legacy/PAS-dropping/model/prototype/token_mask.py` | reusable with verification | Useful token validity/special-token utility; method spec does not fully define special-token policy so this must be explicitly validated. |
| `prototype/legacy/PAS-dropping/model/prototype/prototype_bank.py` | reusable with verification | Initialization machinery is useful; verify that initialization mode/normalization does not override method-defined training behavior. |
| `prototype/legacy/PAS-dropping/model/prototype/losses.py` | not reusable | Adds class-proxy objectives and loss composition not specified in `Prototype4ITSELF.md` (`model/prototype/losses.py:84`, `:379`). |
| `prototype/legacy/PAS-dropping/model/prototype/head.py` | not reusable | Encodes PAS-specific training/eval contracts and surrogate pipelines beyond strict spec scope (`model/prototype/head.py:720`, `:800`). |
| `prototype/legacy/PAS-dropping/model/prototype/direct_head.py` | not reusable | Implements alternate no-bank behavior not part of authoritative host+spec integration baseline. |
| `prototype/legacy/PAS-dropping/model/host_heads.py` | not reusable | Reimplements ITSELF host behavior inside legacy path, violating host source-of-truth priority (`model/host_heads.py:295`, `:867`). |
| `prototype/legacy/PAS-dropping/model/build.py` | not reusable | Wraps host and prototype inside PAS runtime with additional semantics and interfaces, not authoritative (`model/build.py:31`, `:903`). |
| `prototype/legacy/PAS-dropping/tests/test_model_interface_contract.py` and `tests/test_prototype_modules.py` | reusable with verification | Valuable shape/contract test style, but assertions target PAS-specific model semantics and should not be copied as truth. |

## 6. Semantic Risk Register

| risk | where it is likely to happen | how it should later be tested |
|---|---|---|
| Wrong tensor source for prototype basis (`H_j`) | Using pooled `encode_text` output instead of token states from host text stream (`model/build.py:107-109` vs `:236-239`) | Feature-consistency unit test asserting basis consumes `[B,L,D]` token states and not `[B,D]` pooled vectors |
| Wrong projection stage | Mixing projected tokens from `CLIP.encode_text`/`VisionTransformer` with hypothetical pre-projection tensors (`clip_model.py:467-470`, `:589-591`) | Test comparing norm/cosine statistics and explicit tensor provenance tags |
| Routing on wrong image feature | Routing from GRAB/local instead of host global image feature (`model/build.py:103-105` vs `:111-114`) | Routing-input contract test that source equals host global retrieval feature |
| Host score semantic drift | Replacing evaluator host score logic or changing mix policy (`utils/metrics.py:126-153`) | Host parity test: with prototype disabled or `lambda_f=0`, full ranking matrix and metrics match host baseline exactly |
| Fusion at embedding level instead of score level | Injecting prototype into host embeddings before host scorer | Assertion test that `s_total - s_host == lambda_f * s_proto` at score matrix level |
| Pairwise shape/orientation mismatch | Confusing `[text,image]` vs `[image,text]` in prototype logits and retrieval loss | Shape/orientation tests enforcing pairwise matrix `[B,B]`, row-wise i->text objective semantics |
| Diagonal teacher leakage | Using non-diagonal text candidates in exact branch (`docs/Prototype4ITSELF.md:359`, `:513`) | Test where off-diagonal tokens are perturbed; exact diagonal target must remain unchanged |
| Legacy loss contamination | Accidentally including class-proxy or non-spec losses from legacy (`legacy model/prototype/losses.py:84`, `:379`) | Loss-surface test asserting only allowed host + prototype losses are active per config |
| Accidental host code modification | Editing adapter files directly | CI/no-modification check: git diff restricted to `prototype/adapter/WACV2026-Oral-ITSELF/**` must be empty |
| Frozen-host gradient leak | Stage-1 prototype training updates host params | Freeze-behavior test checking zero/non-updating grads on host params when host frozen |
| Train/inference mismatch | Training with one scorer path and evaluating with different incompatible path | End-to-end test comparing configured scorer path with deployed inference semantics |
| Token mask policy mismatch | Incorrect inclusion/exclusion of SOS/EOS/pad tokens during basis/teacher pooling | Controlled token-mask tests over synthetic captions with known special-token positions |
| Dtype/precision drift | Half/float conversions around host/prototype boundaries (`model/build.py` casts and GRAB ops) | Determinism/regression test comparing FP32 reference outputs with configured precision mode |
| `itself` score-weight ambiguity | Host evaluator currently reports max over multiple global/GRAB weights (`utils/metrics.py:139-153`, `:168`) | Explicit evaluation-policy test requiring a single declared host score definition for fusion |

## 7. No-Touch Host Boundary

Directories and files that must not be modified:

- Entire directory tree: `prototype/adapter/WACV2026-Oral-ITSELF/**`
- This includes host model, losses, data pipeline, evaluator, solver, and utility code under:
  - `prototype/adapter/WACV2026-Oral-ITSELF/model/**`
  - `prototype/adapter/WACV2026-Oral-ITSELF/processor/**`
  - `prototype/adapter/WACV2026-Oral-ITSELF/utils/**`
  - `prototype/adapter/WACV2026-Oral-ITSELF/datasets/**`
  - `prototype/adapter/WACV2026-Oral-ITSELF/solver/**`
  - `prototype/adapter/WACV2026-Oral-ITSELF/train.py`
  - `prototype/adapter/WACV2026-Oral-ITSELF/test.py`

Forbidden change types:

- Any rewrite of host feature extraction, projection, scoring, losses, or training/inference flow.
- Any modification to host defaults that changes baseline behavior.
- Any in-place host API signature change done to "make integration easier".

## 8. Proposed Implementation Surface

Within the plug-and-play boundary, new implementation should live only under `prototype/` outside adapter host code. A safe high-level surface is:

- `prototype/integration/` for host wrappers/adapters that call host APIs without editing host files.
- `prototype/prototype_branch/` for prototype bank, routing, basis, surrogate, scorer modules aligned to `docs/Prototype4ITSELF.md`.
- `prototype/fusion/` for score-level residual fusion.
- `prototype/config/` for new toggleable config surface.
- `prototype/tests/` for Phase E verification tests.

Likely config additions (all toggleable, no hardcoded values):

- `prototype.enabled` (on/off)
- `fusion.lambda_f`
- `prototype.num_prototypes`, `prototype.tau_p`, `prototype.tau_b`, `prototype.tau_t`, `prototype.tau_r`
- `prototype.contextualization.enabled`
- `training.stage` / freeze policy for host vs prototype branch

Mandatory tests before implementation can be considered safe:

- Host parity (`lambda_f=0` or prototype disabled -> identical host score/metrics)
- No-touch host diff check
- Routing row-sum and pairwise score shape checks
- Feature provenance checks (global vs local vs token states)
- Diagonal fidelity isolation check
- Freeze behavior check for host parameters

## 9. Open Questions and Ambiguities

1. Host score definition for fusion is ambiguous.
- Evidence: host evaluator computes many global+GRAB mixtures and takes best R1 (`prototype/adapter/WACV2026-Oral-ITSELF/utils/metrics.py:139-153`, `:168`), while method spec describes a single `s_host` with scalar mixing (`docs/Prototype4ITSELF.md:125`, `:426`).

2. Token-state level used by prototype branch needs explicit agreement.
- Evidence: host text encoder returns projected token states (`prototype/adapter/WACV2026-Oral-ITSELF/model/clip_model.py:575-591`), and ITSELF forward uses these before EOT pooling (`model/build.py:236-239`). Spec denotes `H_j` as token-level host states (`docs/Prototype4ITSELF.md:277`), but does not explicitly state projected vs pre-projection in code terms.

3. Stage-1 frozen-host execution policy is not implemented in adapter host.
- Evidence: adapter host has no dedicated freeze-stage controller in options/train flow (`utils/options.py`, `train.py`, `processor/processor.py`). Spec requires staged training with host freezing (`docs/Prototype4ITSELF.md:676`).

4. Which host global/GRAB weighting should be the canonical `s_host` during fusion is unresolved.
- Evidence: legacy has dataset-specific `itself_score_weight_global` references (`prototype/legacy/PAS-dropping/configs/stage0/stage0_itself_host_only.yaml`), but adapter host evaluator still scans a grid and picks best (`utils/metrics.py:139-153`).

5. Method document contains equation formatting corruption in several lines.
- Evidence: multiple malformed equation blocks (e.g., around Sections 8, 16, 17 in `docs/Prototype4ITSELF.md`) that require interpretation before exact loss/formula implementation.

6. `test.py` in adapter host appears environment-specific and not a clean reusable inference contract.
- Evidence: hardcoded path logic in `prototype/adapter/WACV2026-Oral-ITSELF/test.py:15-24`.

---

Audit scope note: this report intentionally contains no implementation code and no modifications under the host adapter directory.
