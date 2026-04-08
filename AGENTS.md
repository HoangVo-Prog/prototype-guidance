# AGENTS.md

## 0. Purpose

This repository integrates a **prototype-augmented retrieval module** into an existing **ITSELF host model** under a strict **plug-and-play constraint**.

The goal is to:

* preserve the original ITSELF behavior and performance
* add a prototype-based retrieval branch as an additive module
* avoid any modification to the original host source code
* ensure reproducibility and semantic correctness

---

## 1. Sources of Truth

The system has three distinct sources of information. They must be prioritized strictly:

### 1.1 Host Source (Highest priority)

```
prototype/adapter/WACV2026-Oral-ITSELF
```

This is the **only source of truth for host behavior**, including:

* model architecture
* feature extraction
* projection heads
* loss functions
* scoring functions
* training and inference pipeline

**Rules:**

* MUST NOT modify any file inside this directory
* MUST NOT reimplement host logic if callable/importable
* MUST treat host outputs as ground truth representations

---

### 1.2 Method Specification

```
docs/Prototype4ITSELF.md
```

This defines:

* prototype bank
* routing mechanism
* basis construction
* surrogate text object
* prototype scoring
* fusion strategy

**Rules:**

* All prototype-related logic MUST follow this document exactly
* If conflict arises:

  * prefer this spec over legacy code

---

### 1.3 Legacy Code (Reference only)

```
prototype/legacy/PAS-dropping
```

This is **NOT a source of truth**.

**Allowed usage:**

* project structure
* CLI patterns
* config system
* logging utilities
* training scaffolding

**Disallowed usage:**

* blindly copying model logic
* trusting mathematical implementation without verification
* overriding spec or host behavior

---

## 2. Core Architectural Invariants

These invariants MUST NEVER be violated:

### 2.1 Host Preservation

* ITSELF host remains unchanged
* host global + local scoring must remain identical
* host loss must remain identical
* host inference path must remain intact

---

### 2.2 Additive Prototype Branch

* prototype module is a **separate branch**
* it reads host features, but does not modify them
* it produces an additional score

Final score MUST be:

```
s_total = s_host + λ_f * s_proto
```

NOT:

* replacement
* concatenation-based fusion
* modification of host embeddings

---

### 2.3 No Source Modification

* DO NOT edit any file in:

  ```
  prototype/adapter/WACV2026-Oral-ITSELF
  ```
* All new logic must live in:

  ```
  prototype/
  ```

---

### 2.4 Reuse over Reimplementation

* ALWAYS import/call host functions if available
* NEVER duplicate host computations (e.g., cosine similarity, projection)
* NEVER recompute host features manually

---

### 2.5 Explicit Data Flow

Prototype branch must follow:

1. read host features:

   * image global feature
   * text token states

2. compute:

   * routing weights
   * prototype summary
   * basis bank
   * surrogate text

3. compute prototype score

4. fuse at score level only

---

## 3. Required Development Workflow

The agent MUST follow this pipeline strictly:

---

### Phase A. Audit (MANDATORY FIRST STEP)

Before writing any code:

* read host source
* read method spec (`Prototype4ITSELF.md`)
* inspect legacy structure

Produce:

* mapping of host APIs to be reused
* list of safe reusable legacy components
* list of forbidden modifications
* list of semantic risk points

DO NOT write implementation code in this phase.

---

### Phase B. Design

Define:

* module structure
* file organization
* interfaces (inputs/outputs, tensor shapes)
* forward flow step-by-step

All modules must have clear contracts.

---

### Phase C. Skeleton Implementation

Create:

* file structure
* class definitions
* function signatures
* TODO markers for each semantic step

No hidden logic.

---

### Phase D. Full Implementation

Implement:

* prototype bank
* router
* basis builder
* surrogate constructor
* prototype scorer
* fusion module

Respect all invariants.

---

### Phase E. Verification (CRITICAL)

Implement tests BEFORE claiming completion.

---

## 4. Mandatory Tests and Invariants

### 4.1 Host Parity Test

When prototype is disabled or:

```
λ_f = 0
```

Then:

* final score MUST equal host score exactly

---

### 4.2 No-Modification Check

Ensure:

* no file under adapter is modified

---

### 4.3 Shape Consistency

Verify:

* routing weights sum to 1
* prototype dimensions correct
* pairwise score matrix shape = [B, B]

---

### 4.4 Feature Consistency

Ensure:

* correct host features are used:

  * projected vs non-projected must not be confused
* token-level states are used for prototype logic

---

### 4.5 Diagonal Fidelity Correctness

* exact branch uses only matched pairs (i,i)
* no leakage from pairwise candidates

---

### 4.6 Freeze Behavior

If host is frozen:

* no gradients should update host parameters

---

## 5. Configuration Rules

* all new behavior must be controlled via config
* prototype module must be toggleable
* fusion weight must be configurable
* no hardcoded hyperparameters inside modules

---

## 6. Logging Rules

* log:

  * host score
  * prototype score
  * final score
* log routing statistics (entropy, distribution)
* log loss components separately

Do NOT integrate wandb yet.

---

## 7. Failure Conditions

The implementation is considered incorrect if:

* host-only performance changes
* host score is altered
* host code is modified
* prototype replaces host behavior
* feature misuse occurs (wrong embedding source)
* silent shape or semantic mismatch exists

---

## 8. Guiding Principle

This is NOT a rewrite of ITSELF.

This is:

> a controlled augmentation of ITSELF with a prototype-mediated retrieval branch

The host is the system.
The prototype is an extension.

