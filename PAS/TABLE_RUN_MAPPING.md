# TABLE_RUN_MAPPING

## Scope

This document maps the planned paper result blocks to concrete PAS runs that can be executed with the current repository state.

## Standard Seed Set

Unless a section explicitly says otherwise, use the standardized full-run seed set:
- `1`
- `2`
- `3`

## Table 1. Main PAS Benchmark Results

- Purpose: produce the primary PAS result rows for the three supported text-based person search benchmarks.
- Run group: `pas_main_benchmarks`
- Mode: train + standalone eval
- Config: `configs/train_pas_v1.yaml`
- Required datasets:
  - `CUHK-PEDES`
  - `ICFG-PEDES`
  - `RSTPReid`
- Required seeds per dataset: `3`
- Primary selection metric: `val/top1` during training, which is the same value as `val/pas/R1`
- Reported metrics available from the repo:
  - `val/pas/R1`
  - `val/pas/R5`
  - `val/pas/R10`
  - `val/pas/mAP`
  - `val/pas/mINP`
  - `val/pas/rSum`
- Required runs:
  - `pas_main_cuhk_seed1`
  - `pas_main_cuhk_seed2`
  - `pas_main_cuhk_seed3`
  - `pas_main_icfg_seed1`
  - `pas_main_icfg_seed2`
  - `pas_main_icfg_seed3`
  - `pas_main_rstp_seed1`
  - `pas_main_rstp_seed2`
  - `pas_main_rstp_seed3`
- Artifact requirements per run:
  - `resolved_config.yaml`
  - `configs.yaml`
  - `train_log.txt`
  - `test_log.txt`
  - `best.pth`
  - wandb run link if wandb is enabled
  - manual summary row for result aggregation
- Dependencies: none

## Table 2. Main Comparison Table With External Baselines

- Purpose: place PAS beside external baselines or published comparison methods.
- Run group: `pas_main_comparison_rows`
- PAS runs needed from this repo: exactly the `pas_main_benchmarks` runs from Table 1.
- Additional baseline rows: not produced by this repository in its current Phase F state.
- Mode: train + standalone eval for PAS rows, external import for non-PAS rows.
- Required seeds for PAS rows: `3`
- Primary PAS metric to report: `val/pas/R1`; also report `R5`, `R10`, `mAP`, `mINP`, `rSum` when the comparison table includes them.
- Artifact requirements:
  - all artifacts from Table 1 PAS runs
  - citation/source note for every external baseline row
- Dependencies:
  - depends on Table 1 runs for the PAS row values

## Table 3. Essential PAS Component Ablations

- Purpose: support the core mechanism claim with the ablations that are runnable from the current repo.
- Run group: `pas_essential_ablations`
- Mode: train + standalone eval
- Primary benchmark for ablations: `CUHK-PEDES`
  Reason: all shipped full-run configs default to `CUHK-PEDES`, and no other primary ablation benchmark is explicitly designated in the repo documents.
- Required seeds per ablation: `3`
- Primary selection metric: `val/top1` / `val/pas/R1`
- Reported metrics:
  - `val/pas/R1`
  - `val/pas/R5`
  - `val/pas/R10`
  - `val/pas/mAP`
  - `val/pas/mINP`
  - `val/pas/rSum`
- Required runs:
  - Main PAS reference
    - config: `configs/train_pas_v1.yaml`
    - runs:
      - `pas_ablation_ref_cuhk_seed1`
      - `pas_ablation_ref_cuhk_seed2`
      - `pas_ablation_ref_cuhk_seed3`
  - No contextualization
    - config: `configs/ablation_pas_no_context.yaml`
    - runs:
      - `pas_ablate_no_context_cuhk_seed1`
      - `pas_ablate_no_context_cuhk_seed2`
      - `pas_ablate_no_context_cuhk_seed3`
  - No diversity loss
    - config: `configs/ablation_pas_no_diversity.yaml`
    - runs:
      - `pas_ablate_no_diversity_cuhk_seed1`
      - `pas_ablate_no_diversity_cuhk_seed2`
      - `pas_ablate_no_diversity_cuhk_seed3`
- Artifact requirements:
  - `resolved_config.yaml`
  - `configs.yaml`
  - `train_log.txt`
  - `test_log.txt`
  - `best.pth`
  - wandb run link if enabled
  - manual summary row for the ablation sheet
- Dependencies: none beyond the shipped configs above

## Table 4. Cross-Dataset Generalization / Robustness

- Purpose: support the cross-dataset generalization claim using PAS checkpoints trained on one benchmark and evaluated on another.
- Run group: `pas_cross_domain_eval`
- Mode: eval-only
- Source checkpoints: best checkpoints from Table 1 main PAS runs
- Config base: `configs/train_pas_v1.yaml`
- Required seeds per source-target pair: match the source training seeds, so `3`
- Primary metric to report: `val/pas/R1`
- Additional metrics:
  - `val/pas/R5`
  - `val/pas/R10`
  - `val/pas/mAP`
  - `val/pas/mINP`
  - `val/pas/rSum`
- Required eval-only run families:
  - `pas_xdomain_cuhk_to_icfg_seed{1,2,3}`
  - `pas_xdomain_cuhk_to_rstp_seed{1,2,3}`
  - `pas_xdomain_icfg_to_cuhk_seed{1,2,3}`
  - `pas_xdomain_icfg_to_rstp_seed{1,2,3}`
  - `pas_xdomain_rstp_to_cuhk_seed{1,2,3}`
  - `pas_xdomain_rstp_to_icfg_seed{1,2,3}`
- Artifact requirements per eval run:
  - `test_log.txt` in a dedicated eval output directory or appended eval log in the run directory
  - checkpoint path used
  - source run ID and target dataset recorded in the summary sheet
  - wandb eval link if enabled
- Dependencies:
  - each eval-only run depends on the matching source checkpoint from Table 1

## Table 5. Technical-Spec Ablations Not Yet Runnable As Full Runs

- Purpose: identify result blocks that the planning documents mention but the current repo does not yet support as first-class full-run configs.
- Run group: `blocked_full_run_groups`
- Status: blocked for Phase F documentation; not executable from the current shipped config set without new configs or recovered model branches.
- Blocked groups:
  - No-bank baseline / text-only pooling baseline
    - reason: the active Phase E code path requires `use_prototype_bank=true`; there is no shipped full-run config for a no-bank baseline.
  - EOS-only pooled baseline
    - reason: no shipped full-run config and no active standalone baseline path.
  - Mean-pooling baseline
    - reason: no shipped full-run config and no active standalone baseline path.
  - Prototype-count sensitivity (`N=16`, `N=64`)
    - reason: mentioned in the technical spec, but no dedicated full-run configs are shipped.
  - Temperature sensitivity sweeps
    - reason: mentioned in the technical spec, but no dedicated full-run configs are shipped.
  - Efficiency / profiling table
    - reason: the repo does not export timing/profiling artifacts as a standardized full-run output.
  - Qualitative retrieval analysis table
    - reason: the repo does not export ranked retrieval visualizations or qualitative bundles automatically.
- Dependency note: these blocks need new full-run configs or extra tooling before they can be scheduled as paper-grade runs.

## Summary Of Concrete Runnable Groups

The current repo supports the following full-run groups without additional implementation work:
- `pas_main_benchmarks`
- `pas_main_comparison_rows` for the PAS rows only
- `pas_essential_ablations` for the shipped no-context and no-diversity ablations
- `pas_cross_domain_eval` using checkpoints from the main benchmark runs
