#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/vhoang/prototype-guidance"
LOG_DIR="${PROJECT_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

run_cmd() {
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S_%N")

    local log_file="${LOG_DIR}/${timestamp}.log"

    echo "========================================"
    echo "Starting run at ${timestamp}"
    echo "Log file: ${log_file}"
    echo "Command: $*"
    echo "========================================"

    if "$@" > "${log_file}" 2>&1; then
        echo "Finished run ${timestamp}"
    else
        local status=$?
        echo "Run failed with exit code ${status}"
        echo "Check log: ${log_file}"
        exit ${status}
    fi
}

# ===== RUNS =====

run_cmd python train.py \
    --config_file configs/semantic_structure/itself_lr_ablation.yaml

run_cmd python train.py \
    --config_file configs/host_only/itself.yaml

run_cmd python train.py \
    --config_file configs/semantic_structure/itself.yaml \
    --epochs 25

run_cmd python train.py \
    --config_file configs/semantic_structure/itself.yaml \
    --epochs 25 \
    --semantic_recompute_interval 1

run_cmd python train.py \
    --config_file configs/semantic_structure/itself.yaml \
    --epochs 25 \
    --semantic_recompute_interval 2

run_cmd python train.py \
    --config_file configs/semantic_structure/itself.yaml \
    --prototype_num_prototypes 64 \
    --epochs 25

run_cmd python train.py \
    --config_file configs/semantic_structure/itself.yaml \
    --prototype_num_prototypes 128 \
    --epochs 25