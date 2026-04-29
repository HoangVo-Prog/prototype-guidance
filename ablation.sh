#!/usr/bin/env bash
set -euo pipefail

run_cmd() {
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")

    mkdir -p logs
    local log_file="logs/${timestamp}.log"

    echo "Starting run at ${timestamp}..."
    echo "Log: ${log_file}"
    echo "Command: $*"

    "$@" > "$log_file" 2>&1
    local status=$?

    if [[ $status -ne 0 ]]; then
        echo "Run failed with exit code $status"
        echo "Check log: ${log_file}"
        exit $status
    fi

    echo "Finished run ${timestamp}"
    echo "-----------------------------"
}

run_cmd \
    python /home/vhoang/prototype-guidance/train.py \
    --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml \
    --semantic_loss_ramp_start_epoch 0

run_cmd \
    python /home/vhoang/prototype-guidance/train.py \
    --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml \
    --use_loss_semantic_hardneg_margin false \
    --lambda_semantic_hardneg_margin 0.0 

run_cmd \
    python /home/vhoang/prototype-guidance/train.py \
    --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml \
    --semantic_recompute_start_epoch 3 \
    --semantic_loss_ramp_start_epoch 3
