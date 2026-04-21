#!/usr/bin/env bash
set -e

run_cmd() {
    shift  

    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")

    local log_file="logs/${timestamp}.log"
    mkdir -p logs

    echo "Starting run at ${timestamp}..."
    nohup "$@" > "$log_file" 2>&1 &
    local pid=$!

    echo "PID: $pid"
    echo "Log: $log_file"

    wait "$pid"
    local status=$?

    if [[ $status -ne 0 ]]; then
        echo "Run failed with exit code $status"
        exit $status
    fi

    echo "Finished run ${timestamp}"
    echo "-----------------------------"
}

run_cmd \
    python train.py \
    --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml \
    --use_loss_semantic_hardneg_margin false \
    --lambda_semantic_pbt 5.0 \
    --lambda_semantic_hardneg_margin 0.0

