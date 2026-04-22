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

BASE_CMD=(
    python /home/vhoang/prototype-guidance/train.py
    --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml
    --epochs 20
)


# hbr_proto_weight_shuffled
run_cmd "${BASE_CMD[@]}" \
    --hbr_control_mode proto_weight_shuffled

# hbr_random_matched_weight
run_cmd "${BASE_CMD[@]}" \
    --hbr_control_mode random_matched_weight

run_cmd "${BASE_CMD[@]}" \
    --hbr_control_mode proto_weight_shuffled \
    --hbr_inner_tail_weight_mode sigmoid_proto

run_cmd "${BASE_CMD[@]}" \
    --hbr_control_mode proto_weight_shuffled \
    --hbr_inner_tail_weight_mode softmax_proto

run_cmd "${BASE_CMD[@]}" \
    --hbr_control_mode random_matched_weight
    --hbr_inner_tail_weight_mode sigmoid_proto

run_cmd "${BASE_CMD[@]}" \
    --hbr_control_mode random_matched_weight
    --hbr_inner_tail_weight_mode softmax_proto
