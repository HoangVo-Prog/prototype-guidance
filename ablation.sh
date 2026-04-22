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

# 1) host_only_baseline
run_cmd "${BASE_CMD[@]}" \
    --runtime_mode host_only \
    --use_hbr false \
    --lambda_hbr 0.0

# 2) host_plus_diag_only (no HBR)
run_cmd "${BASE_CMD[@]}" \
    --runtime_mode joint_training \
    --use_hbr false \
    --lambda_hbr 0.0

# 3) full hbr_proto_weight
run_cmd "${BASE_CMD[@]}" \
    --runtime_mode joint_training \
    --use_hbr true \
    --lambda_hbr 1.0 \
    --hbr_control_mode proto_weight

# 4) hbr_host_only_weight
run_cmd "${BASE_CMD[@]}" \
    --runtime_mode joint_training \
    --use_hbr true \
    --lambda_hbr 1.0 \
    --hbr_control_mode host_only_weight

# 5) hbr_proto_weight_shuffled
run_cmd "${BASE_CMD[@]}" \
    --runtime_mode joint_training \
    --use_hbr true \
    --lambda_hbr 1.0 \
    --hbr_control_mode proto_weight_shuffled

# 6) hbr_random_matched_weight
run_cmd "${BASE_CMD[@]}" \
    --runtime_mode joint_training \
    --use_hbr true \
    --lambda_hbr 1.0 \
    --hbr_control_mode random_matched_weight
