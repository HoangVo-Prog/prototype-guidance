#!/usr/bin/env bash
set -e

run_cmd() {
    local name="$1"
    shift

    local log_file="logs/${name}.log"
    mkdir -p logs

    echo "Starting ${name}..."
    nohup "$@" > "$log_file" 2>&1 &
    local pid=$!

    echo "PID: $pid"
    echo "Log: $log_file"

    wait "$pid"
    local status=$?

    if [[ $status -ne 0 ]]; then
        echo "Run ${name} failed with exit code $status"
        exit $status
    fi

    echo "Finished ${name}"
    echo "-----------------------------"
}

run_cmd "run_sem_pbt_5_no_hardneg" \
    python train.py \
    --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml \
    --use_loss_semantic_hardneg_margin false \
    --lambda_semantic_pbt 5.0 \
    --lambda_semantic_hardneg_margin 0.0

run_cmd "run_sem_pbt_2_5_no_hardneg" \
    python train.py \
    --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml \
    --use_loss_semantic_hardneg_margin false \
    --lambda_semantic_pbt 2.5 \
    --lambda_semantic_hardneg_margin 0.0

run_cmd "run_sem_pbt_5_hardneg_100" \
    python train.py \
    --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml \
    --lambda_semantic_pbt 5.0 \
    --lambda_semantic_hardneg_margin 100.0

run_cmd "run_recompute0_ramp0" \
    python train.py \
    --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml \
    --semantic_recompute_start_epoch 0 \
    --semantic_loss_ramp_start_epoch 0

run_cmd "run_recompute_interval_3" \
    python train.py \
    --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml \
    --semantic_recompute_interval 3

run_cmd "run_ramp_steps_2" \
    python train.py \
    --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml \
    --semantic_loss_ramp_steps 2

run_cmd "run_recompute_interval_4" \
    python train.py \
    --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml \
    --semantic_recompute_interval 4

run_cmd "run_recompute_interval_5" \
    python train.py \
    --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml \
    --semantic_recompute_interval 5