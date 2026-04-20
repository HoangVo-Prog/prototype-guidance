#!/usr/bin/env bash

set -e  # nếu 1 lệnh fail -> dừng toàn bộ

COMMANDS=$(cat <<'EOF'
python train.py --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml --nohup --use_loss_semantic_hardneg_margin false --lambda_semantic_pbt 5.0 --lambda_semantic_hardneg_margin 0.0
python train.py --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml --nohup --use_loss_semantic_hardneg_margin false --lambda_semantic_pbt 2.5 --lambda_semantic_hardneg_margin 0.0
python train.py --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml --nohup --lambda_semantic_pbt 5.0 --lambda_semantic_hardneg_margin 100.0
python train.py --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml --nohup --semantic_recompute_start_epoch 0 --semantic_loss_ramp_start_epoch 0
python train.py --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml --nohup --semantic_recompute_interval 3
python train.py --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml --nohup --semantic_loss_ramp_steps 2
python train.py --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml --nohup --semantic_recompute_interval 4
python train.py --config_file /home/vhoang/prototype-guidance/configs/semantic_structure/itself.yaml --nohup --semantic_recompute_interval 5
EOF
)

# chạy từng lệnh
while IFS= read -r cmd; do
    # bỏ qua dòng rỗng
    [[ -z "$cmd" ]] && continue

    echo "Running: $cmd"
    eval "$cmd"

    echo "Finished: $cmd"
    echo "-----------------------------"
done <<< "$COMMANDS"