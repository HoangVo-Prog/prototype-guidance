#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_FILE="${CONFIG_FILE:-configs/stage1/prototype_init.yaml}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Edit this list to choose which prototype initialization modes to run.
PROTOTYPE_INITS=(
  normalized_random
  orthogonal_normalized_random
  spherical_kmeans_centroids
  hybrid_spherical_kmeans_random
)

# Important:
# Run THIS script with nohup if you want the whole sweep in the background.
# Do not pass --nohup into train.py here, otherwise each training run will
# detach immediately and the loop will start the next run right away.
for prototype_init in "${PROTOTYPE_INITS[@]}"; do
  echo "Running train.py with --prototype_init ${prototype_init}"
  echo "Config: ${CONFIG_FILE}"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  "${PYTHON_BIN}" train.py \
    --config_file "${CONFIG_FILE}" \
    --prototype_init "${prototype_init}" \
    "$@"

  echo "Finished --prototype_init ${prototype_init}"
  echo
done
