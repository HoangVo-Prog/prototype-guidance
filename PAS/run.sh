DATASET_NAME="CUHK-PEDES"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

CUDA_VISIBLE_DEVICES=0 \
python train.py \
  --config_file configs/train_pas_v1.yaml \
  --dataset_name $DATASET_NAME

# Debug launch:
# python train.py --config_file configs/debug_pas_v1.yaml
# Evaluation:
# python test.py --config_file configs/train_pas_v1.yaml --output_dir <run_dir>
