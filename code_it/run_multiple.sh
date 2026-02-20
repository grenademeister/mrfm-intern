#!/bin/bash
set -euo pipefail

USER_NAME=$(whoami)
LOG_DATE="/home/$USER_NAME/fm2026/fm_flow/code_it/logs"

echo "[INFO] Removing nohup.out"
rm -rf nohup.out

CONDA_ENV_NAME=$CONDA_DEFAULT_ENV
PYTHON_PATH="/home/$USER_NAME/.conda/envs/$CONDA_ENV_NAME/bin/python"
TENSORBOARD_PATH="/home/$USER_NAME/.conda/envs/$CONDA_ENV_NAME/bin/tensorboard"
echo "[INFO] Current directory: $(pwd)"

export DATA_ROOTS="/fast_storage/intern/data/instruction_tuning/multi_task/acceleration_and_crossmodal, /fast_storage/intern/data/instruction_tuning/multi_task/acceleration_and_segmentation, /fast_storage/intern/data/instruction_tuning/multi_task/denoising_and_crossmodal, /fast_storage/intern/data/instruction_tuning/multi_task/denoising_and_segmentation"
export RUN_DIR="$LOG_DATE"
export TRAIN_ITER=1

GPU="3,4,5"
TRAIN_BATCH=24
export CUDA_VISIBLE_DEVICES="$GPU"
MASTER_PORT=29503

# Staged training setup
FILE_COUNTS=(500 2000 5000)
DEFAULT_EPOCHS=13
EPOCHS=()

build_limit_list() {
  local count=$1
  local value=$2
  local result=""
  local i=0
  while [ $i -lt $count ]; do
    if [ -z "$result" ]; then
      result="$value"
    else
      result="${result},${value}"
    fi
    i=$((i + 1))
  done
  echo "$result"
}

IFS=',' read -ra ROOTS <<< "$DATA_ROOTS"
TASK_COUNT=${#ROOTS[@]}

# Kill existing TensorBoard processes
pkill -f "tensorboard --logdir /home/$USER_NAME/fm2026/fm_flow/code_it/logs" || true
sleep 2

# Start TensorBoard with external access
nohup "$TENSORBOARD_PATH" \
  --logdir /home/$USER_NAME/fm2026/fm_flow/code_it/logs \
  --port 6009 \
  --bind_all \
  > /dev/null 2>&1 &

echo "[INFO] TensorBoard started on port 6009 (accessible externally)"

for idx in "${!FILE_COUNTS[@]}"; do
  FILE_COUNT="${FILE_COUNTS[$idx]}"
  EPOCH="${EPOCHS[$idx]:-$DEFAULT_EPOCHS}"
  TRAIN_MAX_PER_TASK="$(build_limit_list "$TASK_COUNT" "$FILE_COUNT")"

  echo "[INFO] Stage $((idx + 1))/$(( ${#FILE_COUNTS[@]} )): files per task=${FILE_COUNT}, epochs=${EPOCH}"

  nohup "$PYTHON_PATH" -m torch.distributed.run \
    --nproc_per_node=3 \
    --master_port="$MASTER_PORT" \
    train.py \
    --gpu "$GPU" \
    --train_batch "$TRAIN_BATCH" \
    --valid_batch 32 \
    --model_type "listfm_it" \
    --pretrain "/home/intern4/fm2026/fm_flow/code_it/logs/00023_train/checkpoints/checkpoint_8.ckpt" \
    --grad_encoder False \
    --use_bottleneck False \
    --use_vision_decoder_weights True \
    --from_scratch False \
    --debugmode True \
    --train_max_per_task "$TRAIN_MAX_PER_TASK" \
    --train_epoch "$EPOCH" \
    --text_encoding "llm" \
    --num_workers 3 \
    > "$RUN_DIR/torchrun_llm_${FILE_COUNT}.out" 2>&1 &

  pid=$!
  wait "$pid"

  latest_run_dir=$(ls -td "$RUN_DIR"/*_train | head -n 1)
  ckpts=("$latest_run_dir"/checkpoints/checkpoint_*.ckpt)
  if [ ${#ckpts[@]} -eq 0 ]; then
    echo "[ERROR] No checkpoint found in $latest_run_dir/checkpoints. Stopping."
    exit 1
  fi
  latest_ckpt=$(ls -t "${ckpts[@]}" | head -n 1)
  PRETRAIN_CKPT="$latest_ckpt"
done

echo "[INFO] Training finished for all stages."

unset DATA_ROOT
unset TRAIN_ITER
unset RUN_DIR

echo "[INFO] script finished."
