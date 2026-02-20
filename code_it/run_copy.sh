#!/bin/bash
USER_NAME=$(whoami)
LOG_DATE="/home/$USER_NAME/fm2026/fm_flow/code_it/logs"

echo "[INFO] Removing nohup.out"
rm -rf nohup.out

CONDA_ENV_NAME=$CONDA_DEFAULT_ENV

PYTHON_PATH=/home/$USER_NAME/.conda/envs/$CONDA_ENV_NAME/bin/python
TENSORBOARD_PATH=/home/$USER_NAME/.conda/envs/$CONDA_ENV_NAME/bin/tensorboard
echo "[INFO] Current directory: $(pwd)"

# export DATA_ROOTS="/fast_storage/intern/data/instruction_tuning/fastmri_acceleration_mat_t1, /fast_storage/intern/data/instruction_tuning/fastmri_crossmodal_mat_t1tot2"
export DATA_ROOTS="/fast_storage/intern/data/instruction_tuning/multi_task/acceleration_and_crossmodal, /fast_storage/intern/data/instruction_tuning/multi_task/acceleration_and_segmentation, /fast_storage/intern/data/instruction_tuning/multi_task/denoising_and_crossmodal, /fast_storage/intern/data/instruction_tuning/multi_task/denoising_and_segmentation"

export RUN_DIR=$LOG_DATE
export TRAIN_ITER=1

GPU="3,4,5"
TRAIN_BATCH=24
export CUDA_VISIBLE_DEVICES="$GPU"
nohup $PYTHON_PATH -m torch.distributed.run \
  --nproc_per_node=3 \
  --master_port=29503 \
  train.py \
  --gpu $GPU \
  --train_batch $TRAIN_BATCH \
  --valid_batch 32 \
  --model_type "listfm_it" \
  --pretrain "/home/intern4/fm2026/fm_flow/code_it/logs/00023_train/checkpoints/checkpoint_8.ckpt" \
  --grad_encoder False \
  --use_bottleneck False \
  --use_vision_decoder_weights True \
  --from_scratch False \
  --debugmode True \
  --train_max_per_task 5000 \
  --text_encoding "llm" \
  --num_workers 3 \
  --resume True \
  --resume_path /home/intern4/fm2026/fm_flow/code_it/logs/00028_train/checkpoints/checkpoint_10.ckpt \
  > $RUN_DIR/torchrun_llm.out 2>&1 &

echo "[INFO] Training started on GPU: $GPU with batch size: $TRAIN_BATCH"

# original: /fast_storage/intern/code/share/checkpoint_3m.ckpt

# Kill existing TensorBoard processes
pkill -f "tensorboard --logdir /home/$USER_NAME/fm2026/fm_flow/code_it/logs" || true
sleep 2

# Start TensorBoard with external access
nohup $TENSORBOARD_PATH \
  --logdir /home/$USER_NAME/fm2026/fm_flow/code_it/logs \
  --port 6009 \
  --bind_all \
  > /dev/null 2>&1 &

echo "[INFO] TensorBoard started on port 6009 (accessible externally)"

# sleep 20

# GPU="4,5,6,7"
# TRAIN_BATCH=32
# nohup $PYTHON_PATH train.py \
#   --gpu $GPU \
#   --train_batch $TRAIN_BATCH \
#   --model_type "listfm_it" \
#   --pretrained "/home/juhyung/code/fm2026/code_downstream_recon/checkpoint_300k.ckpt" \
#   --from_scratch False \
#   --subject_num 10 \
#   > /dev/null 2>&1 &


# sleep 20

unset DATA_ROOT
unset TRAIN_ITER
unset RUN_DIR

echo "[INFO] script finished."

# echo "[INFO] Tail logs in: $LOG_DATE"
# find $LOG_DATE -type f -name "*.log" -exec tail -n 100 -f {} +
