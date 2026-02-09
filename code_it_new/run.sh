#!/bin/bash
USER_NAME=$(whoami)
LOG_DATE="/home/$USER_NAME/fm2026/mrfm-intern/code_it_new/logs"

echo "[INFO] Removing nohup.out"
rm -rf nohup.out

CONDA_ENV_NAME=$CONDA_DEFAULT_ENV

PYTHON_PATH=/home/$USER_NAME/.conda/envs/$CONDA_ENV_NAME/bin/python
TENSORBOARD_PATH=/home/$USER_NAME/.conda/envs/$CONDA_ENV_NAME/bin/tensorboard
echo "[INFO] Current directory: $(pwd)"

export DATA_ROOTS="/fast_storage/intern/data/instruction_tuning/fastmri_acceleration_mat"
# export DATA_ROOTS="/fast_storage/intern/data/instruction_tuning/brats_crossmodal_mat_simple"
# export DATA_ROOTS="/fast_storage/intern/data/instruction_tuning/brats_denoise_mat"
# export DATA_ROOTS="/fast_storage/intern/data/instruction_tuning/brats_segmentation_mat_simple"
# export DATA_ROOTS="/fast_storage/intern/data/instruction_tuning/oasis3_longitudinal_mat_new"

# use all
# export DATA_ROOT=["/fast_storage/intern/data/instruction_tuning/fastmri_acceleration_mat",
# "/fast_storage/intern/data/instruction_tuning/brats_crossmodal_mat_simple",
# "/fast_storage/intern/data/instruction_tuning/brats_denoise_mat",
# "/fast_storage/intern/data/instruction_tuning/brats_segmentation_mat_simple"]

export RUN_DIR=$LOG_DATE
export TRAIN_ITER=1

GPU="2,3,4"
TRAIN_BATCH=48
nohup $PYTHON_PATH train.py \
  --gpu $GPU \
  --train_batch $TRAIN_BATCH \
  --model_type "listfm_it" \
  --pretrained "/fast_storage/intern/code/share/checkpoint_3m.ckpt" \
  --from_scratch False \
  > /dev/null 2>&1 &

# original: /fast_storage/intern/code/share/checkpoint_3m.ckpt

echo "[INFO] Training started on GPU: $GPU with batch size: $TRAIN_BATCH"

nohup $TENSORBOARD_PATH --logdir /home/$USER_NAME/fm2026/mrfm-intern/code_it_new/logs > /dev/null 2>&1 &
echo "[INFO] TensorBoard started."

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
