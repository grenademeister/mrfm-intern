#!/bin/bash

LOG_DATE="/home/intern4/fm2026/fm_flow/code_it/logs"

# echo "[INFO] Removing previous log directory: $LOG_DATE"
# rm -rf $LOG_DATE

echo "[INFO] Removing nohup.out"
rm -rf nohup.out

PYTHON_PATH=/home/intern4/.conda/envs/research/bin/python

echo "[INFO] Current directory: $(pwd)"
# echo "[INFO] Python files and contents in current directory:"
# for f in *.py; do
#   if [ -f "$f" ]; then
#     echo "==================== $f ===================="
#     cat "$f"
#     echo ""  
#   fi
# done
# for f in *.sh; do
#   if [ -f "$f" ]; then
#     echo "==================== $f ===================="
#     cat "$f"
#     echo ""  
#   fi
# done

# export DATA_ROOTS="/fast_storage/intern/data/instruction_tuning/fastmri_acceleration_mat"
export DATA_ROOTS="/fast_storage/intern/data/instruction_tuning/brats_crossmodal_mat"

# export DATA_ROOTS="/fast_storage/intern/data/instruction_tuning/fastmri_acceleration_mat,/fast_storage/intern/data/instruction_tuning/brats_crossmodal_mat, /fast_storage/intern/data/instruction_tuning/oasis3_longitudinal_mat"
export RUN_DIR=$LOG_DATE
export TRAIN_ITER=1
GPU="2,3,4,5,6,7"
TRAIN_BATCH=48
LOG_FILE="$LOG_DATE/train_$(date +%Y%m%d_%H%M%S).log"

nohup $PYTHON_PATH train.py \
  --gpu $GPU \
  --train_batch $TRAIN_BATCH \
  --model_type "listfm_it" \
  --pretrained "/fast_storage/intern/code/share/checkpoint_3m.ckpt" \
  --from_scratch False \
  > /dev/null 2>&1 &

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

unset DATA_ROOTS
unset TRAIN_ITER
unset RUN_DIR

# echo "[INFO] Tail logs in: $LOG_DATE"
# find $LOG_DATE -type f -name "*.log" -exec tail -n 100 -f {} +
