#!/usr/bin/env bash
set -euo pipefail

USER_NAME=$(whoami)
CONDA_ENV_NAME=$CONDA_DEFAULT_ENV
PYTHON_PATH=/home/$USER_NAME/.conda/envs/$CONDA_ENV_NAME/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="/home/intern4/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
DATA_ROOT="/fast_storage/intern/data/data_curation"
OUT_DIR="/home/intern4/fm2026/mrfm-intern/code_it_llm_ca/dapt/outputs"
LOG_DIR="/home/intern4/fm2026/mrfm-intern/code_it_llm_ca/dapt/logs"
LOG_FILE="${LOG_DIR}/qwen_lora_dapt.log"

GPU="4,5"
export CUDA_VISIBLE_DEVICES="${GPU}"

mkdir -p "${LOG_DIR}"
echo "[INFO] GPU: ${GPU}"
echo "[INFO] Log: ${LOG_FILE}"

# nohup $PYTHON_PATH "${SCRIPT_DIR}/prepare_dapt_data.py" \
#   --data_roots \
#   "${DATA_ROOT}/brats2023_gli" \
#   "${DATA_ROOT}/brats2023_men" \
#   "${DATA_ROOT}/isles2022" \
#   "${DATA_ROOT}/wmh" \
#   --out "${OUT_DIR}/mri_captions.jsonl" \
#   > "${LOG_FILE}" 2>&1 &

# sleep 5

nohup $PYTHON_PATH "${SCRIPT_DIR}/train_lora_dapt.py" \
  --model_path "${MODEL_PATH}" \
  --data_path "${OUT_DIR}/mri_captions.jsonl" \
  --output_dir "${OUT_DIR}/lora" \
  --max_length 256 \
  --per_device_batch_size 8 \
  --grad_accum_steps 4 \
  --lr 2e-4 \
  --epochs 3 \
  --dtype fp16 \
  >> "${LOG_FILE}" 2>&1 &

echo "[INFO] DAPT started in background."
