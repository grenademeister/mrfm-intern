#!/bin/bash

LOG_DATE="/home/intern6/workspace/skeleton/data_prep/logs/brats2023_gli_crossmodal/$(date +'%Y%m%d_%H%M%S')"

echo "[INFO] Removing nohup.out"
rm -rf nohup.out

PYTHON_PATH=/home/intern6/.conda/envs/mrifm/bin/python

echo "[INFO] Current directory: $(pwd)"

echo "[INFO] Running brats2023_gli_crossmodal.py with nohup (background)"
mkdir -p "$(dirname "$LOG_DATE")"

nohup "$PYTHON_PATH" -u /home/intern6/workspace/skeleton/data_prep/downstream_tasks/brats2023_gli_crossmodal.py > "${LOG_DATE}.log" 2>&1 &

echo "[INFO] Started PID: $!"

echo "[INFO] Tail logs in: ${LOG_DATE}.log"
