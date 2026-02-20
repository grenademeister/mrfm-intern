#!/bin/bash

LOG_DATE="/home/intern6/workspace/skeleton/data_prep/logs/oasis3_longitudinal/$(date +'%Y%m%d_%H%M%S')"

echo "[INFO] Removing nohup.out"
rm -rf nohup.out

PYTHON_PATH=/home/intern6/.conda/envs/mrifm/bin/python

echo "[INFO] Current directory: $(pwd)"

echo "[INFO] Running oasis3_longitudinal.py with nohup (background)"
mkdir -p "$(dirname "$LOG_DATE")"

nohup "$PYTHON_PATH" -u /home/intern6/workspace/skeleton/data_prep/downstream_tasks/oasis3_longitudinal.py > "${LOG_DATE}.log" 2>&1 &

echo "[INFO] Started PID: $!"

echo "[INFO] Tail logs in: ${LOG_DATE}.log"
