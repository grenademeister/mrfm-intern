#!/bin/bash
USER_NAME=$(whoami)

echo "[INFO] Searching for running train.py processes owned by $USER_NAME..."

PIDS=$(ps -u "$USER_NAME" -f | grep -F "train.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "[INFO] No train.py processes found."
else
    echo "[INFO] Found the following train.py process IDs: $PIDS"
    for pid in $PIDS; do
        echo "[INFO] Killing process ID: $pid"
        kill -9 $pid
    done
    echo "[INFO] All train.py processes owned by $USER_NAME have been terminated."
fi

echo "[INFO] Searching for TensorBoard on port 6008..."
TB_PIDS=$(ps -u "$USER_NAME" -f | grep -F "tensorboard" | grep -F -- "--port 6008" | grep -v grep | awk '{print $2}')

if [ -z "$TB_PIDS" ]; then
    echo "[INFO] No TensorBoard processes found on port 6008."
else
    echo "[INFO] Found the following TensorBoard process IDs: $TB_PIDS"
    for pid in $TB_PIDS; do
        echo "[INFO] Killing TensorBoard process ID: $pid"
        kill -9 $pid
    done
    echo "[INFO] TensorBoard processes on port 6008 have been terminated."
fi
