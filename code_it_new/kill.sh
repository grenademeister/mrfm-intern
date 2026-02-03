#!/bin/bash
USER_NAME=$(whoami)

echo "[INFO] Searching for running train.py processes owned by $USER_NAME..."

TRAIN_PIDS=$(ps -u $USER_NAME -f | grep train.py | grep -v grep | awk '{print $2}')

if [ -z "$TRAIN_PIDS" ]; then
    echo "[INFO] No train.py processes found."
else
    echo "[INFO] Found the following train.py process IDs: $TRAIN_PIDS"
    for pid in $TRAIN_PIDS; do
        echo "[INFO] Killing process ID: $pid"
        kill -9 $pid
    done
    echo "[INFO] All train.py processes owned by $USER_NAME have been terminated."
fi

echo "[INFO] Searching for running tensorboard processes owned by $USER_NAME..."

TB_PIDS=$(pgrep -u "$USER_NAME" -f "tensorboard")

if [ -z "$TB_PIDS" ]; then
    echo "[INFO] No tensorboard processes found."
else
    echo "[INFO] Found the following tensorboard process IDs: $TB_PIDS"
    echo "[INFO] Killing all tensorboard processes owned by $USER_NAME..."
    pkill -9 -u "$USER_NAME" -f "tensorboard"
    echo "[INFO] All tensorboard processes owned by $USER_NAME have been terminated."
fi
