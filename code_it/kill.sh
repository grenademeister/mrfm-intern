#!/bin/bash
USER_NAME=$(whoami)
TAG="code_it_llm_ca"

echo "[INFO] Searching for running train.py processes owned by $USER_NAME with tag: $TAG..."

PIDS=$(ps -u "$USER_NAME" -f | grep -F "train.py" | grep -F -- "--tag ${TAG}" | grep -v grep | awk '{print $2}')

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
