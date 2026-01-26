#!/bin/bash

echo "[INFO] Searching for running train.py processes owned by intern2..."

PIDS=$(ps -u intern2 -f | grep train.py | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "[INFO] No train.py processes found."
else
    echo "[INFO] Found the following train.py process IDs: $PIDS"
    for pid in $PIDS; do
        echo "[INFO] Killing process ID: $pid"
        kill -9 $pid
    done
    echo "[INFO] All train.py processes owned by intern2 have been terminated."
fi
