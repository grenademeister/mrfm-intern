#!/bin/bash

echo "[INFO] Searching for running train.py processes..."

PIDS=$(ps -ef | grep train.py | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "[INFO] No train.py processes found."
else
    echo "[INFO] Found the following train.py process IDs: $PIDS"
    for pid in $PIDS; do
        echo "[INFO] Killing process ID: $pid"
        kill -9 $pid
    done
    echo "[INFO] All train.py processes have been terminated."
fi


PIDS=$(ps -ef | grep test_wholebrain.py | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "[INFO] No test_wholebrain.py processes found."
else
    echo "[INFO] Found the following test_wholebrain.py process IDs: $PIDS"
    for pid in $PIDS; do
        echo "[INFO] Killing process ID: $pid"
        kill -9 $pid
    done
    echo "[INFO] All test_wholebrain.py processes have been terminated."
fi


PIDS=$(ps -ef | grep test_raw.py | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "[INFO] No test_raw.py processes found."
else
    echo "[INFO] Found the following test_raw.py process IDs: $PIDS"
    for pid in $PIDS; do
        echo "[INFO] Killing process ID: $pid"
        kill -9 $pid
    done
    echo "[INFO] All test_raw.py processes have been terminated."
fi
