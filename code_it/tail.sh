clear

LOG_DIR=logs
echo "[INFO] Finding last log in: $LOG_DIR"
LAST_LOG=$(find $LOG_DIR -type f -name "*.log" | sort | tail -n 1)
echo "[INFO] Printing last log: $LAST_LOG"
echo "========================================"
cat "$LAST_LOG"
