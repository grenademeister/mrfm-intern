clear

LOG_DATE=/home/juhyung/data/fm2026/log/log_downstream_recon
echo "[INFO] Tail logs in: $LOG_DATE"
find $LOG_DATE -type f -name "*.log" -exec tail -n 100 -f {} +
