#!/usr/bin/env bash
set -euo pipefail

USER_NAME=$(whoami)


run_dir="/home/$USER_NAME/fm2026/mrfm-intern/code_it_new/logs"
if [[ -n "${RUN_DIR:-}" ]]; then
  run_dir="$RUN_DIR"
fi

if [[ ! -d "$run_dir" ]]; then
  echo "Log dir not found: $run_dir" >&2
  exit 1
fi

latest_dir=$(ls -1dt "$run_dir"/*/ 2>/dev/null | head -n 1 || true)
if [[ -z "$latest_dir" ]]; then
  echo "No log directories found in: $run_dir" >&2
  exit 1
fi

rm -rf "$latest_dir"
