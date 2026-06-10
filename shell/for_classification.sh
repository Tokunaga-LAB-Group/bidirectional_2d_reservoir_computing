#!/bin/bash
set -e

LOG_DIR="./logs/launcher"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/launch_${TIMESTAMP}.log"

echo "[INFO] Launching run_all_experiments.sh with nohup"
echo "[INFO] Launcher log: ${LOG_FILE}"

nohup bash shell/run_classification2.sh > "${LOG_FILE}" 2>&1 &

PID=$!

echo "[INFO] Process started"
echo "[INFO] PID = ${PID}"
echo "[INFO] Monitor with:"
echo "  tail -f ${LOG_FILE}"
