#!/bin/bash
set -e

#######################################
# User settings
#######################################
DATASET="cifar_10"          # mnist | cifar10
MODEL_TYPE="esn"           # esn | bi_esn | bi_esn2d

N_CV=5
N_SEED=3
N_TRIALS=30

SAVE_NAME="2026-01-31_cifar10_esn"
SAVE_PATH="./results"
LOG_PATH="./logs"

PYTHON_SCRIPT="./src/classification.py"
#######################################

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Final paths
EXP_SAVE_PATH="${SAVE_PATH}/${SAVE_NAME}"
EXP_LOG_PATH="${LOG_PATH}/${SAVE_NAME}"
LOG_FILE="${EXP_LOG_PATH}/run_${TIMESTAMP}.log"

#######################################
# Make directories
#######################################
echo "[INFO] Creating directories..."
mkdir -p "${EXP_SAVE_PATH}"
mkdir -p "${EXP_LOG_PATH}"

#######################################
# Run
#######################################
echo "[INFO] Starting experiment"
echo "[INFO] Dataset     : ${DATASET}"
echo "[INFO] Model type  : ${MODEL_TYPE}"
echo "[INFO] N_CV        : ${N_CV}"
echo "[INFO] N_SEED      : ${N_SEED}"
echo "[INFO] N_TRIALS    : ${N_TRIALS}"
echo "[INFO] Save path   : ${EXP_SAVE_PATH}"
echo "[INFO] Log file    : ${LOG_FILE}"

nohup python "${PYTHON_SCRIPT}" \
    --dataset "${DATASET}" \
    --model_type "${MODEL_TYPE}" \
    --N_cv "${N_CV}" \
    --N_seed "${N_SEED}" \
    --n_trials "${N_TRIALS}" \
    --save_name "${EXP_SAVE_PATH}" \
    --tune_units \
    --tune_connectivity \
    --tune_leaky \
    --tune_spectral_radius \
    --tune_beta \
    > "${LOG_FILE}" 2>&1 &

PID=$!

#######################################
# After launch
#######################################
echo "[INFO] Process started successfully"
echo "[INFO] PID = ${PID}"
echo "[INFO] Monitor log with:"
echo "  tail -f ${LOG_FILE}"
