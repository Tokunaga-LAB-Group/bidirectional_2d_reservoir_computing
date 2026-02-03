#!/bin/bash
set -e

#######################################
# Global settings
#######################################
DATASETS=(
  "cifar_10"
  # "mnist"
  )
MODEL_TYPES=("esn" "bi_esn" "bi_esn2d")

N_CV=5
N_SEED=3
N_TRIALS=30

SAVE_PATH="./results"
LOG_PATH="./logs"

PYTHON_SCRIPT="./src/classification.py"
#######################################

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

#######################################
# Loop
#######################################
for DATASET in "${DATASETS[@]}"; do
  for MODEL_TYPE in "${MODEL_TYPES[@]}"; do

    SAVE_NAME="${TIMESTAMP}_${DATASET}_${MODEL_TYPE}_units-1024"
    EXP_SAVE_PATH="${SAVE_PATH}/${SAVE_NAME}"
    EXP_LOG_PATH="${LOG_PATH}/${SAVE_NAME}"
    LOG_FILE="${EXP_LOG_PATH}/run.log"

    echo "=================================================="
    echo "[INFO] Dataset     : ${DATASET}"
    echo "[INFO] Model type  : ${MODEL_TYPE}"
    echo "[INFO] Save path   : ${EXP_SAVE_PATH}"
    echo "[INFO] Log file    : ${LOG_FILE}"
    echo "=================================================="

    mkdir -p "${EXP_SAVE_PATH}"
    mkdir -p "${EXP_LOG_PATH}"

    python "${PYTHON_SCRIPT}" \
      --gpu 7 \
      --dataset "${DATASET}" \
      --model_type "${MODEL_TYPE}" \
      --N_cv "${N_CV}" \
      --N_seed "${N_SEED}" \
      --n_trials "${N_TRIALS}" \
      --units 1024 \
      --save_name "${EXP_SAVE_PATH}" \
      --tune_connectivity \
      --tune_leaky \
      --tune_spectral_radius \
      --tune_beta \
      2>&1 | tee "${LOG_FILE}"

    echo "[INFO] Finished ${DATASET} / ${MODEL_TYPE}"
    echo
  done
done

echo "[ALL DONE] All experiments finished."
