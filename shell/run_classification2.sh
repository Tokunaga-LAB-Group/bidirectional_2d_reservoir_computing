#!/bin/bash
set -e

#######################################
# Global settings
#######################################
DATASETS=(
  # "cifar_10"
  # "mnist"
  "stl_10"
)
MODEL_TYPES=(
  "esn"
  "bi_esn"
  "bi_esn2d"
)
UNITS=(
  128
  256
  512
  1024
)

N_CV=5
N_SEED=3
N_TRIALS=30

SAVE_PATH="./results/nm2026_2"
LOG_PATH="./logs"

PYTHON_SCRIPT="./src/classification.py"
#######################################

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

#######################################
# Loop
#######################################
for DATASET in "${DATASETS[@]}"; do
  for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    for UNIT in "${UNITS[@]}"; do

      SAVE_NAME="${TIMESTAMP}_${DATASET}_${MODEL_TYPE}_units-${UNIT}"
      EXP_SAVE_PATH="${SAVE_PATH}/${SAVE_NAME}"
      EXP_LOG_PATH="${LOG_PATH}/${SAVE_NAME}"
      LOG_FILE="${EXP_LOG_PATH}/run.log"

      echo "=================================================="
      echo "[INFO] Dataset     : ${DATASET}"
      echo "[INFO] Model type  : ${MODEL_TYPE}"
      echo "[INFO] Units       : ${UNIT}"
      echo "[INFO] Save path   : ${EXP_SAVE_PATH}"
      echo "[INFO] Log file    : ${LOG_FILE}"
      echo "=================================================="

      mkdir -p "${EXP_SAVE_PATH}"
      mkdir -p "${EXP_LOG_PATH}"

      python "${PYTHON_SCRIPT}" \
        --gpu 0 \
        --dataset "${DATASET}" \
        --model_type "${MODEL_TYPE}" \
        --N_cv "${N_CV}" \
        --N_seed "${N_SEED}" \
        --n_trials "${N_TRIALS}" \
        --units "${UNIT}" \
        --patch_h 16 \
        --patch_w 16 \
        --save_name "${EXP_SAVE_PATH}" \
        --tune_connectivity \
        --tune_leaky \
        --tune_spectral_radius \
        --tune_beta \
        2>&1 | tee "${LOG_FILE}"

      echo "[INFO] Finished ${DATASET} / ${MODEL_TYPE} / units-${UNIT}"
      echo

    done
  done
done

echo "[ALL DONE] All experiments finished."