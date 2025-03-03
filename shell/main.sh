#!/bin/bash

# Training data
TRAIN_DATA_PATH="MVTec/bottle/train/good"
TEST_DATA_PATH="MVTec/bottle/test"

# Save name
SAVE_NAME="YYYY-MM-DD_XXXX"
SAVE_PATH="./results"
LOG_PATH="./logs"

# Make log directories
mkdir -p "${LOG_PATH}/${SAVE_NAME}"

nohup python ./src/main.py \
    --gpu 0 \
    --save_path "${SAVE_PATH}/${SAVE_NAME}" \
    --resizes 256 256 \
    --img_mode "rgb" \
    --model "v2" \
    --train_data_path "${TRAIN_DATA_PATH}" \
    --test_data_path "${TEST_DATA_PATH}" \
    >"${LOG_PATH}/${SAVE_NAME}/main.log" &
