#!/bin/bash
# 条件1: 空間混合なし。画像を平坦化して固定ランダム全結合 (Extreme Learning Machine 相当)
source "$(dirname "$0")/common.sh"

run_condition fcl input_scaling
