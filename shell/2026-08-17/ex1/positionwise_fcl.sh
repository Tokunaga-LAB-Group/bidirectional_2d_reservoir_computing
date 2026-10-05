#!/bin/bash
# 条件2: 各位置に同一の固定ランダム射影。bi_esn2d から再帰結合だけを取り除いた対照
source "$(dirname "$0")/common.sh"

run_condition positionwise_fcl input_scaling
