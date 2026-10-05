#!/bin/bash
# 条件3: 局所 (K x K) x 畳み込み
source "$(dirname "$0")/common.sh"

run_condition conv2d input_scaling
