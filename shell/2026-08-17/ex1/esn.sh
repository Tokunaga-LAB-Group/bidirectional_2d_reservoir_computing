#!/bin/bash
# 条件5: 全対全 x 逐次 (単方向)。系列長は N_h * N_w
source "$(dirname "$0")/common.sh"

run_condition esn connectivity leaky spectral_radius input_scaling
