#!/bin/bash
# 条件6: 全対全 x 逐次 (双方向)。系列長は N_h * N_w
source "$(dirname "$0")/common.sh"

run_condition bi_esn connectivity leaky spectral_radius input_scaling
