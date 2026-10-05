#!/bin/bash
# 条件7: 十字 (軸方向) x 逐次。本研究の提案手法。系列長は N_h または N_w
source "$(dirname "$0")/common.sh"

run_condition bi_esn2d connectivity leaky spectral_radius input_scaling
