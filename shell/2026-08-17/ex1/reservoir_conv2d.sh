#!/bin/bash
# 条件4: 局所 (K x K) x 逐次 (Tanaka & Tamukoh)
#
# NOTE: num_reservoirs=2 とすると、リザバー本数 (縦横 x 2 = 4 本) と 1 本あたりのノード数
#       (D/4) が bi_esn2d と一致し、両者の差が「K x K の窓を走査するか行/列全体を走査するか」に絞れる。
#       ただし leak rate は {0.1, 0.9} の 2 種になり、原論文の 5 種 {0.1, 0.3, 0.5, 0.7, 0.9} は
#       再現していない (5 種を保つには D が 10 の倍数である必要がある)
source "$(dirname "$0")/common.sh"

NUM_RESERVOIRS=2

# feature_dim = 2 * num_reservoirs * units なので、D から 1 リザバーあたりの units を逆算する
# (D=64 -> 16, D=1024 -> 256)。--expect_feature_dim が毎回検証する
extra_args() {
    EXTRA=(--num_reservoirs "${NUM_RESERVOIRS}" --units $(( $1 / (2 * NUM_RESERVOIRS) )))
}
run_condition reservoir_conv2d connectivity spectral_radius input_scaling
