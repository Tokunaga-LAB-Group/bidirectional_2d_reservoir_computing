#!/bin/bash
# 条件0: 特徴抽出なし。画像を平坦化してそのままリッジ回帰に入れる真の下限
#
# NOTE: feature_dim は入力次元 (MNIST/Fashion-MNIST 784、CIFAR-10 3072) で固定されるため
#       D 走査には参加しない。UNITS_LIST を 1 点に潰し、--expect_feature_dim も無効にする
source "$(dirname "$0")/common.sh"

UNITS_LIST=(0)
run_condition raw
