#!/bin/bash
# 条件8: 十字 (軸方向) x attention。CCNet (Huang et al., ICCV 2019) 原典準拠
#
# NOTE: Q/K は units//8 へのボトルネック、単一ヘッド、1/sqrt(d_k) スケーリング無し、出力射影無し。
#       softmax の温度は実装から自動的に決まるため探索対象を持たない (beta のみ)
#
# NOTE: 位置エンコーディングは既定の sincos を使う。attention は置換不変なので、
#       none にすると差の大部分が受容野ではなく位置盲によるものになり比較が成立しない
source "$(dirname "$0")/common.sh"

run_condition criss_cross_attention input_scaling
