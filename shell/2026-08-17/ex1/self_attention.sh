#!/bin/bash
# 条件9 (Appendix): 全対全 x attention
#
# NOTE: nn.MultiheadAttention を固定重みにしただけの構成。ヘッド数は 1 (CCNet 原典に合わせる)、
#       softmax の温度は 1/sqrt(d_head) が実装から自動的に決まるため探索対象は beta のみ
#
# NOTE: 計算量が O((N_h * N_w)^2) なので STL-10 (パッチ後 24x24 = 576 位置) では扱えないが、
#       MNIST / Fashion-MNIST (7x7 = 49 位置) と CIFAR-10 (8x8 = 64 位置) は問題ない
source "$(dirname "$0")/common.sh"

run_condition self_attention input_scaling
