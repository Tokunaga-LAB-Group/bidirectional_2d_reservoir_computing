#!/bin/bash
# 混合構成: 空間混合 (bi) とチャネル混合 (fc) を交互に積んだ場合の効果を見る
#
# NOTE: bi_esn2d は 1 層で受容野が画像全体になるため単純な積層で改善しない
#       (実測 CIFAR-10 で L=1 50.81 -> L=4 49.92)。Transformer の attention + FFN と同じ発想で
#       チャネル混合を挟むと劣化が止まるか、を 15 runs で確認する
#
# NOTE: 構造パラメータは実験1 で選ばれた中央値に固定し、beta のみ探索する。
#       構成の違いだけを見たいので、探索由来の分散を抑える
source "$(dirname "$0")/common.sh"

BLOCKS="${BLOCKS:-bi-fc-bi-fc}"
N_LAYER=$(awk -F- '{print NF}' <<<"${BLOCKS}")

extra_args() {
    EXTRA=(--blocks "${BLOCKS}" --connectivity 0.22 --leaky 0.85 --spectral_radius 0.90)
}
run_condition hybrid
