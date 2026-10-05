#!/bin/bash
# パッチ化なし (P=1) での実験1 を 2 枚の GPU に振り分けて実行する。
#
# 振り分けは実測コスト (CIFAR-10 32x32, P=1 -> 1024 位置, D=128/512/2048, 3 データセット) に基づく:
#   self_attention 29.5h | bi_esn 13.2h | esn 8.8h | criss_cross 7.3h
#   reservoir_conv2d 5.4h | bi_esn2d 2.9h | positionwise_fcl 0.6h | conv2d 0.5h | fcl/raw ~0h
#
# 軽い条件と主要比較 (esn / bi_esn / bi_esn2d) を先に流し、最も重い self_attention を最後に置く。
# 22 時間ほどで RQ-A2 (esn < bi_esn < bi_esn2d) の判定に必要な条件が揃う。

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="${HERE}/../../../logs/exp1/launcher"
mkdir -p "${LOG}"
T=$(date +%Y%m%d_%H%M%S)

# GPU 0: 軽量条件 (~3.9h) -> self_attention (29.5h)  = 約 33.4h
GPU0=(raw fcl conv2d positionwise_fcl bi_esn2d self_attention)
# GPU 1: esn (8.8h) -> bi_esn (13.2h) -> criss_cross (7.3h) -> reservoir_conv2d (5.4h) = 約 34.7h
GPU1=(esn bi_esn criss_cross_attention reservoir_conv2d)

launch() {
    local gpu="$1"; shift
    # NOTE: $c はサブシェル側で展開させるため、シングルクォートで囲わずエスケープする
    nohup bash -c "for c in $*; do GPU=${gpu} bash \"${HERE}/\$c.sh\"; done; echo '[INFO] GPU ${gpu} 完了'" \
        >"${LOG}/p1_gpu${gpu}_${T}.log" 2>&1 &
    echo "[INFO] GPU ${gpu}: $*  PID=$!"
}
launch 0 "${GPU0[*]}"
launch 1 "${GPU1[*]}"
echo "[INFO] 進捗: tail -f ${LOG}/p1_gpu{0,1}_${T}.log"
