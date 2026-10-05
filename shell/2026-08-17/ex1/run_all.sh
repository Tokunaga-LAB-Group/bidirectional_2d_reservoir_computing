#!/bin/bash
# 実験1 の全条件を 2 枚の GPU に振り分けて実行する。
#
#   ./run_all.sh            # バックグラウンド実行
#   ./run_all.sh --fg       # フォアグラウンド実行 (動作確認用)
#
# 振り分けは実測スループット (CIFAR-10, batch 128, A5000) に基づく所要時間の見積もり。
# D = 64/128/256/512/1024 の 5 点 x 3 データセット x 10 trials x 15 folds で、
#
#   reservoir_conv2d 3.7 h | bi_esn 1.7 h | esn 1.0 h | bi_esn2d 0.9 h
#   criss_cross 0.7 h | conv2d 0.4 h | positionwise_fcl 0.3 h | fcl 0.0 h   (合計 8.7 h)
#
# reservoir_conv2d はパッチ化せず生画像解像度で走るため飛び抜けて重く、単独で全体の 4 割強を占める。

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_LOG="${HERE}/../../../logs/exp1/launcher"
mkdir -p "${LAUNCH_LOG}"

# GPU 0 : reservoir_conv2d 単独 (~3.7 h)
GPU0_CONDITIONS=(reservoir_conv2d)
# GPU 1 : 残り全部 (~5.0 h)
GPU1_CONDITIONS=(bi_esn esn bi_esn2d criss_cross_attention conv2d positionwise_fcl fcl)

run_group() {
    local gpu="$1"
    shift
    for cond in "$@"; do
        GPU="${gpu}" bash "${HERE}/${cond}.sh"
    done
    echo "[INFO] GPU ${gpu} の全条件が終了しました"
}

if [[ "${1:-}" == "--fg" ]]; then
    run_group 0 "${GPU0_CONDITIONS[@]}" &
    PID0=$!
    run_group 1 "${GPU1_CONDITIONS[@]}" &
    PID1=$!
    wait "${PID0}" "${PID1}"
    echo "[INFO] 実験1 完了"
    exit 0
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
nohup bash -c "$(declare -f run_group); HERE='${HERE}'; run_group 0 ${GPU0_CONDITIONS[*]}" \
    >"${LAUNCH_LOG}/gpu0_${TIMESTAMP}.log" 2>&1 &
PID0=$!
nohup bash -c "$(declare -f run_group); HERE='${HERE}'; run_group 1 ${GPU1_CONDITIONS[*]}" \
    >"${LAUNCH_LOG}/gpu1_${TIMESTAMP}.log" 2>&1 &
PID1=$!

echo "[INFO] GPU 0 (${GPU0_CONDITIONS[*]}) PID=${PID0}"
echo "[INFO] GPU 1 (${GPU1_CONDITIONS[*]}) PID=${PID1}"
echo "[INFO] 進捗の確認:"
echo "  tail -f ${LAUNCH_LOG}/gpu0_${TIMESTAMP}.log"
echo "  tail -f ${LAUNCH_LOG}/gpu1_${TIMESTAMP}.log"
