#!/bin/bash
# 実験1 (分類性能比較) の全条件で共通の設定。
#
# 個別スクリプト (bi_esn2d.sh など) はこれを source し、条件固有の差だけを与えて
# run_condition を呼ぶ。共通設定をここ 1 か所に置くのは、n_trials や N_seed が
# 条件ごとにずれると「全条件を同一の予算で探索した」という統制が崩れるため。
#
#   使い方:
#     run_condition bi_esn2d beta connectivity leaky spectral_radius
#
#   feature_dim が --units と一致しない条件 (reservoir_conv2d) は extra_args を上書きする。
#
#   環境変数:
#     GPU         使用する GPU ID (既定 0)
#     DATASETS    対象データセット (既定 mnist fashion_mnist cifar_10)
#     UNITS_LIST  走査する D (既定 128 512 2048)
#     WARM_START_ROOT  指定すると、同じ (dataset, model, D, P) の過去 study を読み込んで探索を継続する。
#                      n_trials は合計として解釈されるので、30 -> 50 に増やすと 20 試行だけ追加される

set -euo pipefail

PROJECT_ROOT="/workspace/nakanishi/bidirectional_2d_reservoir_computing"
PYTHON="/workspace/opt/conda_envs/torch/bin/python"
DATA_ROOT="/dataset/torchvision"

# 設計書 §3.2 の固定軸
#
# NOTE: パッチサイズは 1 (= パッチ化しない) に統一する。4x4 パッチ化を一部の条件にだけ
#       適用していると、conv 系 (生画素 H x W) とパッチ化系 (H/4 x W/4 格子) で空間解像度が
#       16 倍違い、精度・計算時間の比較が「到達範囲」ではなく「空間粒度」を測ってしまう。
#       実測でも、格子を揃えると積層時の優劣と CPU 実行時間の優劣がどちらも逆転した。
#       P=1 なら局所 (K x K 画素) / 十字 (行・列) / 全対全 が画素単位で純粋に表現される。
#       P=4 の結果は runs/exp1_patch4/ に退避してあり、Appendix で対照に使う
# NOTE: P はデータセットの画像サイズを割り切る必要がある (MNIST 28 -> 1,2,4,7,14 / CIFAR 32 -> 1,2,4,8,16)。
#       P=1 は「パッチ化しない」= 画素単位。P を持たない条件 (raw/fcl/conv2d/reservoir_conv2d) では無視される
PATCH="${PATCH:-1}"
N_LAYER=1
ACTIVATION="tanh"
# NOTE: 既定は 3 seed (本番)。探索・デバッグ段階は N_SEED=1 で上書きすると 1/3 の時間で回る。
#       seed 間 SD は 0.07-0.84 pt で fold 内分散に比べ小さく、1 seed でも平均値の順位は動かない
N_SEED="${N_SEED:-3}"
N_CV=5
# NOTE: n_trials=10 は TPESampler の n_startup_trials (既定 10) 以下なので、全試行が
#       事前分布からのランダムサンプリングになる。実測でも TPE / RandomSampler / startup を
#       下げた TPE の間に差は出なかった (5 seeds の平均 val acc がいずれも 0.514)。
#       条件間で完全に同一の手続きになるため、既定のまま使う
# NOTE: n_trials=30 で n_startup_trials (既定 10) を超えるため、後半 20 試行が実際に TPE の
#       推定に基づく探索になる。10 のままだと全試行がランダムサンプリングに退化していた
N_TRIALS="${N_TRIALS:-10}"
# NOTE: 位置符号は全条件で "none" に統一する。比較軸は受容野 (局所 K x K / 十字 / 全対全) と
#       重み共有の仕方であって、外部から与える位置情報はそのどちらでもない第三の要素になる。
#       sincos を attention 系にだけ入れると、同じく置換不変な positionwise_fcl との間で
#       扱いが揃わない (置換不変性の実測: positionwise_fcl 8.9e-08 / fcl 2.0e+00)。
#       位置符号の寄与自体は ablation として別途 sincos で測る
POS_ENCODING="${POS_ENCODING:-none}"
BATCH_SIZE=128
OPTUNA_METRIC="acc"
SAMPLER="${SAMPLER:-tpe}"
# NOTE: 探索次元が 1 (input_scaling だけ) の条件は grid で総当たりする。決定的で再現可能になり、
#       「探索予算が足りない」という論点が原理的に生じない。実測で 12 点格子と 30 試行 TPE の差は 0.07 pt
GRID_POINTS="${GRID_POINTS:-12}"
# NOTE: MedianPruner は should_prune() が呼ばれず実質無効なので none を指定する
PRUNER="none"

BASE_SEED=0
TPE_SEED=0

GPU="${GPU:-0}"
# NOTE: STL-10 は esn/bi_esn の系列長が 576 になり負荷が跳ね上がるため実験1では扱わない
IFS=' ' read -r -a DATASETS <<<"${DATASETS:-mnist fashion_mnist cifar_10}"
# NOTE: D は 8 の倍数に限る。bi_esn2d が D % 4、attention が D % n_head (最大 8) を要求するため
IFS=' ' read -r -a UNITS_LIST <<<"${UNITS_LIST:-128 512 2048}"

RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/exp1}"
LOG_ROOT="${PROJECT_ROOT}/logs/exp1"

# 条件ごとに D から追加の CLI 引数を決めるフック。既定は追加なし。
# feature_dim が --units と一致しない条件 (reservoir_conv2d) だけが上書きする。
extra_args() { EXTRA=(); }

# run_condition <model_type> [探索するパラメータ名...]
run_condition() {
    local model="$1"
    shift
    local tune=("$@")

    for dataset in "${DATASETS[@]}"; do
        for units in "${UNITS_LIST[@]}"; do
            extra_args "${units}"

            # NOTE: units=0 は「D を持たない条件」(raw) の印。検証を外し、パス名も D なしにする
            local dim_args=(--units "${units}" --expect_feature_dim "${units}")
            local dim_tag="D${units}"
            if [ "${units}" -eq 0 ]; then
                dim_args=()
                dim_tag="Dinput"
            fi

            # NOTE: P は保存先の名前に必ず含める。含めないと P を変えた実行が同じディレクトリを
            #       上書きしてしまい、P 間の比較ができなくなる
            local run_tag="${dim_tag}_P${PATCH}_L${N_LAYER}"
            local save_dir="${RUN_ROOT}/${dataset}/${model}/${run_tag}"
            local log_file="${LOG_ROOT}/${dataset}_${model}_${run_tag}.log"

            mkdir -p "${save_dir}" "$(dirname "${log_file}")"

            echo "[INFO] ${dataset} / ${model} / D=${units} (GPU ${GPU}) -> ${save_dir}"

            # NOTE: EXTRA は --units の後ろに置く。argparse は後勝ちなので、reservoir_conv2d だけ
            #       リザバー 1 本あたりの units に上書きできる
            "${PYTHON}" "${PROJECT_ROOT}/src/classification.py" \
                --dataset "${dataset}" \
                --model_type "${model}" \
                --data_root "${DATA_ROOT}" \
                ${dim_args[@]+"${dim_args[@]}"} \
                --n_layer "${N_LAYER}" \
                --patch_h "${PATCH}" --patch_w "${PATCH}" \
                --activation "${ACTIVATION}" \
                --pos_encoding "${POS_ENCODING}" \
                ${WARM_START_ROOT:+--warm_start_from "${WARM_START_ROOT}/${dataset}/${model}/${run_tag}"} \
                ${EXTRA[@]+"${EXTRA[@]}"} \
                --tune "${tune[@]}" \
                --N_seed "${N_SEED}" \
                --N_cv "${N_CV}" \
                --n_trials "${N_TRIALS}" \
                --batch_size "${BATCH_SIZE}" \
                --optuna_metric "${OPTUNA_METRIC}" \
                --sampler "${SAMPLER}" \
                --grid_points "${GRID_POINTS}" \
                --pruner "${PRUNER}" \
                --base_seed "${BASE_SEED}" \
                --tpe_seed "${TPE_SEED}" \
                --save_name "${save_dir}" \
                --gpu "${GPU}" \
                2>&1 | tee "${log_file}"
        done
    done
}
