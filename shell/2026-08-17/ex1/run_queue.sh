#!/bin/bash
# (条件, データセット, D, P) の 1 セルを最小単位として作業キューに積み、空いた GPU から
# 順に取り出して実行する。
#
# 条件をまるごと GPU に割り当てる run_all.sh 方式だと、self_attention の D=2048 だけで
# 3 seed 24.5 h かかり、割り当てられた側の GPU がボトルネックになって全体が伸びる。
# 重いセルから先に配る (LPT スケジューリング) と makespan が理論下限に近づく。
#
#   CONDITIONS="esn bi_esn2d" PATCHES="1 2 4" UNITS="128 512" N_SEED=1 N_TRIALS=30 \
#     RUN_ROOT=/path/to/runs/exp1_p ./run_queue.sh
#
# 環境変数 (いずれも省略可):
#   CONDITIONS  対象条件 (既定 = 実験1 の全 10 条件)。"cond:128,512" でその条件だけ D を絞れる
#   DATASETS    対象データセット (既定 mnist fashion_mnist cifar_10)
#   UNITS       走査する D (既定 128 512 2048)
#   PATCHES     走査する P (既定 1)。P を取らない条件では 1 セルに畳む
#   GPUS        使う worker (既定 "0 1")。同じ GPU ID を複数書くと多重化する。
#               リザバーの走査は 1 ステップの行列積 (33 MMAC ~ 2.5 us) よりカーネル起動
#               (5-10 us) の方が長く、逐次依存のため 1 プロセス内では隠せない。実測では
#               4 多重で合計スループットが 2.93 倍になった。GPU あたり 3 本を既定とする。
#   SKIP_DONE   1 (既定) なら完了済みセルをキューから外す。中断からの再開用
#   N_SEED / N_TRIALS / RUN_ROOT / POS_ENCODING は common.sh にそのまま渡る
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/workspace/nakanishi/bidirectional_2d_reservoir_computing"
LAUNCH_LOG="${PROJECT_ROOT}/logs/exp1/launcher"
mkdir -p "${LAUNCH_LOG}"

GPUS="${GPUS:-0 1}"
CONDITIONS="${CONDITIONS:-raw fcl positionwise_fcl conv2d esn bi_esn bi_esn2d reservoir_conv2d criss_cross_attention self_attention}"
DATASETS="${DATASETS:-mnist fashion_mnist cifar_10}"
UNITS="${UNITS:-128 512 2048}"
PATCHES="${PATCHES:-1}"

TS=$(date +%Y%m%d_%H%M%S)
QUEUE="${LAUNCH_LOG}/queue_${TS}.txt"
LOCK="${LAUNCH_LOG}/queue_${TS}.lock"

# patch_sizes を引数に取る条件。これ以外は P を変えても同じ計算になるので 1 セルに畳む
HAS_PATCH=" positionwise_fcl esn bi_esn bi_esn2d criss_cross_attention self_attention hybrid "
# D を持たない条件 (画像をそのまま読み出しに入れる)
NO_UNITS=" raw "

# セルを (推定コスト, 条件, データセット, D, P) で生成し、コスト降順に並べる (LPT)。
# 推定コストは実測 (3 seed, 10 trials, D=512, P=1) の相対値。順序付けにしか使わないので粗くてよい
python3 - "${QUEUE}" <<PYEOF
import sys
# "cond" または "cond:128,512" (条件固有の D)。attention だけ D=2048 を外す用途
conds=[]
cond_units={}
for tok in """${CONDITIONS}""".split():
    name=tok
    if ":" in tok:
        name,us=tok.split(":",1)
        cond_units[name]=us.split(",")
    conds.append(name)
dss="""${DATASETS}""".split()
units="""${UNITS}""".split()
patches="""${PATCHES}""".split()
has_patch=set("""${HAS_PATCH}""".split())
no_units=set("""${NO_UNITS}""".split())

# D=512, P=1 の 1 セルあたり実測時間 (h) の相対値
W={"esn":1.33,"bi_esn":1.45,"bi_esn2d":0.54,"reservoir_conv2d":0.79,"conv2d":0.08,
   "positionwise_fcl":0.09,"fcl":0.03,"raw":0.01,"criss_cross_attention":0.83,"self_attention":3.07}
# D に対して 2 乗で効く条件 (再帰行列 D^2 / QKV 射影 D^2)
QUAD={"esn","bi_esn","bi_esn2d","reservoir_conv2d","criss_cross_attention","self_attention"}
# トークン数 N=(H/P)(W/P) に比例する条件。positionwise_fcl は N*(P^2 C)*D なので P に依らない
SCALE_P={"esn","bi_esn","bi_esn2d","criss_cross_attention","self_attention"}

cells=[]
for c in conds:
    ps = patches if c in has_patch else ["1"]
    us = ["0"] if c in no_units else cond_units.get(c, units)
    for ds in dss:
        for u in us:
            for p in ps:
                d=int(u) if u!="0" else 512
                cost=W.get(c,0.5)*((d/512)**2 if c in QUAD else d/512)
                if c in SCALE_P: cost/= int(p)**2
                cells.append((cost,c,ds,u,p))
cells.sort(key=lambda x:-x[0])
with open(sys.argv[1],"w") as f:
    for cost,c,ds,u,p in cells:
        f.write(f"{c} {ds} {u} {p}\n")
print(f"[INFO] 推定総計算量 {sum(c[0] for c in cells):.1f} h 相当 (3 seed, 10 trials 換算)", file=sys.stderr)
PYEOF

# 完了済みのセルをキューから外す。中断して投入し直したときに最初からやり直さずに済む。
# 「完了」= save_dir 配下の cv-*_seed-* が EXPECT_RUNS 個あり、全てに metrics.csv がある
if [ -n "${RUN_ROOT:-}" ] && [ "${SKIP_DONE:-1}" = "1" ]; then
    EXPECT_RUNS="${EXPECT_RUNS:-$(( ${N_SEED:-3} * ${N_CV:-5} ))}"
    : > "${QUEUE}.keep"
    skipped=0
    while read -r c ds u pp; do
        tag="D${u}_P${pp}_L${N_LAYER_TAG:-1}"
        [ "${u}" = "0" ] && tag="Dinput_P${pp}_L${N_LAYER_TAG:-1}"
        d="${RUN_ROOT}/${ds}/${c}/${tag}"
        n=$(ls -d "${d}"/cv-*_seed-*/metrics.csv 2>/dev/null | wc -l)
        if [ "${n}" -eq "${EXPECT_RUNS}" ]; then
            skipped=$((skipped+1))
        else
            printf "%s %s %s %s\n" "${c}" "${ds}" "${u}" "${pp}" >> "${QUEUE}.keep"
        fi
    done < "${QUEUE}"
    mv "${QUEUE}.keep" "${QUEUE}"
    [ "${skipped}" -gt 0 ] && echo "[INFO] 完了済み ${skipped} セルをスキップ (EXPECT_RUNS=${EXPECT_RUNS})"
fi

TOTAL=$(wc -l < "${QUEUE}")
echo "[INFO] ${TOTAL} セルをキューに投入: ${QUEUE}"

pop_cell() {
    flock "${LOCK}" bash -c '
        q="$1"
        line=$(head -n 1 "$q")
        [ -z "$line" ] && exit 1
        tail -n +2 "$q" > "$q.tmp" && mv "$q.tmp" "$q"
        echo "$line"
    ' _ "${QUEUE}"
}

worker() {
    local gpu="$1" cell cond dataset units patch left
    while cell=$(pop_cell); do
        read -r cond dataset units patch <<<"${cell}"
        left=$(wc -l < "${QUEUE}")
        echo "[$(date +%H:%M:%S)] GPU${gpu} <- ${cond} ${dataset} D=${units} P=${patch} (残り ${left})"
        # raw は UNITS_LIST を自前で (0) に潰すので D を渡さない
        if [ "${units}" = "0" ]; then
            GPU="${gpu}" DATASETS="${dataset}" PATCH="${patch}" bash "${HERE}/${cond}.sh"
        else
            GPU="${gpu}" DATASETS="${dataset}" UNITS_LIST="${units}" PATCH="${patch}" bash "${HERE}/${cond}.sh"
        fi
        echo "[$(date +%H:%M:%S)] GPU${gpu} done ${cond} ${dataset} D=${units} P=${patch} (exit $?)"
    done
    echo "[$(date +%H:%M:%S)] GPU${gpu} キューが空になりました"
}

if [[ "${1:-}" == "--fg" ]]; then
    for g in ${GPUS}; do worker "${g}" & done
    wait
    echo "[INFO] 全セル完了"
    exit 0
fi

ENVPASS="N_SEED='${N_SEED:-}' N_TRIALS='${N_TRIALS:-}' RUN_ROOT='${RUN_ROOT:-}' POS_ENCODING='${POS_ENCODING:-}'"
for g in ${GPUS}; do
    nohup bash -c "$(declare -f pop_cell worker); HERE='${HERE}'; QUEUE='${QUEUE}'; LOCK='${LOCK}';
        export QUEUE LOCK HERE;
        $(for v in N_SEED N_TRIALS RUN_ROOT POS_ENCODING WARM_START_ROOT; do
            [ -n "${!v:-}" ] && echo "export ${v}='${!v}';"
          done)
        worker ${g}" >"${LAUNCH_LOG}/queue_gpu${g}_${TS}.log" 2>&1 &
    echo "[INFO] GPU ${g} worker PID=$!"
done

echo "[INFO] 進捗:  tail -f ${LAUNCH_LOG}/queue_gpu0_${TS}.log"
echo "[INFO] 残り:  wc -l ${QUEUE}"
