# 引き継ぎ書 — BiRC2D 実験1（分類性能比較）

最終更新 2026-10-05。前任セッションからの引き継ぎ。
**実験1 の主要部分は完了している。** 残っているのは attention 2 条件の再実行と 3 seed 本番。

---

## 0. 最初に読むべきこと

- 本ファイル（方法論の確定事項、運用手順、落とし穴）
- `runs/exp1_tables.md` — 集計表
- 整理済みの結果ページ: https://claude.ai/artifact/SZ7Qwho1aRSc6FXyTVJgbz

**重要**: 以下の「確定した方法論」は、長い議論と実測を経て決着したもの。
変更する前に必ず根拠（各項目に記載）を確認すること。安易に戻すと過去の議論を繰り返す。

---

## 1. 研究の狙い

訓練不要・固定ランダム重みの画像特徴抽出器として **BiRC2D（`bi_esn2d`）** を提案し、
同じ枠組みの他モジュールと比較する。学習するのは**読み出しのリッジ回帰だけ**。

比較軸は **受容野**（局所 K×K / 十字 / 全対全 / 位置ごと）と **重み共有の仕方**。

| 条件 | 中身 | 受容野 |
|---|---|---|
| `raw` | 特徴抽出なし。画像を平坦化して読み出しへ | — |
| `positionwise_fcl` | 各位置に同一の dense。空間混合なし | パッチ内のみ |
| `fcl` | 平坦化 → dense（位置ごとに別の重み） | 全体 |
| `conv2d` | 固定ランダム畳み込み K=3 | K×K |
| `reservoir_conv2d` | K×K 窓をリザバーで走査（Tanaka & Tamukoh, NOLTA 2022） | K×K |
| `esn` | 全パッチを 1 系列に平坦化して走査 | 全体（長い走査） |
| `bi_esn` | 同上を順逆 2 方向 | 全体 |
| **`bi_esn2d`** | **行・列を順逆 4 方向に走査（提案）** | **行・列** |
| `self_attention` | `nn.MultiheadAttention` を固定重み化、head=1 | 全対全 |
| `criss_cross_attention` | CCNet の criss-cross attention を固定重み化 | 十字 |

関連研究として **CIRCLE（arXiv:2606.27095）** が BiRC2D を引用して継続学習に応用している。
BiRC2D から 4 点を変更（ランダム畳み込み stem、leaky ReLU、Kaiming 初期化、空間連結 + 昇次元射影）。
査読で「なぜ tanh か」を問われる可能性が高い（§3 の活性化の項を参照）。

---

## 2. 現在の状態

### 実験の完成状況

| 条件 | 探索設定 | 保存先 | 状態 |
|---|---|---|---|
| `esn` `bi_esn` `bi_esn2d` `reservoir_conv2d` | 50 試行 TPE、範囲拡張後 | `runs/exp1_wide` | **完了**（90 セル） |
| `raw` `fcl` `positionwise_fcl` `conv2d` | 12 点グリッド、`input_scaling` 上限 100 | `runs/exp1_grid` | **完了**（48 セル） |
| `self_attention` `criss_cross_attention` | 30 試行 TPE、**旧範囲** | `runs/exp1_p` | **未更新**、D=2048 空欄 |

すべて **1 seed × 5 fold**。本番は 3 seed が必要。

### 主要な結果

最大 D（attention は D=512）での `bi_esn2d` との一対比較。差は bi_esn2d − 相手。

| 相手 | MNIST | Fashion | CIFAR-10 |
|---|---:|---:|---:|
| `esn` | +0.96 *** | +0.73 * | +3.19 *** |
| `bi_esn` | +0.84 ** | +0.23 * | +3.11 *** |
| `reservoir_conv2d` | +10.09 *** | +2.42 *** | +2.67 *** |
| `conv2d` | +9.02 *** | +2.67 *** | **+0.11 n.s.** |
| `fcl` | +2.27 *** | +0.52 * | +17.80 *** |
| `positionwise_fcl` | +22.01 *** | +6.87 *** | +11.65 *** |
| `criss_cross_attention` (D512) | +29.99 *** | +17.12 *** | +25.69 *** |
| `self_attention` (D512) | +43.05 *** | +27.82 *** | +30.47 *** |
| `raw` | +12.19 *** | +6.16 *** | +24.42 *** |

`*** p<.001  ** p<.01  * p<.05`

**27 件中 26 件で bi_esn2d が有意に優位。** 唯一の例外が CIFAR-10 の `conv2d`（64.62 対 64.51、p=0.57）。

`bi_esn2d` の値: MNIST 97.93、Fashion 87.01、CIFAR-10 64.62（いずれも D=2048、P=2）。

### 旧い run（参照用、現行設定ではない）

| ディレクトリ | 内容 |
|---|---|
| `runs/exp1` | 3 seed / 10 trials / P=1 固定 / 位置符号あり。10 trials は TPE が起動せず全ランダムだった |
| `runs/exp1_p` | 1 seed / 30 trials / P 走査。attention はここの値を使っている |
| `runs/exp1_p50` | 1 seed / 50 trials（warm start で +20）/ 旧 HP 範囲 |
| `runs/exp1_patch4` | P=4 固定の旧実験 |
| `runs/exp1_noscale` | `input_scaling` 導入前 |
| `exp1_attention_v1` `exp1_convK_tuned_v1` `exp1_p1_glorot_win` `exp3_hybrid` | 本会話より前の実験 |

---

## 3. 確定した方法論（変更前に根拠を確認すること）

### D を揃える

線形読み出しに入る特徴次元 `D` を全条件で統一し、D ∈ {128, 512, 2048} を走査する。
**訓練されるパラメータは D × クラス数で全条件同一。** 内部構成は連結後が D になるよう決まり、
`bi_esn2d` は D/4 ノードのリザバー 4 本（縦の順逆・横の順逆）。
`--expect_feature_dim` が起動時に検証する。

**根拠**: ノード数で見れば全条件が D で揃う。パラメータ数で揃えようとすると
`conv2d` が D=116,736、`positionwise_fcl` が D=1,050,624 という非現実的な値になり基準として機能しない。
固定ランダム重み数は条件により 2,048〜16,779,264（D=2048, MNIST）と 8000 倍の開きがあるが、
これらは訓練対象ではない。

### P は外側の格子

パッチサイズ P ∈ {1, 2, 4} を全条件で走査し、各条件が最良の P を選べる。TPE の探索対象ではない。

**根拠**: 最良 P は P=2 が最多（48 通りの (条件, D) で P=2 が 24 回、P=4 が 16 回、P=1 が 8 回）。
以前の P=1 固定は多くの条件で最良点を外していた。
トークンあたりの入力次元は C·P²、トークン数は HW/P²。パッチ化は前者を上げる代わりに後者を失うので
最適は内点になる。D=2048 の全 6 組（3 データセット × esn/bi_esn2d）で P=2 が最良。

### β は解析的に総当たり

正則化係数 β は `[1e-6, 1e3]` の 19 点を**同一の正規方程式から解く**ので追加コストがほぼゼロ。
探索対象にしない（`--beta_grid`）。

**根拠**: 特徴抽出を 1 回行えば `ZTZ` を使い回せる。以前 β を Optuna で探索していたときは
範囲 `[1e-5, 1e-3]` が狭すぎて実質無効だった。

### 位置符号なし（全条件）

`POS_ENCODING=none` を全条件で明示的に渡す。

**根拠**: 比較軸は受容野と重み共有であり、外部から与える位置情報はそのどちらでもない第三の要素。
attention 系だけに入れると、同じく置換不変な `positionwise_fcl` との扱いが揃わない
（置換不変性の実測: `positionwise_fcl` 8.9e-08 / `fcl` 2.0e+00）。
なお **`criss_cross` は位置符号なしでも置換不変ではない**（einsum の添字 h, w が格子構造を参照する。
実測 1.18e-02）。`self_attention` は完全に置換不変（1.19e-07）。

位置符号の寄与自体は大きい（P=1 の MNIST で self_attention +66.6 pt）が、P を上げると消える
（P=14 で +2.3 pt）。`POS_ENCODING=sincos` で ablation 可能。

### 活性化は tanh（全条件）

**根拠**: ReLU / leaky ReLU とハイパーパラメータを同一格子（`input_scaling` 4 点 ×
`spectral_radius` 3 点 × `leaky` 4 点）で探索して比較し、**5 件すべてで tanh が最良**。

| dataset | 条件 | tanh | relu | leaky_relu |
|---|---|---:|---:|---:|
| MNIST | bi_esn2d | **95.90** | 95.32 | 95.40 |
| MNIST | esn | **93.90** | 92.56 | 未測定 |
| CIFAR-10 | bi_esn2d | **57.32** | 56.27 | 56.21 |
| CIFAR-10 | esn | **53.99** | 未測定 | 未測定 |

順伝播系でも ReLU は 10 件中 1 件も最良にならず（`criss_cross` の MNIST で −13.7 pt）。
**空間平均プーリング（GAP）が非負出力の符号情報を失うため**と考えられる。
CIRCLE が ReLU で成功しているのは GAP ではなく空間連結を使うからという仮説が立つ。

`networks/modules.py` に `reservoir_act`（`"tanh"` / `"relu"` / `"leaky_relu"`）を実装済み。
**既定は tanh で、既定経路の出力は実装前と差 0.000e+00**（既存結果は無効にならない）。
D=512 のみ・粗い格子・CV なしの簡易測定なので、論文に載せるなら本条件で回し直すこと。

### 探索予算

| 探索次元 | 方式 | 対象 |
|---|---|---|
| 4（`connectivity` `leaky` `spectral_radius` `input_scaling`） | TPE **50 試行** | `esn` `bi_esn` `bi_esn2d` |
| 3（`leaky` なし） | TPE 50 試行 | `reservoir_conv2d` |
| 1（`input_scaling` のみ） | **12 点グリッドを総当たり** | `fcl` `conv2d` `positionwise_fcl` `self_attention` `criss_cross_attention` |
| 0 | なし | `raw` |

**根拠**:
- 10 試行は `n_startup_trials=10` 以下なので TPE が起動せず全ランダムだった。30 に上げると
  `esn` が MNIST で **+7.37 pt** 改善。50 にするとさらに `esn` +0.44、`bi_esn` +0.50、`bi_esn2d` +0.08。
- 最良試行の分布は後半に偏る（30 試行のとき `esn` は 85% が後半 15 回、[25-29] 区間が最多）。
  打ち切りの影響があるので 50 にした。
- 1 次元条件は 12 点グリッドと 30 試行 TPE の差が **0.07 pt**。決定的・再現可能になり、
  探索予算の議論が原理的に不要になる。

**注意**: `reservoir_conv2d` だけ `leaky` を探索せず 0.9 固定。統制の不整合であり、
`reservoir_conv2d` が不利になっている。直すなら `shell/2026-08-17/ex1/reservoir_conv2d.sh` の
`run_condition` に `leaky` を追加する。

### HP の探索範囲

`src/classification.py` の `suggest_params()` と `param_distributions()` の**両方**に定義がある。

| パラメータ | 範囲 | 分布 |
|---|---|---|
| `spectral_radius` | **[0.0, 1.0]** | 一様 |
| `leaky` | **[0.01, 1.0]** | 一様 |
| `connectivity` | **[0.01, 1.0]** | 対数一様 |
| `input_scaling` | **[1e-2, 1e2]** | 対数一様 |

**根拠**:
- `spectral_radius` の上限 1.0 はエコー状態性の条件 ρ(W) < 1 由来の**理論的制約**。
  ρ > 1 まで広げた掃引では 4 件中 3 件で悪化（MNIST `esn` は ρ=1.4 で −5.7 pt 崩壊）。
  旧範囲 `[0.5, 0.99]` で上限に 53〜56% 張り付いていたのは **edge of stability の再現**であり、
  範囲の切り過ぎではなかった。下限を 0 まで開けたのは CIFAR-10 の `esn` が旧下限 0.5 で最良だったため。
- `leaky` は旧範囲 `[0.5, 1.0]` の下限に 20〜24% 張り付いていた。範囲拡張後、`esn` の最良値の **93%**、
  `bi_esn` の **82%** が旧下限の外（中央値 0.136、0.193）。`bi_esn2d` は 0.484 で 53% のみ。
  **`leaky=0` は状態が更新されず特徴が全ゼロに縮退する**（実測 std 0.0、実効ランク 0.0）ので下限は 0.01。
- `connectivity` は張り付きがない（下限 4〜8%、上限 11〜12%）。一様分布に変えると選択値の集中域
  （中央値 0.24〜0.27）の探索が薄くなるので対数一様を維持。
- `input_scaling` の上限を 10 → 100 にしたのは、位置符号を外して以降 attention 系と
  `positionwise_fcl` の最良値が上限に張り付いたため（`criss_cross` 69%、`self_attention` 57%、
  `positionwise_fcl` 45%）。リザバー系は上限 10 でも張り付き 0〜1% で最適域が内点にあったので、
  **`runs/exp1_wide` を回し直す必要はない**。

---

## 4. 運用手順

### 実験の起動

`shell/2026-08-17/ex1/run_queue.sh` が (条件, データセット, D, P) の 1 セルを最小単位として
作業キューに積み、空いた worker が `flock` で排他的に取り出す。

```bash
cd /workspace/nakanishi/bidirectional_2d_reservoir_computing/shell/2026-08-17/ex1
R=/workspace/nakanishi/bidirectional_2d_reservoir_computing/runs

# リザバー系を 50 試行で
GPUS="0 0 0 1 1 1" N_SEED=1 N_TRIALS=50 POS_ENCODING=none \
RUN_ROOT=$R/exp1_wide \
CONDITIONS="esn bi_esn bi_esn2d reservoir_conv2d" PATCHES="1 2 4" UNITS="128 512 2048" \
./run_queue.sh

# 1 次元条件をグリッドで。attention だけ D を絞る記法も使える
GPUS="0 0 0 1 1 1" N_SEED=1 SAMPLER=grid GRID_POINTS=12 POS_ENCODING=none \
RUN_ROOT=$R/exp1_grid \
CONDITIONS="raw fcl positionwise_fcl conv2d criss_cross_attention:128,512 self_attention:128,512" \
PATCHES="1 2 4" UNITS="128 512 2048" \
./run_queue.sh
```

| 環境変数 | 意味 |
|---|---|
| `GPUS` | 使う worker。**同じ GPU ID を複数書くと多重化**（既定 `"0 1"`） |
| `CONDITIONS` | 対象条件。`"cond:128,512"` でその条件だけ D を絞れる |
| `PATCHES` / `UNITS` / `DATASETS` | 走査する P / D / データセット |
| `N_SEED` / `N_TRIALS` / `SAMPLER` / `GRID_POINTS` / `POS_ENCODING` / `RUN_ROOT` | `common.sh` にそのまま渡る |
| `SKIP_DONE` | 1（既定）なら完了済みセルをキューから外す。**中断からの再開に使える** |

### GPU の多重化

**GPU あたり 3 本（合計 6）が既定。** リザバーの走査は 1 ステップの行列積（33 MMAC ≒ 2.5 µs）より
カーネル起動（5〜10 µs）の方が長く、逐次依存のため 1 プロセス内では隠せない。
実効スループットは **5.95 / 6.0**（ほぼ理論値）。メモリは 1 プロセス約 3.4 GB で 24 GB に余裕がある。

ただし `nvidia-smi` の `utilization.gpu` は「1 つでもカーネルが動いていた時間の割合」なので、
この状況でも 100% と表示される。実効スループットで判断すること。

### 探索の warm start

`--warm_start_from <RUN_ROOT>/<dataset>/<model>/<tag>` で過去の `study/trials.csv` を読み、
その試行を新しい study に登録してから探索を再開する（`n_trials` は**合計**として解釈）。
`WARM_START_ROOT` 環境変数で `run_queue.sh` から渡せる。

**制約**: 分布定義が変わると使えない（過去の試行が誤った値として登録される）。
HP 範囲を変えたときはゼロから回し直すこと。sampler の RNG 状態は復元しないので、
最初から n_trials で回した場合とは別の系列になる（「同じ予算で探索した」ことは保たれる）。

### 進捗確認

```bash
cd /workspace/nakanishi/bidirectional_2d_reservoir_computing
Q=$(ls -t logs/exp1/launcher/queue_*.txt | head -1)
echo "残り $(wc -l < $Q) セル、実行中 $(ps -eo args | grep -c '[c]lassification.py')"
grep -h "done " logs/exp1/launcher/queue_gpu*.log | grep -v "exit 0)"   # 異常終了
for d in runs/exp1_wide/*/*/*/; do n=$(ls -d $d/cv-*_seed-* 2>/dev/null | wc -l); [ "$n" -ne 5 ] && echo "$d -> $n"; done
```

---

## 5. 残っている作業（優先度順）

### 1) 3 seed 本番 — 最重要

現在すべて 1 seed × 5 fold。論文に載せるには 3 seed が必要。
6 並列で全条件 50 試行なら**約 42 時間**の見込み。

seed 間 SD は 0.07〜0.84 pt で fold 内分散より小さく、1 seed でも平均値の順位は動かないことを
確認済み（SE は 0.164 → 0.283）。順位は変わらない見込みだが、統計的信頼性のため必要。

### 2) attention 2 条件の再実行

現状だけ 30 試行 TPE・旧範囲（`input_scaling` 上限 10）。探索次元が 1 なので 12 点格子に揃えれば安価。
上限に 57〜69% 張り付いていたので、**上限 100 への拡張で伸びる可能性がある**。
3 seed 本番と同時に回すのが効率的。

### 3) attention の D=2048

主表に空欄が残る。CIFAR-10 で `conv2d` と拮抗している以上、埋めないと「D=2048 で最良」と言い切れない。
ただし D=512 時点で `bi_esn2d` から 25〜43 pt 離れており追いつく見込みは低い。

コストは `self_attention` が支配的（D=2048 の 3 セルで 3 seed 24.5 時間）。
演算量の内訳は `in_proj`（QKV 射影、$N \cdot 3D^2$）が 60%、`out_proj` 20%、
$N^2$ の項（スコアと AV）が合わせて 20%。「注意行列が重い」という直感は誤り。

**付録向けの観察**: `bi_esn2d` の固定重みは $4(D/4)^2 = D^2/4$、`self_attention` は $4D'^2$。
$D' = D/4$ で**厳密に一致**する（D=2048 の 1,050,624 と D=512 の 1,049,088）。
重み数を揃える観点では D≤512 が対応点にあたる。ただし主表の D 軸を打ち切る根拠にはならない。

### 4) `fcl` と `positionwise_fcl` の統合

P = 画像全体にすると**厳密に一致する**（特徴の差 0.000e+00、重みの形と値も同一、MNIST/CIFAR 両方で確認）。
独立条件ではなく P でつながった 1 つの族として示す方が受容野の枠組みと整合する。

### 5) CIFAR-10 における `conv2d` との同等性の扱い

精度で区別できず（p=0.57）、しかもコストで劣る。

| 条件 | 固定ランダム重み（D=2048, MNIST） | MAC/枚（D=2048, CIFAR） | 実行時間（D=2048） |
|---|---:|---:|---:|
| `conv2d` | 18,432 | 0.06 G | 11 分 |
| `bi_esn2d` | 1,050,624 | 0.27 G | 39 分 |
| `self_attention` | 16,779,264 | 21.48 G | — |

どの軸で主張を立てるかの判断が必要。MNIST（+9.02）と Fashion（+2.67）では大きく上回るので、
**データセット依存性として正面から扱う**のが誠実。

---

## 6. 主張に使える補助測定（実施済み）

いずれも簡易測定（サブセット・粗い格子・CV なし）。論文に載せるなら本条件で回し直すこと。

### 平行移動頑健性

MNIST で訓練は無シフト、テストのみ k 画素シフト（4 方向平均、0 埋め）。最適 HP を使用。

| 条件 | k=0 | k=4 | 劣化 |
|---|---:|---:|---:|
| `esn` | 97.40 | 93.04 | −4.36 |
| `bi_esn2d` | **97.87** | 85.35 | −12.52 |
| `conv2d` | 88.99 | 73.77 | −15.22 |
| `reservoir_conv2d` | 87.86 | 60.47 | −27.39 |
| `fcl` | 95.82 | 20.82 | **−75.00** |
| `raw` | 85.81 | 18.83 | −66.98 |
| `positionwise_fcl` | 76.23 | 75.32 | −0.91（置換不変） |

**走査に依存しながら conv より頑健。** 絶対精度では全シフト量で `bi_esn2d` が `conv2d` を上回る。
`fcl` は壊滅的で、特徴抽出がかえって平行移動を増幅している。

`leaky` が受容野の広さを調整するつまみとして働く。`bi_esn2d` で leaky 0.5〜0.8 のとき
無シフト精度は 97.91〜97.97 と平坦だが k=4 精度は 83.77〜86.01 と 2.2 pt 動く
（**精度をほぼ犠牲にせず頑健性を選べる**）。leaky を下げすぎると逆に脆くなる（0.1 で −17.9）。
ただし `esn` の方が頑健なので、2 次元化は頑健性を高めない。

### なぜ 2 次元化が効くのか

P=2 の MNIST で走査長は `esn` 196（14×14 を 1 本に平坦化）、`bi_esn2d` **14**（行・列を 4 方向）。
`esn` は記憶を保つため小さい `leaky` と 1 に近い `spectral_radius` を強く要求するが、
`bi_esn2d` は中庸の値で足りる。**同じ到達範囲をより緩い設定で達成している。**

### fcl と conv2d の相補性

`fcl` は MNIST 95.65 / CIFAR-10 46.82、`conv2d` は MNIST 88.90 / CIFAR-10 64.51 と正反対。
`fcl` は位置ごとに別の重みを持つが局所性がなく、`conv2d` はその逆。
**3 データセットすべてで両者以上なのは `bi_esn2d` だけ。**

CIFAR-10 をグレースケール化（解像度は保持）しても `conv2d` が `fcl` を 19.8 pt 上回るので、
逆転の原因は**チャネル数ではなく画像の性質**（MNIST は対象が中央に固定、CIFAR は位置も見えも多様）。
色情報の喪失への頑健性は `conv2d` −2.66 < `bi_esn2d` −5.2〜5.9 < `fcl` −7.02 < `positionwise_fcl` −7.88。

### 探索の収束

30 試行時点での「t 試行までの最良 val」の伸びと、最良試行番号の分布。

| 条件 | 探索次元 | 20→30 の伸び | 外挿 30→50 | 最良試行が後半 15 回 |
|---|---:|---:|---:|---:|
| `esn` | 4 | +0.89 | +1.04 | **85%** |
| `bi_esn` | 4 | +0.67 | +0.98 | 84% |
| `bi_esn2d` | 4 | +0.20 | +0.36 | 79% |
| `conv2d` | 1 | +0.03 | +0.15 | 58% |
| `fcl` | 1 | +0.02 | +0.04 | 53% |

1 次元条件は飽和済み。4 次元条件は後半に偏るので 50 試行にした。

---

## 7. 実装の検証状況

### attention 2 条件は原典と数値一致

| 条件 | 参照した原典 | 照合結果 |
|---|---|---|
| `self_attention` | PyTorch `nn.MultiheadAttention`（論文ではなく実装） | 最大差 **0.000e+00** |
| `criss_cross_attention` | `github.com/speedinghzl/CCNet` の `cc_attention/functions.py` | 最大差 **0.000e+00** |

CCNet は実際のソースを取得して逐語照合済み（`INF` 関数、各 `permute`/`view`、`softmax(dim=3)`、
`att_H`/`att_W` の切り出し、`out_H`/`out_W` の組み立て）。
`self_attention` は温度 1/√d_k 内蔵、head=1、初期化も PyTorch 既定と同分布であることを確認。

### 原典との意図的な差異（4 点）

1. **両条件に `W_embed` を追加** — `nn.MultiheadAttention` は出力次元が query の次元に固定されるため、
   D を他条件と揃えるのに必要（ViT の patch embedding 相当）。原典にない追加なので
   $W \sim U(-s, s)$ とし $s$ を探索対象にしている。
2. **criss-cross の残差 $\gamma(\text{out}_H + \text{out}_W) + x$ を再現しない** — $\gamma$ は `zeros(1)`
   初期化の学習パラメータで、訓練しない本研究では値の根拠がない（0 のままだと恒等写像）。
   他の全条件が残差を持たないため、残差の有無が受容野の比較に混ざるのを避けた。
3. **criss-cross の Q/K/V に bias がない** — 原典の `nn.Conv2d` は既定で bias を持つ。
   他の全条件と揃えるため省いた（影響は出力の 2.26%）。**意図しない見落としだったが、
   統制としては現状が正しいと判断した。論文に明記が必要。**
4. **attention ブロック直後に tanh を掛ける** — 原典にはない。コードのコメントは
   「層を重ねたときの表現力」を理由としているが、現在の実験はすべて 1 層なのでその理由は成立していない。

### Q/K ボトルネックは criss-cross を不利にしていない

原典は $D/8$。$D$ まで広げても改善しない（MNIST 61.09 → 61.10、CIFAR-10 36.88 → **32.50**）。
訓練しない Q/K では次元を上げると softmax が過度に尖るため、低次元が暗黙の平滑化として働く。
原典が $D/8$ とスケーリング無しを組み合わせているのは整合的な設計。

### 特徴テンソルの変形（参考）

全条件に共通する骨格は 画像 → パッチ格子 → 各条件の変換 → GAP → D 次元ベクトル。

```
入力画像        (B, C, H, W)
  ↓ Patchify (P×P の窓をチャネル方向にまとめる。情報は捨てない)
パッチ格子      (B, N_h, N_w, P²C)
  ↓ 各条件の変換
特徴マップ      (B, N_h, N_w, D)
  ↓ GAP  x.mean(dim=(1,2))
特徴ベクトル    (B, D)
```

`bi_esn2d` は横方向で `reshape(B·N_h, N_w, C)`、縦方向で `permute` 後 `reshape(B·N_w, N_h, C)` として
**バッチと空間の一方の軸をまとめて「系列の束」にする**。`esn` は `(B, N_h·N_w, C)` と 1 本に平坦化。
`self_attention` は `reshape` で格子を潰すので置換不変になるが、`criss_cross` は格子を保つので
置換不変にならない。

---

## 8. 落とし穴（前任が踏んだもの）

### 所要時間の見積りを繰り返し外した

前任は所要時間を **3 回連続で楽観的に外した**（20 時間 → 外れ、2.6 時間 → 外れ、2 時間 → 外れ）。
原因は「完了済みセルの平均時間を残りセルに当てはめる」こと。6 並列で同時に重いセルを引くと
1 本あたりが大きく遅くなり、平均値では捉えられない。
**実効スループット（プロセス時間/時）から計算し、幅を持って伝えること。**

参考: 実測の 1 セルあたり時間（6 並列、50 試行、1 seed × 5 fold）

| 条件 | D=2048 P=1 | D=2048 P=2 |
|---|---:|---:|
| `esn` | 9.15 h | 3.71 h |
| `bi_esn` | 8.76 h | 5.19 h |
| `reservoir_conv2d` | 8.79 h | — |
| `bi_esn2d` | **4.80 h** | **2.12 h** |

### 分布定義が 2 箇所にある

`src/classification.py` の `suggest_params()`（探索用）と `param_distributions()`（warm start 用）。
**ずれると warm start が誤った値を登録する。** 変更時は必ず両方を直し、以下で照合すること。

```python
import sys; sys.path.insert(0,'src')
import argparse, optuna, classification as C
args=argparse.Namespace(tune=["connectivity","leaky","spectral_radius","input_scaling"],
    model_type="bi_esn2d",dataset="mnist",patch_h=1,patch_w=1,units=512,connectivity=.1,
    leaky=.9,spectral_radius=.95,input_scaling=1.0,kernel_size=3,n_layer=1,n_head=1,
    blocks="bi-fc",num_reservoirs=2,activation="tanh",pos_encoding="none",grid_points=12)
seen={}
def obj(t): C.suggest_params(args,t); seen.update(t.distributions); return 0.0
optuna.create_study().optimize(obj,n_trials=1)
assert seen==C.param_distributions(args), "2 箇所の定義がずれている"
```

### 保存先に P を含めないと上書きされる

`common.sh` の `run_tag` は `D{units}_P{patch}_L{n_layer}`。P を含めないと
P を変えた実行が同じディレクトリを上書きして P 間の比較ができなくなる。

### `pgrep -f` / `pkill -f` が自分自身にマッチする

前任は自分のシェルを 2 回殺した。`ps -eo pid,args` で絞ってから `kill` すること。

### WebFetch の要約を信用しない

CCNet の実装も CIRCLE の活性化も、WebFetch の要約は**誤っていた**
（CIRCLE は tanh ではなく leaky ReLU。本文には tanh が 0 回、ReLU が 19 回）。
`pdftotext` や生ファイルの取得で**必ず原文を確認**すること。

### `import networks` が cwd 依存

`src/classification.py` は `sys.path.insert(0, Path(__file__).resolve().parents[1])` で解決済み。

---

## 9. 未解決の疑問（議論の余地があるもの）

- **CIFAR-10 で `conv2d` と区別できない**（p=0.57）。D=128/512 では `conv2d` が上回り D=2048 で逆転する。
  どの軸で主張するか未決。
- **Fashion-MNIST での優位が薄い**。D=2048 で `bi_esn2d` 対 `bi_esn` が +0.23 pt（p=0.0405）。
  27 セル中 3 件の例外はすべて Fashion で、いずれも `bi_esn` に負けている。
- **`reservoir_conv2d` の `leaky` が探索されていない**（0.9 固定）。統制の不整合。
- **attention ブロック直後の tanh** は原典にない追加で、1 層実験では理由が成立しない。外すべきか未決。
- **多層実験**（`n_layer` > 1）は未実施。前任との合意は「層を増やした実験も入れるが、解析は主に 1 層」。
- **訓練 CNN ベースライン**は「査読者に言われたらやる」で保留。

---

## 10. 環境

| 項目 | 値 |
|---|---|
| Python | `/workspace/opt/conda_envs/torch/bin/python` |
| データ | `/dataset/torchvision` |
| GPU | A5000 × 2（各 24 GB） |
| ブランチ | `dev_ver3` |

`git status` は未コミットの変更が多数ある状態（`networks/modules.py`、`src/classification.py`、
各 classifier、`shell/2026-08-17/`、分析スクリプト群）。コミットは前任が行っていない。
