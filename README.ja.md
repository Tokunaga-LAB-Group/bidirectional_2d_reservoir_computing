# Bidirectional 2D Reservoir Computing

[English](README.md) | **日本語**

## TL;DR

- **学習不要の画像異常検知。** 双方向2次元リザバーコンピューティング（BiRC2D）は、ランダムに初期化して固定したエコーステートネットワーク（ESN）で特徴を抽出します。そのため、特徴抽出器の学習は必要ありません。
- **PaDiMフレームワーク。** 正常画像の特徴から位置ごとにガウス分布を推定し、テスト画像とのマハラノビス距離を求めて、ピクセル単位の異常マップを作ります。
- **2つのモデルバージョン。** `v1` は元のBiRC2D（NOLTA 2024）、`v2` は階層的プーリングを加えたもの（NOLTA 2026）です。比較用にResNet-50のベースライン（`cnn`）も入っています。
- **クイックスタート:**
  ```bash
  pip install -r requirements.txt
  python src/main.py --gpu 0 --save_path results/test \
      --train_data_path MVTec/bottle/train/good --test_data_path MVTec/bottle/test
  ```

## 論文

このリポジトリは次の2本の論文を実装しています。どちらを使うかは `--model` 引数で選びます。

| `--model` | 論文 | アーキテクチャ |
| --- | --- | --- |
| `v1` | K. Nakanishi and T. Tokunaga, "Bidirectional 2D reservoir computing for image anomaly detection without any training," *Nonlinear Theory and Its Applications, IEICE*, vol. 15, no. 4, pp. 838–850, 2024. [[J-STAGE]](https://www.jstage.jst.go.jp/article/nolta/15/4/15_838/_article) [[ResearchGate]](https://www.researchgate.net/publication/384470607_Bidirectional_2D_reservoir_computing_for_image_anomaly_detection_without_any_training) | 画像を64×64、32×32、16×16のパッチグリッド（3スケール）に分割します。各スケールは10層のBiESN2Dを通り、出力をアップサンプリングして結合します。 |
| `v2`（デフォルト） | K. Nakanishi, R. Ishibashi, R. Takeyama and T. Tokunaga, "Hierarchical Pooling-Enhanced Bidirectional 2D Reservoir Computing for Edge-Efficient Image Anomaly Detection," *Nonlinear Theory and Its Applications, IEICE*, vol. 17, no. 1, pp. 93–107, 2026. [[J-STAGE]](https://www.jstage.jst.go.jp/article/nolta/17/1/17_93/_article/-char/ja/) | 5つのBiESN2Dブロックと2×2のmax poolingを交互に重ね、ブロックごとに次元数を2倍にします（32 → 512）。64×64、32×32、16×16の解像度の特徴を固定のランダム射影に通し、アップサンプリングして結合します。 |
| `cnn` | ベースライン: ImageNetで事前学習したResNet-50を使ったPaDiM | `conv2_block3_out`、`conv3_block4_out`、`conv4_block6_out` の出力を結合し、128チャンネルをランダムに選びます。 |

## ディレクトリ構成

```
.
├── models/
│   ├── modules.py          # Patches2Vectors, BiESN, BiESN2D レイヤー
│   ├── model.py            # 特徴抽出器（v1 / v2 / cnn）
│   └── padim_framework.py  # ガウス分布の推定とマハラノビス距離
├── src/
│   ├── main.py             # エントリーポイント
│   ├── config.py           # コマンドライン引数
│   ├── data.py             # 画像の読み込み
│   └── utils.py            # シード設定などの補助関数
├── shell/main.sh           # 実行スクリプトの例
├── notebooks/main.ipynb    # 異常マップを可視化するデモ
└── requirements.txt
```

## インストール

Python 3.10または3.11が必要です。コードで `match` 文を使っているため3.10以上が必要です。また、ESNレイヤーに使っている `tensorflow-addons` はTensorFlow 2.13〜2.15にしか対応していないため、Pythonは3.11以下に限られます。

```bash
git clone https://github.com/Tokunaga-LAB-Group/bidirectional_2d_reservoir_computing.git
cd bidirectional_2d_reservoir_computing

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

LinuxでGPUを使う場合は、`tensorflow==2.15.1` の代わりに `tensorflow[and-cuda]==2.15.1` を入れてください。ノートブックを動かす場合は `jupyter` も入れてください。

## データセット

例では [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) データセットを使っています。画像は指定したディレクトリから再帰的に読み込むので、サブディレクトリがあっても構いません。どんな画像ディレクトリでも使えます。

```
MVTec/
└── bottle/
    ├── train/
    │   └── good/          # 正常画像（--train_data_path）
    └── test/              # テスト画像（--test_data_path）
        ├── good/
        ├── broken_large/
        └── ...
```

## 使い方

リポジトリのルートで実行します。

```bash
python src/main.py \
    --gpu 0 \
    --save_path results/bottle_v2 \
    --resizes 256 256 \
    --img_mode rgb \
    --model v2 \
    --train_data_path MVTec/bottle/train/good \
    --test_data_path MVTec/bottle/test
```

`shell/main.sh` を編集して実行することもできます。このスクリプトは `nohup` で `src/main.py` をバックグラウンド実行し、ログを `logs/<SAVE_NAME>/main.log` に書き出します。

```bash
bash shell/main.sh
```

### 引数

| 引数 | 型 | デフォルト | 説明 |
| --- | --- | --- | --- |
| `--gpu` | str | **必須** | 使うGPUのID。`CUDA_VISIBLE_DEVICES` に設定されます（例: `0`、`0,1`）。 |
| `--save_path` | str | **必須** | 出力先ディレクトリ。なければ作成されます。 |
| `--train_data_path` | str | **必須** | ガウス分布の推定に使う正常画像のディレクトリ。 |
| `--test_data_path` | str | **必須** | 異常マップを計算する画像のディレクトリ。 |
| `--resizes` | int int | `256 256` | 入力サイズ `H W`。正方形のみ対応です。どちらのモデルも64×64、32×32、16×16の特徴を前提にしているため、`256 256` を推奨します。 |
| `--img_mode` | str | `rgb` | 入力画像のカラーモード: `rgb` または `gray`。 |
| `--model` | str | `v2` | 特徴抽出器: `v1`、`v2`、`cnn`。[論文](#論文)を参照してください。 |

リザバーのハイパーパラメータ（層数、結合率、リーク率、スペクトル半径など）は `models/model.py` の `get_feature_extractor` で設定しています。

### 出力

異常マップはグレースケールのPNG画像として保存されます。ディレクトリ構成は `--test_data_path` と同じです。

```
<save_path>/anomaps/good/000.png
<save_path>/anomaps/broken_large/000.png
...
```

スコアはテストセット全体で0〜1にmin–max正規化しています。明るいピクセルほど異常度が高いことを表します。

## 引用

このコードを使う場合は、使ったバージョンに対応する論文を引用してください。

```bibtex
@article{nakanishi2024birc2d,
  title   = {Bidirectional 2D reservoir computing for image anomaly detection without any training},
  author  = {Nakanishi, Keiichi and Tokunaga, Terumasa},
  journal = {Nonlinear Theory and Its Applications, IEICE},
  volume  = {15},
  number  = {4},
  pages   = {838--850},
  year    = {2024},
  doi     = {10.1587/nolta.15.838}
}

@article{nakanishi2026hierarchical,
  title   = {Hierarchical Pooling-Enhanced Bidirectional 2D Reservoir Computing for Edge-Efficient Image Anomaly Detection},
  author  = {Nakanishi, Keiichi and Ishibashi, Ryosuke and Takeyama, Ren and Tokunaga, Terumasa},
  journal = {Nonlinear Theory and Its Applications, IEICE},
  volume  = {17},
  number  = {1},
  pages   = {93--107},
  year    = {2026},
  url     = {https://www.jstage.jst.go.jp/article/nolta/17/1/17_93/_article/-char/ja/}
}
```

## ライセンス

[LICENSE](LICENSE) を参照してください。
