# Bidirectional 2D Reservoir Computing

**English** | [日本語](README.ja.md)

## TL;DR

- **Training-free image anomaly detection.** Bidirectional 2D reservoir computing (BiRC2D) extracts features with fixed, randomly initialized echo state networks (ESNs), so the feature extractor needs no training.
- **PaDiM framework.** Fits a Gaussian per position to the features of normal images, then scores test images by Mahalanobis distance to produce pixel-level anomaly maps.
- **Two model versions.** `v1` is the original BiRC2D (NOLTA 2024). `v2` adds hierarchical pooling (NOLTA 2026). A ResNet-50 baseline (`cnn`) is also included.
- **Quick start:**
  ```bash
  pip install -r requirements.txt
  python src/main.py --gpu 0 --save_path results/test \
      --train_data_path MVTec/bottle/train/good --test_data_path MVTec/bottle/test
  ```

## Papers

This repository implements the following two papers. Select the version with the `--model` argument.

| `--model` | Paper | Architecture |
| --- | --- | --- |
| `v1` | K. Nakanishi and T. Tokunaga, "Bidirectional 2D reservoir computing for image anomaly detection without any training," *Nonlinear Theory and Its Applications, IEICE*, vol. 15, no. 4, pp. 838–850, 2024. [[J-STAGE]](https://www.jstage.jst.go.jp/article/nolta/15/4/15_838/_article) [[ResearchGate]](https://www.researchgate.net/publication/384470607_Bidirectional_2D_reservoir_computing_for_image_anomaly_detection_without_any_training) | Splits the image into 64×64, 32×32 and 16×16 grids of patches (three scales). Each scale passes through a stack of 10 BiESN2D layers. The outputs are upsampled and concatenated. |
| `v2` (default) | K. Nakanishi, R. Ishibashi, R. Takeyama and T. Tokunaga, "Hierarchical Pooling-Enhanced Bidirectional 2D Reservoir Computing for Edge-Efficient Image Anomaly Detection," *Nonlinear Theory and Its Applications, IEICE*, vol. 17, no. 1, pp. 93–107, 2026. [[J-STAGE]](https://www.jstage.jst.go.jp/article/nolta/17/1/17_93/_article/-char/ja/) | Five BiESN2D blocks alternate with 2×2 max pooling, doubling the width at each block (32 → 512). Features at the 64×64, 32×32 and 16×16 resolutions go through a fixed random projection, then are upsampled and concatenated. |
| `cnn` | Baseline: PaDiM with an ImageNet-pretrained ResNet-50 | Layers `conv2_block3_out`, `conv3_block4_out` and `conv4_block6_out` are concatenated, then 128 channels are randomly sampled. |

## Repository structure

```
.
├── models/
│   ├── modules.py          # Patches2Vectors, BiESN, BiESN2D layers
│   ├── model.py            # Feature extractors (v1 / v2 / cnn)
│   └── padim_framework.py  # Gaussian fitting and Mahalanobis distance
├── src/
│   ├── main.py             # Entry point
│   ├── config.py           # Command-line arguments
│   ├── data.py             # Image loading
│   └── utils.py            # Seed setting and helpers
├── shell/main.sh           # Example run script
├── notebooks/main.ipynb    # Interactive demo with anomaly map visualization
└── requirements.txt
```

## Installation

You need Python 3.10 or 3.11. The code uses `match` statements, which require Python 3.10+. `tensorflow-addons` (used for its ESN layer) only supports TensorFlow 2.13–2.15, which limits the Python version to 3.11 or lower.

```bash
git clone https://github.com/Tokunaga-LAB-Group/bidirectional_2d_reservoir_computing.git
cd bidirectional_2d_reservoir_computing

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU support on Linux, install `tensorflow[and-cuda]==2.15.1` instead of `tensorflow==2.15.1`. To run the notebook, also install `jupyter`.

## Dataset

The examples use the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset. Any directory of images works: images are loaded recursively from the given directory, so subdirectories are fine.

```
MVTec/
└── bottle/
    ├── train/
    │   └── good/          # Normal images (--train_data_path)
    └── test/              # Test images (--test_data_path)
        ├── good/
        ├── broken_large/
        └── ...
```

## Usage

Run from the repository root.

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

You can also edit and run `shell/main.sh`. It runs `src/main.py` in the background with `nohup` and writes its log to `logs/<SAVE_NAME>/main.log`.

```bash
bash shell/main.sh
```

### Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--gpu` | str | **required** | GPU ID(s) to use. The value is set as `CUDA_VISIBLE_DEVICES`, e.g. `0` or `0,1`. |
| `--save_path` | str | **required** | Output directory. It is created if it does not exist. |
| `--train_data_path` | str | **required** | Directory of normal images used to fit the Gaussian distributions. |
| `--test_data_path` | str | **required** | Directory of images to compute anomaly maps for. |
| `--resizes` | int int | `256 256` | Input size `H W`. Images must be square. `256 256` is recommended, because both models assume features at 64×64, 32×32 and 16×16. |
| `--img_mode` | str | `rgb` | Color mode of the input images: `rgb` or `gray`. |
| `--model` | str | `v2` | Feature extractor: `v1`, `v2` or `cnn`. See [Papers](#papers). |

The reservoir hyperparameters (number of layers, connectivity, leak rate, spectral radius, etc.) are set in `models/model.py` under `get_feature_extractor`.

### Output

Anomaly maps are saved as grayscale PNG images. They keep the directory structure of `--test_data_path`:

```
<save_path>/anomaps/good/000.png
<save_path>/anomaps/broken_large/000.png
...
```

Scores are min–max normalized to [0, 1] across the whole test set. Brighter pixels are more anomalous.

## Citation

If you use this code, please cite the paper for the version you used:

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

## License

See [LICENSE](LICENSE).
