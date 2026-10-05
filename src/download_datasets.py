"""torchvision 形式で公開データセットを共有ストレージへダウンロードする。

保存先は torchvision の ``root`` にそのまま渡せる構成になる。

    /dataset/torchvision/
        MNIST/raw/...
        FashionMNIST/raw/...
        cifar-10-batches-py/...
        stl10_binary/...

使い方 (torchvision が入った環境で実行する)::

    /workspace/opt/conda_envs/torch/bin/python src/download_datasets.py
    /workspace/opt/conda_envs/torch/bin/python src/download_datasets.py --root /path/to/root --datasets mnist cifar10
"""

import argparse

from torchvision import datasets

DEFAULT_ROOT = "/dataset/torchvision"

# データセット名 -> (torchvision クラス, download 時に必要な split 引数のリスト)
DATASETS = {
    "mnist": (datasets.MNIST, [{"train": True}, {"train": False}]),
    "fashion_mnist": (datasets.FashionMNIST, [{"train": True}, {"train": False}]),
    "cifar10": (datasets.CIFAR10, [{"train": True}, {"train": False}]),
    # STL-10 は 1 回の download で train/test/unlabeled を含む tar が展開される。
    "stl10": (datasets.STL10, [{"split": "train"}, {"split": "test"}, {"split": "unlabeled"}]),
}


def download(name: str, root: str) -> None:
    dataset_cls, split_kwargs = DATASETS[name]

    for kwargs in split_kwargs:
        dataset = dataset_cls(root=root, download=True, **kwargs)
        print(f"[{name}] {kwargs} -> {len(dataset)} samples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT, help=f"保存先ルート (default: {DEFAULT_ROOT})")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()

    for name in args.datasets:
        print(f"=== {name} ===")
        download(name, args.root)


if __name__ == "__main__":
    main()
