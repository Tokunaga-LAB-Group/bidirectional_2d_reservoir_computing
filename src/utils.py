import os
import warnings

warnings.filterwarnings("ignore")

import random
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# フォントの設定
plt.rcParams["font.size"] = 17
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
# 軸の設定
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.grid"] = True

import tensorflow as tf


def reset_seed(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_created_dir(*args: str) -> str:
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path


# 比較画像の表示
def save_compare_figure(
    f_save: str,
    title_list: List[str],
    img_list: List[NDArray[np.float32]],
    cmap_list: List[str],
    suptitle: str = None,
) -> None:
    if suptitle is None:
        suptitle = os.path.splitext(os.path.basename(f_save))[0]

    N_sub = len(title_list)

    plt.figure(figsize=(5 * N_sub, 5), facecolor="white")
    plt.suptitle(suptitle, fontsize=20)
    for idx, data in enumerate(zip(title_list, img_list, cmap_list)):
        title, img, cmap = data
        plt.subplot(1, N_sub, idx + 1)
        plt.title(title, fontsize=18)
        plt.axis("off")
        plt.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
    # plt.show()
    plt.savefig(f_save, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close()
    plt.clf()
