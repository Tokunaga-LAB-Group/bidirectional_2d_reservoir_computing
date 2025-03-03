import glob
import os
from typing import List, Literal, Tuple

import numpy as np
from natsort import natsorted
from numpy.typing import NDArray
from PIL import Image


def path_to_files(img_path: str) -> List[str]:
    img_files = natsorted(glob.glob(os.path.join(img_path, "**", "*.*"), recursive=True))
    return img_files


def files_to_imgs(
    img_files: List[str],
    img_mode: Literal["rgb", "gray"] = "rgb",
    resizes: Tuple[int, int] = (256, 256),
) -> NDArray[np.float32]:

    H, W = resizes

    match img_mode:
        case "rgb":
            mode = "RGB"
            C = 3
        case "gray":
            mode = "L"
            C = 1
        case _:
            raise ValueError("Select img_mode: rgb or gray.")

    imgs = [np.array(Image.open(file).convert(mode).resize((W, H), Image.BILINEAR), np.float32) for file in img_files]
    imgs = np.array(imgs, np.float32).reshape(-1, H, W, C) / 255.0

    return imgs
