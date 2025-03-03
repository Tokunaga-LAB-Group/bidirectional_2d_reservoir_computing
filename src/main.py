import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(os.getcwd())
sys.path.append("..")
warnings.filterwarnings("ignore")

import pprint

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

import models
import src


def main(args):
    # Reset seed
    src.utils.reset_seed(0)

    TRAIN_PATH = args.train_data_path
    TEST_PATH = args.test_data_path
    SAVE_PATH = src.utils.get_created_dir(args.save_path)

    H, W = args.resizes
    C = 3 if args.img_mode == "rgb" else 1

    # Train (normal images)
    train_files = src.data.path_to_files(TRAIN_PATH)
    train_imgs = src.data.files_to_imgs(train_files, img_mode=args.img_mode, resizes=(H, W))

    # Test
    test_files = src.data.path_to_files(TEST_PATH)
    test_imgs = src.data.files_to_imgs(test_files, img_mode="rgb", resizes=(H, W))

    # Model
    feature_extractor = models.model.get_feature_extractor(
        input_shape=(H, W, C),
        output_shape=(64, 64, 32 * 3),
        model_type=args.model,
    )

    # PaDiM
    # Calculate the statistics of the normal distribution
    normal_statistics = models.padim_framework.cal_statistics(
        normal_imgs=train_imgs,
        feature_extractor=feature_extractor,
        prioritize_memory=False,
    )

    # Calculate the anomaly scores
    anomaps = models.padim_framework.cal_mahalanobis_distance(
        input_imgs=test_imgs,
        normal_statistics=normal_statistics,
        feature_extractor=feature_extractor,
        prioritize_memory=False,
    )

    # Create the anomaly maps
    anomaps = keras.layers.Resizing(H, W, interpolation="bilinear")(anomaps)
    anomaps = anomaps.numpy().reshape(-1, H, W)
    anomaps = (anomaps - np.min(anomaps)) / (np.max(anomaps) - np.min(anomaps))

    # Save the anomaly maps
    for i, anomap in enumerate(anomaps):
        save_file = test_files[i].replace(TEST_PATH, os.path.join(SAVE_PATH, "anomaps"))
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        plt.imsave(save_file, anomap, cmap="gray", vmin=0.0, vmax=1.0)


if __name__ == "__main__":
    args = src.config.get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    pprint.pprint(args.__dict__)
    main(args)
