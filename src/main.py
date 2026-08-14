import os
import sys
import warnings

sys.path.append(os.getcwd())
sys.path.append("..")
warnings.filterwarnings("ignore")

import pprint

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import networks
import src


# (B, H, W, C) の numpy 画像を PyTorch の (B, C, H, W) tensor にする
def to_tensor(imgs):
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()


def main(args):
    # Reset seed
    src.utils.reset_seed(args.seed)

    TRAIN_PATH = args.train_data_path
    TEST_PATH = args.test_data_path
    SAVE_PATH = src.utils.get_created_dir(args.save_path)

    H, W = args.resizes
    C = 3 if args.img_mode == "rgb" else 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # Train (normal images)
    train_files = src.data.path_to_files(TRAIN_PATH)
    train_imgs = to_tensor(src.data.files_to_imgs(train_files, img_mode=args.img_mode, resizes=(H, W)))

    # Test
    test_files = src.data.path_to_files(TEST_PATH)
    test_imgs = to_tensor(src.data.files_to_imgs(test_files, img_mode=args.img_mode, resizes=(H, W)))

    # Model
    feature_extractor = networks.build_extractor(
        args.model,
        input_shape=(C, H, W),
        output_shape=tuple(args.output_shape),
        seed=args.seed,
    )

    # PaDiM のランダム次元削減は、削減後の次元が元の次元より小さいときだけ意味がある
    random_dim_size = args.random_dim_size if args.random_dim_size > 0 else None
    if random_dim_size is not None and random_dim_size >= feature_extractor.output_dim:
        print(
            f"[INFO] random_dim_size ({random_dim_size}) >= extractor output_dim "
            f"({feature_extractor.output_dim}); skip the random dimensionality reduction."
        )
        random_dim_size = None

    padim = networks.PaDiM(feature_extractor, random_dim_size=random_dim_size, seed=args.seed).to(device)
    print(f"[INFO] embedding dim: {padim.embedding_dim} (extractor output_dim: {feature_extractor.output_dim})")

    # Calculate the statistics of the normal distribution
    padim.fit(train_imgs, batch_size=args.batch_size)

    # Calculate the anomaly scores
    dist_maps = padim.score(test_imgs, batch_size=args.batch_size)

    # Create the anomaly maps
    anomaps = F.interpolate(dist_maps.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)
    anomaps = anomaps.squeeze(1).cpu().numpy()
    anomaps = (anomaps - anomaps.min()) / (anomaps.max() - anomaps.min())

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
