import argparse


def get_args():
    # NOTE: 選択肢を extractors のレジストリから引くため、ここで import する
    #       (src.data など torch に依存しないモジュールを import しただけで torch が読まれるのを避ける)
    import networks

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, required=True, help="Select available GPU.")
    parser.add_argument("--save_path", type=str, required=True, help="Save path.")
    parser.add_argument("--resizes", type=int, nargs=2, default=[256, 256], help="Image size. H: int, W: int.")
    parser.add_argument("--img_mode", type=str, default="rgb", choices=["rgb", "gray"], help="Image mode.")

    parser.add_argument(
        "--model",
        type=str,
        default="birc2d_base_v2",
        choices=networks.list_extractors(),
        help="Select feature extractor.",
    )
    parser.add_argument(
        "--output_shape",
        type=int,
        nargs=3,
        default=[96, 64, 64],
        help="Feature map shape of the extractor. D: int, H: int, W: int. (resnet_50 では無視される)",
    )
    parser.add_argument(
        "--random_dim_size",
        type=int,
        default=128,
        help="Number of channels PaDiM keeps after random dimensionality reduction. 0 以下で間引きを無効化。",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument("--train_data_path", type=str, required=True, help="Load all train data.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Load all test data.")

    return parser.parse_args()
