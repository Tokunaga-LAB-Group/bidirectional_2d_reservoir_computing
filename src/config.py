import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, required=True, help="Select available GPU.")
    parser.add_argument("--save_path", type=str, required=True, help="Save path.")
    parser.add_argument("--resizes", type=int, nargs=2, default=[256, 256], help="Image size. H: int, W: int.")
    parser.add_argument("--img_mode", type=str, default="rgb", choices=["rgb", "gray"], help="Image mode.")

    parser.add_argument("--model", type=str, default="v2", choices=["v1", "v2", "cnn"], help="Select model.")

    parser.add_argument("--train_data_path", type=str, required=True, help="Load all train data.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Load all test data.")

    return parser.parse_args()
