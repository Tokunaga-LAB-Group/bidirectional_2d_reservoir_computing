import numpy as np
import tqdm


def cal_statistics(normal_imgs, feature_extractor, prioritize_memory=True):
    if prioritize_memory:
        feature_maps = feature_extractor.predict(normal_imgs, verbose=0)
    else:
        feature_maps = feature_extractor(normal_imgs)
        feature_maps = feature_maps.numpy()

    H = feature_maps.shape[1]
    W = feature_maps.shape[2]
    D = feature_maps.shape[3]

    embedding_vectors = feature_maps.reshape(-1, H * W, D)

    # Mean
    normal_mean = np.mean(embedding_vectors, axis=(0), keepdims=True)

    # Inverse of covariance
    normal_cov_inv = np.zeros((H * W, D, D), np.float32)
    for i in tqdm.tqdm(range(H * W), desc="Cal Cov"):
        normal_cov = np.cov(embedding_vectors[:, i, :], rowvar=False) + 0.01 * np.eye(D)
        normal_cov_inv[i] = np.linalg.inv(normal_cov)

    normal_statistics = (normal_mean, normal_cov_inv)

    return normal_statistics


def cal_mahalanobis_distance(input_imgs, normal_statistics, feature_extractor, prioritize_memory=True):
    if prioritize_memory:
        feature_maps = feature_extractor.predict(input_imgs, verbose=0)
    else:
        feature_maps = feature_extractor(input_imgs)
        feature_maps = feature_maps.numpy()

    H = feature_maps.shape[1]
    W = feature_maps.shape[2]
    D = feature_maps.shape[3]

    embedding_vectors = feature_maps.reshape(-1, H * W, D)

    normal_mean, normal_cov_inv = normal_statistics

    temp = embedding_vectors - normal_mean
    mahalanobis_dists = np.sqrt(np.einsum("nmk, nmk -> nm", np.einsum("nmd, mdk -> nmk", temp, normal_cov_inv), temp))

    dist_maps = mahalanobis_dists.reshape(-1, H, W, 1)

    return dist_maps
