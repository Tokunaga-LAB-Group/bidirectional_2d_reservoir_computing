# TF/Keras
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Custom
from models import modules


def get_feature_extractor(input_shape, output_shape, model_type="cnn", **kwargs):
    match model_type:
        case "v1":
            feature_extractor = _get_feature_extractor_v1(
                input_shape=input_shape,
                output_shape=output_shape,
                sample_sizes=(64, 32, 16),
                N=10,
                connectivity=0.9,
                leaky=0.8,
                spectral_radius=0.9,
            )
        case "v2":
            feature_extractor = _get_feature_extractor_v2(
                input_shape=input_shape,
                output_shape=output_shape,
                N_block=5,
                N_subblock=1,
                sample_sizes=(64, 32, 16),
                base_filters=32,
                connectivity=0.7,
                leaky=0.9,
                spectral_radius=0.9,
            )
        case "cnn":
            feature_extractor = _get_feature_extractor_cnn(
                input_shape=input_shape,
                sample_sizes=(64, 32, 16),
                random_dim_size=128,
            )
        case _:
            raise ValueError("Select model: v1 or v2.")

    return feature_extractor


def _get_feature_extractor_v1(
    input_shape,
    output_shape,
    sample_sizes=(64, 32, 16),
    N=10,
    connectivity=0.9,
    leaky=0.8,
    spectral_radius=0.9,
    is_concat=True,
):
    H, W, _ = input_shape
    H_, W_, D_ = output_shape
    L = len(sample_sizes)
    D__ = D_ // L

    assert H == W, "Only square images are supported."
    assert H_ == W_, "Only square images are supported."

    inputs = keras.layers.Input(shape=input_shape)

    x = inputs
    features = []
    for ss in sample_sizes:
        ps = int(H / ss)

        # (H, W, C) -> (ss, ss, ps*ps*C)
        y = modules.Patches2Vectors((ps, ps))(x)

        for _ in range(N):
            # (ss, ss, ps*ps*C) -> (ss, ss, D__)
            y = modules.BiESN2D(D__, connectivity, leaky, spectral_radius)(y)

        # (ss, ss, D__) -> (H_, W_, D__)
        if is_concat:
            y = keras.layers.UpSampling2D((int(H_ / ss), int(W_ / ss)), interpolation="bilinear")(y)
        features.append(y)

    if is_concat:
        x = keras.layers.Concatenate()(features)
    else:
        x = features

    outputs = x

    model = keras.Model(inputs, outputs)

    return model


def _get_feature_extractor_v2(
    input_shape,
    output_shape,
    N_block=5,
    N_subblock=10,
    sample_sizes=(64, 32, 16),
    base_filters=128,
    connectivity=0.1,
    leaky=0.8,
    spectral_radius=0.9,
    is_concat=True,
):
    H, W, _ = input_shape
    H_, W_, D_ = output_shape
    D__ = D_ // len(sample_sizes)

    assert H == W, "Only square images are supported."
    assert H_ == W_, "Only square images are supported."

    inputs = keras.layers.Input(shape=input_shape)

    x = inputs
    features = []
    for i in range(N_block):
        for j in range(N_subblock):

            # bottleneck
            if i != 0 and j == 0:
                x = keras.layers.MaxPooling2D((2, 2))(x)

            x = modules.BiESN2D(
                base_filters * 2**i,
                connectivity=connectivity,
                leaky=leaky,
                spectral_radius=spectral_radius,
            )(x)

        if H / 2**i in sample_sizes:
            if is_concat:
                y = keras.layers.Dense(
                    D__,
                    use_bias=False,
                    kernel_initializer=keras.initializers.GlorotUniform(seed=0),
                )(x)
                y = keras.layers.UpSampling2D((int(H_ / H * 2**i), int(W_ / W * 2**i)), interpolation="bilinear")(y)
            else:
                y = x
            features.append(y)

    if is_concat:
        x = keras.layers.Concatenate()(features)
    else:
        x = features

    outputs = x

    model = keras.Model(inputs, outputs)

    return model


def _get_feature_extractor_cnn(input_shape, sample_sizes=(64, 32, 16), random_dim_size=128):

    LAYER_NAMES = []
    for ss, name in zip([64, 32, 16], ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]):
        if ss in sample_sizes:
            LAYER_NAMES.append(name)

    inputs = keras.layers.Input(shape=input_shape)
    input_tensor = keras.applications.resnet50.preprocess_input(inputs)
    backbone = keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=input_tensor)
    backbone.trainable = False

    feature_maps = [backbone.get_layer(name).output for name in LAYER_NAMES]

    fm_sizes = [fm.shape[1:3] for fm in feature_maps]
    max_fm_size = max(fm_sizes, key=lambda x: x[1])

    multi_scale_feature_map = []
    for feature_map, fm_size in zip(feature_maps, fm_sizes):
        r_H, r_W = max_fm_size[0] // fm_size[0], max_fm_size[1] // fm_size[1]
        if (r_H != 1) and (r_W != 1):
            if random_dim_size is not None:
                feature_map = keras.layers.UpSampling2D(size=(r_H, r_W), interpolation="bilinear")(feature_map)
        multi_scale_feature_map.append(feature_map)

    if random_dim_size is not None:
        multi_scale_feature_map = keras.layers.Concatenate(axis=-1)(multi_scale_feature_map)
        sampled_feature_map = SamplingFeatureMaps(random_dim_size)(multi_scale_feature_map)
    else:
        sampled_feature_map = multi_scale_feature_map

    outputs = sampled_feature_map

    return keras.Model(inputs=inputs, outputs=outputs, name="feature_extractor")


class SamplingFeatureMaps(keras.layers.Layer):
    def __init__(self, random_dim_size, seed=0, name="sampling_feature_maps", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.seed = seed
        self.random_dim_size = random_dim_size

    def build(self, input_shape):
        C = input_shape[-1]
        np.random.seed(self.seed)
        self.random_indices = np.random.permutation(C)[: self.random_dim_size]

    def call(self, inputs):
        sampling_features = tf.gather(inputs, self.random_indices, axis=-1)

        return sampling_features
