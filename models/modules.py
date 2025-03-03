# TF/Keras
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras


# B x H x W x C -> B x N_h x N_w x D
class Patches2Vectors(keras.layers.Layer):
    def __init__(self, patch_sizes):
        super().__init__()
        self.Wp, self.Hp = patch_sizes[0], patch_sizes[1]

    def call(self, images):
        B, H, W, C = tf.shape(images)[0], tf.shape(images)[1], tf.shape(images)[2], tf.shape(images)[3]

        N_h, N_w = H // self.Hp, W // self.Wp

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.Hp, self.Wp, 1],
            strides=[1, self.Hp, self.Wp, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        patches = tf.reshape(patches, [B, N_h, N_w, self.Hp * self.Wp * C])
        return patches


# B x T x input_dim -> B x T x output_dim
class BiESN(keras.layers.Layer):
    def __init__(self, output_dim, connectivity=0.1, leaky=0.9, spectral_radius=0.95, activation="tanh", seed=0):
        super().__init__()
        initializer = keras.initializers.GlorotUniform(seed=seed)

        forward_esn_layer = tfa.layers.ESN(
            units=output_dim // 2,
            connectivity=connectivity,
            leaky=leaky,
            spectral_radius=spectral_radius,
            activation=activation,
            return_sequences=True,
            kernel_initializer=initializer,
            recurrent_initializer=initializer,
        )

        backward_esn_layer = tfa.layers.ESN(
            units=output_dim // 2,
            connectivity=connectivity,
            leaky=leaky,
            spectral_radius=spectral_radius,
            activation=activation,
            return_sequences=True,
            kernel_initializer=initializer,
            recurrent_initializer=initializer,
            go_backwards=True,
        )

        self.bi_esn_layer = keras.layers.Bidirectional(
            layer=forward_esn_layer,
            merge_mode="concat",
            backward_layer=backward_esn_layer,
        )

    def call(self, inputs):
        x = self.bi_esn_layer(inputs)
        return x


# B x N_h x N_w x input_dim -> B x N_h x N_w x output_dim
class BiESN2D(keras.layers.Layer):
    def __init__(self, output_dim, connectivity=0.1, leaky=0.9, spectral_radius=0.95, activation="tanh"):
        super().__init__()
        self.D = output_dim
        self.vertical_bi_esn_layer = BiESN(output_dim // 2, connectivity, leaky, spectral_radius, activation)
        self.horizontal_bi_esn_layer = BiESN(output_dim // 2, connectivity, leaky, spectral_radius, activation)

    def call(self, inputs):
        B, N_h, N_w, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]

        x = tf.reshape(inputs, (B, N_h, N_w, C))

        horizontal_x = tf.reshape(x, (B * N_h, N_w, C))
        horizontal_x = self.horizontal_bi_esn_layer(horizontal_x)
        horizontal_x = tf.reshape(horizontal_x, (B, N_h, N_w, self.D // 2))

        vertical_x = tf.reshape(tf.einsum("bhwc->bwhc", x), (B * N_w, N_h, C))
        vertical_x = self.vertical_bi_esn_layer(vertical_x)
        vertical_x = tf.einsum("bwhc->bhwc", tf.reshape(vertical_x, (B, N_w, N_h, self.D // 2)))

        x = keras.layers.Concatenate()([horizontal_x, vertical_x])

        return x
