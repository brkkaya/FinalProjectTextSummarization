import tensorflow.keras as keras
import tensorflow.keras.backend as k
import tensorflow as tf
import numpy as np


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, seq_length: int = 512, model_dim: int = 768, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.seq_length = seq_length
        self.model_dim = model_dim

    def get_angles(self, position_vector, frequency_vector):
        angle_rates = 1 / np.power(
            10000, (2 * (frequency_vector // 2)) / np.float32(self.model_dim)
        )
        return position_vector * angle_rates

    def call(self, inputs):
        angle_rads = self.get_angles(
            np.arange(self.seq_length)[:, np.newaxis],
            np.arange(self.model_dim)[np.newaxis, :],
        )

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        print(angle_rads.shape)
        return tf.cast(pos_encoding, dtype=tf.float32)


# f = PositionalEncoding()
# s = f(2)
# s
