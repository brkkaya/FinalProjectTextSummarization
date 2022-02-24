import tensorflow as tf
import tensorflow.keras as keras


class PointWiseFeedForward(keras.layers.Layer):
    def __init__(self, model_dim: int, dropout_rate: float, **kwargs):
        """Position-wise because this small net will be applied independently to every token

        Args:
            model_dim (int): Respesentation lenght of a token
            drop_out_rate (float): Dropout rate to remove weights
        """
        super().__init__(**kwargs)
        self.seq = keras.layers.Dense(model_dim * 4, activation=tf.nn.relu)
        self.model = keras.layers.Dense(model_dim)
        self.drop = keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.seq(inputs)
        x = self.drop(x)
        x = self.model(x)
        return x
