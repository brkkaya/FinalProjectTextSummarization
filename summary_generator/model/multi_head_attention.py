import tensorflow.keras as keras
import tensorflow.keras.backend as k
import tensorflow as tf


class MultiHeadAttention(keras.layers.Layer):
    def __init__(
        self,
        number_of_heads: int = 8,
        model_dim: int = 768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert model_dim % number_of_heads == 0

        self.number_of_heads = number_of_heads
        self.model_dim = model_dim
        self.depth = model_dim // number_of_heads

        self.drop = keras.layers.Dropout(0.5)
        self.wq = keras.layers.Dense(units=model_dim)
        self.wk = keras.layers.Dense(units=model_dim)
        self.wv = keras.layers.Dense(units=model_dim)

        self.out = keras.layers.Dense(units=model_dim)

    def __scaled_dot_product(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: tf.Tensor,
    ):

        qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.sqrt(tf.cast(tf.shape(key)[-1], dtype=tf.float32))
        scaled_attention = tf.nn.softmax(qk / dk, axis=-1)

        if mask is not None:
            mask = mask * -1e9
            scaled_attention += mask

        return tf.matmul(scaled_attention, value)

    def __split_heads(self, weights: tf.Tensor, batch_size: int) -> tf.Tensor:
        print(f"weights {weights.shape}")
        weights = tf.reshape(
            weights, (batch_size, -1, self.number_of_heads, self.depth)
        )
        return tf.transpose(weights, perm=[0, 2, 1, 3])

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: tf.Tensor,
    ):
        """input of qkv must be (BatchSize, SequenceSize(512), ModelSize(768)"""
        batch_size = tf.shape(query)[0]
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        q = self.__split_heads(weights=q, batch_size=batch_size)
        k = self.__split_heads(weights=k, batch_size=batch_size)
        v = self.__split_heads(weights=v, batch_size=batch_size)

        scaled_attention = self.__scaled_dot_product(
            query=q, key=k, value=v, mask=mask
        )

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.model_dim)
        )

        return self.out(concat_attention)
