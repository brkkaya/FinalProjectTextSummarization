from typing import List
from transformers import AutoModel
from src.services.base_service import BaseService
from DataRetrieve.data_reader import DataReader
import tensorflow as tf
import tensorflow.keras.backend as k
import tensorflow.keras as keras


class SummarizationModel(BaseService):
    def __init__(self, data_reader: DataReader) -> None:
        super().__init__()
        self.data_reader = data_reader


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, model_dimension: int, max_sequence_length: int) -> None:
        super().__init__()
        self.dropout = keras.Dropout(p=0.5)
        position_id = k.arange(start=0, end=max_sequence_length)
        frequencies = k.pow(
            x=10000,
            exponent=-k.arange(0, model_dimension, 2, dtype=tf.float32)
            / model_dimension,
        )
        positional_encodings = k.zeros(
            shape=(max_sequence_length, model_dimension)
        )
        positional_encodings[:, 0::2] = tf.math.sin(position_id * frequencies)
        positional_encodings[:, 1::2] = tf.math.cos(position_id * frequencies)


def look_ahead_mask(model_dimension: int):
    mask = 1 - tf.linalg.band_part(
        tf.ones((model_dimension, model_dimension), -1, 0)
    )
    return mask


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

        self.output = keras.layers.Dense(units=model_dim)

    def scaled_dot_product(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: tf.Tensor,
    ):
        qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.sqrt(tf.shape(key)[-1])
        scaled_attention = tf.nn.softmax(qk / dk, axis=-1)

        if mask is not None:
            scaled_attention += mask * 1e9

        return tf.matmul(scaled_attention, value)

    def split_heads(self, weights: tf.Tensor, batch_size: int) -> tf.Tensor:
        weights = tf.reshape(
            weights, (batch_size, -1, self.num_heads, self.depth)
        )
        return tf.transpose(weights, perm=[0, 2, 1, 3])

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: tf.Tensor,
    ):
        batch_size = tf.shape(query)[0]
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        q = self.split_heads(weights=q, batch_size=batch_size)
        k = self.split_heads(weights=k, batch_size=batch_size)
        v = self.split_heads(weights=v, batch_size=batch_size)

        scaled_attention = self.scaled_dot_product(
            query=q, key=k, value=v, mask=mask
        )

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.model_dim)
        )

        return self.output(concat_attention)


class PointWiseFeedForward(keras.layers.Layer):
    def __init__(self, model_dim: int, seq_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.seq = keras.layers.Dense(seq_dim, activation="relu")
        self.model = keras.layers.Dense(model_dim)
    
    def call(self,weights:tf.Tensor)->tf.Tensor:
        x = self.seq(weights)
        x = self.model(weights)
        return x

