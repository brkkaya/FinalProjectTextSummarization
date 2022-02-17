import tensorflow as tf
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as f


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def point_wise_feed_forward_network(dimension_out: int, dimension_in: int):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    dimension_in, activation="relu"
                ),  # (batch_size, seq_len, dff)
                tf.keras.layers.Dense(
                    dimension_out
                ),  # (batch_size, seq_len, d_model)
            ]
        )


class MultiheadAttention(nn.Module):
    def __init__(self, dimension: int, num_heads: int = 8) -> None:
        super().__init__()
        assert dimension % num_heads == 0

        self.dimension = dimension
        self.num_heads = num_heads

        self.depth = dimension // num_heads

        self.wq = nn.Linear(dimension)
        self.wk = nn.Linear(dimension)
        self.wv = nn.Linear(dimension)

        self.out = nn.Linear(dimension)

    def split_heads(self, weights: Tensor, batch_size: int) -> Tensor:
        weights = tf.reshape(
            weights, (batch_size, -1, self.num_heads, self.depth)
        )
        return tf.transpose(weights, perm=[0, 2, 1, 3])

    def scaled_dot_product(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask=None,
    ) -> Tensor:
        qk = torch.matmul(query, key.transpose(-2, -1))
        dk = key.size()[-1]
        scaled_attention = qk / torch.sqrt(dk)
        if mask is not None:
            scaled_attention += mask * -1e9

        attention_weights = f.softmax(scaled_attention, dim=-1)

        return torch.matmul(attention_weights, value)

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: tf.Tensor,
    ) -> tf.Tensor:
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
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(
            batch_size, -1, self.dimension
        )
        output = self.out(concat_attention)
        return output

        