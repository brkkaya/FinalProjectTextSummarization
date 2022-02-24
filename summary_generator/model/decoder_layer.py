from typing import Union
import tensorflow as tf
import tensorflow.keras as keras
from summary_generator.model.feed_forward import PointWiseFeedForward
from summary_generator.model.multi_head_attention import MultiHeadAttention


class DecoderLayer(keras.layers.Layer):
    def __init__(
        self,
        model_dim: int = 768,
        seq_dim: int = 512,
        number_of_heads: int = 12,
        epsilon: float = 1e-6,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.masked_mha = MultiHeadAttention(
            model_dim=model_dim,
            number_of_heads=number_of_heads,
        )
        self.mha = MultiHeadAttention(
            model_dim=model_dim,
            number_of_heads=number_of_heads,
        )

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=epsilon)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=epsilon)
        self.layer_norm3 = keras.layers.LayerNormalization(epsilon=epsilon)

        self.dropout1 = keras.layers.Dropout(rate=dropout_rate)
        self.dropout2 = keras.layers.Dropout(rate=dropout_rate)
        self.dropout3 = keras.layers.Dropout(rate=dropout_rate)

        self.feed_forward = PointWiseFeedForward(
            model_dim=model_dim, dropout_rate=dropout_rate
        )

    def call(
        self,
        input_tensor: tf.Tensor,
        encoder_representation: tf.Tensor,
        look_ahead_mask: Union[tf.Tensor, None],
        padding_mask: Union[tf.Tensor, None],
    ) -> tf.Tensor:
        look_ahead_attention_output = self.masked_mha(
            query=input_tensor,
            key=input_tensor,
            value=input_tensor,
            mask=look_ahead_mask,
        )
        look_ahead_attention_output = self.dropout1(
            look_ahead_attention_output, training=False
        )
        out1 = self.layer_norm1(look_ahead_attention_output + input_tensor)
        padding_attention_output = self.mha(
            query=out1,
            key=encoder_representation,
            value=encoder_representation,
            mask=padding_mask,
        )
        padding_attention_output = self.dropout2(
            padding_attention_output, training=False
        )

        out2 = self.layer_norm2(out1 + padding_attention_output)

        feed_forward = self.feed_forward(out2)
        feed_forward = self.dropout3(feed_forward, training=False)
        out3 = self.layer_norm3(out2 + feed_forward)
        return out3
