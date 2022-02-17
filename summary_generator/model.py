from typing import List, Union
from transformers import TFAutoModel
from src.services.base_service import BaseService
from data_retrieve.data_reader import DataReader
import tensorflow as tf
import tensorflow.keras.backend as k
import tensorflow.keras as keras


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


class MultiHeadAttention(keras.layers.Layer):
    def __init__(
        self,
        number_of_heads: int = 8,
        seq_dim: int = 768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert seq_dim % number_of_heads == 0

        self.number_of_heads = number_of_heads
        self.seq_dim = seq_dim
        self.depth = seq_dim // number_of_heads

        self.drop = keras.layers.Dropout(0.5)
        self.wq = keras.layers.Dense(units=seq_dim)
        self.wk = keras.layers.Dense(units=seq_dim)
        self.wv = keras.layers.Dense(units=seq_dim)

        self.output = keras.layers.Dense(units=seq_dim)

    def __scaled_dot_product(
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

    def __split_heads(self, weights: tf.Tensor, batch_size: int) -> tf.Tensor:
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

        q = self.__split_heads(weights=q, batch_size=batch_size)
        k = self.__split_heads(weights=k, batch_size=batch_size)
        v = self.__split_heads(weights=v, batch_size=batch_size)

        scaled_attention = self.__scaled_dot_product(
            query=q, key=k, value=v, mask=mask
        )

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.seq_dim)
        )

        return self.output(concat_attention)


class PointWiseFeedForward(keras.layers.Layer):
    def __init__(
        self, model_dim: int, seq_dim: int, drop_out_rate: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.seq = keras.layers.Dense(seq_dim, activation=tf.nn.gelu)
        self.model = keras.layers.Dense(model_dim)
        self.drop = keras.layers.Dropout(rate=drop_out_rate)

    def call(self, weights: tf.Tensor) -> tf.Tensor:
        x = self.seq(weights)
        x = self.drop(x)
        x = self.model(x)
        return x


class EncoderLayer(keras.layers.Layer):
    def __init__(
        self,
        model_dim: int,
        seq_dim: int,
        number_of_heads: int,
        epsilon: float,
        drop_out_rate: float,
        **kwargs,
    ) -> None:
        self.mha = MultiHeadAttention(
            seq_dim=seq_dim, number_of_heads=number_of_heads
        )
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=epsilon)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=epsilon)
        self.feed_forward = PointWiseFeedForward(
            model_dim=model_dim,
            seq_dim=seq_dim,
            drop_out_rate=drop_out_rate,
        )
        self.dropout1 = keras.layers.Dropout(drop_out_rate)
        self.dropout2 = keras.layers.Dropout(drop_out_rate)

    def call(
        self, input_tensor: tf.Tensor, mask: Union[tf.Tensor, None] = None
    ):
        attention_output = self.mha(
            query=input_tensor, key=input_tensor, value=input_tensor, mask=mask
        )
        attention_output = self.dropout1(attention_output, training=False)
        out1 = self.layer_norm1(input_tensor + attention_output)

        feed_forward_out = self.feed_forward(out1)
        feed_forward_out = self.dropout2(feed_forward_out, training=False)
        out2 = self.layer_norm2(out1 + feed_forward_out)
        return out2


class DecoderLayer(keras.layers.Layer):
    def __init__(
        self,
        model_dim: int,
        seq_dim: int,
        number_of_heads: int,
        epsilon: float,
        drop_out_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.look_ahead_mha = MultiHeadAttention(
            number_of_heads=number_of_heads,
            seq_dim=seq_dim,
        )
        self.padding_mha = MultiHeadAttention(
            number_of_heads=number_of_heads,
            seq_dim=seq_dim,
        )
        self.dropout1 = keras.layers.Dropout(rate=drop_out_rate)
        self.dropout2 = keras.layers.Dropout(rate=drop_out_rate)
        self.dropout3 = keras.layers.Dropout(rate=drop_out_rate)

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=epsilon)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=epsilon)
        self.layer_norm3 = keras.layers.LayerNormalization(epsilon=epsilon)

        self.feed_forward = PointWiseFeedForward(
            model_dim=model_dim,
            seq_dim=seq_dim,
            drop_out_rate=drop_out_rate,
        )

    def call(
        self,
        input_tensor: tf.Tensor,
        encoder_representation: tf.Tensor,
        look_ahead_mask: Union[tf.Tensor, None],
        padding_mask: Union[tf.Tensor, None],
    ):
        look_ahead_attention_output = self.look_ahead_mha(
            query=input_tensor,
            key=input_tensor,
            value=input_tensor,
            mask=look_ahead_mask,
        )
        look_ahead_attention_output = self.dropout1(
            look_ahead_attention_output, training=False
        )
        out1 = self.layer_norm1(input_tensor + look_ahead_attention_output)

        padding_attention_output = self.padding_mha(
            query=encoder_representation,
            key=encoder_representation,
            value=out1,
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


class SummarizationModel(BaseService):
    def __init__(self, data_reader: DataReader) -> None:
        super().__init__()
        self.data_reader = data_reader


class Encoder(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = TFAutoModel.from_pretrained(
            "dbmdz/bert-base-turkish-128k-uncased"
        )

    def call(self, x: tf.Tensor):
        """

        Args:
            x (tf.Tensor): Tokenized raw string inputs [None, seq_dim,model_dim]

        Returns:
            _type_: _description_
        """
        x = self.encoder(x)
        return x


class Decoder(keras.layers.Layer):
    def __init__(
        self,
        number_of_decoder: int,
        vocab_size: int,
        model_dim: int,
        seq_dim: int,
        number_of_heads: int,
        epsilon: float,
        drop_out_rate: float,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.number_of_decoder = number_of_decoder
        self.model_dim = model_dim

        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=seq_dim
        )
        self.pos_encoding = PositionalEncoding(
            model_dimension=model_dim, max_sequence_length=seq_dim
        )

        self.decoder = DecoderLayer(
            model_dim=model_dim,
            seq_dim=seq_dim,
            number_of_heads=number_of_heads,
            epsilon=epsilon,
            drop_out_rate=drop_out_rate,
        )
        self.decoder_layers = [
            DecoderLayer(
                model_dim=model_dim,
                seq_dim=seq_dim,
                number_of_heads=number_of_heads,
                epsilon=epsilon,
                drop_out_rate=drop_out_rate,
            )
            for _ in range(number_of_decoder)
        ]

        self.dropout = keras.layers.Dropout(rate=drop_out_rate)

    def call(
        self,
        input: tf.Tensor,
        encoder_value: tf.Tensor,
        look_ahead_mask: tf.Tensor,
        padding_mask: tf.Tensor,
    ):
        seq_len = tf.shape(input)[-1]
        embedding = self.embedding(input)
        embedding *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        pos_encoded = self.pos_encoding[:, :seq_len, :]

        x = self.dropout(pos_encoded, training=False)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(
                x,
                encoder_value,
                look_ahead_mask,
                padding_mask,
            )

        return x


class BertTransformers(keras.Model):
    def __init__(
        self,
        model_dim: int,
        seq_dim: int,
        number_of_decoder: int,
        epsilon: float,
        drop_out_rate: float,
        vocab_size: int,
        number_of_head: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bert = Encoder()
        self.decoder = Decoder(
            model_dim=model_dim,
            seq_dim=seq_dim,
            number_of_decoder=number_of_decoder,
            number_of_heads=number_of_head,
            epsilon=epsilon,
            drop_out_rate=drop_out_rate,
            vocab_size=vocab_size,
        )
        self.final_layer = keras.layers.Dense(vocab_size, activation="softmax")

    def __create_padding_mask(self, seq: tf.Tensor):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def __look_ahead_mask(self, model_dimension: int):
        mask = 1 - tf.linalg.band_part(
            tf.ones((model_dimension, model_dimension), -1, 0)
        )

        return mask

    def create_masks(self, source: tf.Tensor, target: tf.Tensor):
        """"""
        # Encoder padding
        encoder_padding_mask = self.__create_padding_mask(seq=source)

        # This used in second attention block in the decoder
        decoder_padding_mask = self.__create_padding_mask(seq=source)

        # first attention block in decoder
        look_ahead_mask = self.__look_ahead_mask(
            model_dimension=tf.shape(target)[-1]
        )
        decoder_target_padding = self.__create_padding_mask(seq=target)
        look_ahead_mask = tf.maximum(decoder_target_padding, look_ahead_mask)

        return encoder_padding_mask, decoder_padding_mask, look_ahead_mask

    def call(
        self,
        source: tf.Tensor,
        target: tf.Tensor,
        source_padding_mask: tf.Tensor=None,
        target_padding_mask: tf.Tensor=None,
        look_ahead_target_mask: tf.Tensor=None,
        
    ):
        (
            encoder_padding_mask,
            look_ahead_mask,
            decoder_padding_mask,
        ) = self.create_masks(source=source, target=target)
        """input and target must be tokenized."""
        encoder_output = self.bert(source)
        decoder_output = self.decoder(
            input=source,
            encoder_value=encoder_output,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_padding_mask,
        )
    
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


