from typing import List, Tuple, Union
from transformers import TFAutoModel
from src.services.base_service import BaseService
from data_retrieve.data_reader import DataReader
import tensorflow as tf
import tensorflow.keras.backend as k
import tensorflow.keras as keras


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, model_dimension: int, max_sequence_length: int) -> None:
        super().__init__()
        self.dropout = keras.layers.Dropout(0.5)
        self.model_dimension = tf.cast(model_dimension, dtype=tf.float32)
        self.max_sequence_length = max_sequence_length
        position_id = tf.expand_dims(
            k.arange(0, self.max_sequence_length, dtype=tf.float32), axis=-1
        )
        frequencies = k.pow(
            10000,
            -tf.expand_dims(
                k.arange(0, self.model_dimension, 2, dtype=tf.float32), axis=0
            )
            / self.model_dimension,
        )
        positional_encodings = k.zeros(
            shape=(self.max_sequence_length, self.model_dimension)
        )
        import numpy as np

        positional_encodings[:, 0::2] = tf.math.sin(position_id * frequencies)
        positional_encodings[:, 1::2] = tf.math.cos(position_id * frequencies)

    def call(self):
        position_id = tf.expand_dims(
            k.arange(0, self.max_sequence_length), axis=-1
        )
        frequencies = k.pow(
            10000,
            -tf.expand_dims(
                k.arange(0, self.model_dimension, 2, dtype=tf.float32), axis=0
            )
            / self.model_dimension,
        )
        positional_encodings = k.zeros(
            shape=(self.max_sequence_length, self.model_dimension)
        )
        positional_encodings[:, 0::2] = tf.math.sin(position_id * frequencies)
        positional_encodings[:, 1::2] = tf.math.cos(position_id * frequencies)
        return positional_encodings


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

        self.out = keras.layers.Dense(units=seq_dim)

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
            scaled_attention += mask * 1e9

        return tf.matmul(scaled_attention, value)

    def __split_heads(self, weights: tf.Tensor, batch_size: int) -> tf.Tensor:
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

        return self.out(concat_attention)


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
        print(tf.shape(input_tensor))
        print(tf.shape(look_ahead_attention_output))
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
            "dbmdz/bert-base-turkish-128k-cased"
        )
        for layer in self.encoder.layers:
            layer.trainable = False
            
    @tf.function
    def call(self, tokens: tf.Tensor, attention: tf.Tensor):
        
        """
        Args:
            x (tf.Tensor): Tokenized raw string inputs [None, seq_dim,model_dim]

        Returns:
            _type_: _description_
        """
        # tokens, attention = inputs
        x = self.encoder(tokens, attention)[0]
        # x = self.encoder([tokens, attention])[2][-1]
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
        self.seq_dim = seq_dim
        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=seq_dim
        )
        # self.pos_encoding = PositionalEncoding(
        #     model_dimension=model_dim, max_sequence_length=seq_dim
        # )

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

    def positional_encoding(
        self,
    ):

        import numpy as np

        def get_angles(position_vector, frequency_vector, model_dim):
            angle_rates = 1 / np.power(
                10000, (2 * (frequency_vector // 2)) / np.float32(model_dim)
            )
            return position_vector * angle_rates

        angle_rads = get_angles(
            np.arange(self.seq_dim)[:, np.newaxis],
            np.arange(self.model_dim)[np.newaxis, :],
            self.model_dim,
        )

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

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
        pos_encoded = self.positional_encoding()[:, :seq_len, :]

        x = self.dropout(pos_encoded, training=False)
        print(tf.shape(x))
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
        self.input_layer = keras.layers.Input(shape=(model_dim, seq_dim))
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
        print(f"++++++++{seq}")

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def __look_ahead_mask(self, model_dimension: int):
        mask = 1 - tf.linalg.band_part(
            tf.ones((model_dimension, model_dimension)), -1, 0
        )

        return mask

    def create_masks(self, source: tf.Tensor, target: tf.Tensor):
        """"""
        # Encoder padding
        print(f"----------{tf.shape(source)}")
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

    def create_mk(self, inp, tar):
        print(tf.shape(inp))
        print(tf.shape(tar))
        enc_padding_mask = self.__create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.__create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self.__look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.__create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask

    def call(
        self,
        inputs,
        training,
    ):
        source, target = inputs[0], inputs[1]
        (
            encoder_padding_mask,
            look_ahead_mask,
            decoder_padding_mask,
        ) = self.create_mk(inp=source, tar=target)
        """input and target must be tokenized."""
        encoder_output = self.bert(source, encoder_padding_mask)
        decoder_output = self.decoder(
            input=source,
            encoder_value=encoder_output,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_padding_mask,
        )
        return self.final_layer(decoder_output)



#%%

# import tensorflow as tf
# def create_padding_mask(seq):
#   seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

#   # add extra dimensions to add the padding
#   # to the attention logits.
#   return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
# def create_look_ahead_mask(size):
#   mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#   return mask  # (seq_len, seq_len)
# def create_masks( inp, tar):
#     # Encoder padding mask
#     enc_padding_mask = create_padding_mask(inp)

#     # Used in the 2nd attention block in the decoder.
#     # This padding mask is used to mask the encoder outputs.
#     dec_padding_mask = create_padding_mask(inp)

#     # Used in the 1st attention block in the decoder.
#     # It is used to pad and mask future tokens in the input received by
#     # the decoder.
#     look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#     dec_target_padding_mask = create_padding_mask(tar)
#     look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

#     return enc_padding_mask, look_ahead_mask, dec_padding_mask

# i = tf.random.uniform((50,512))
# s = tf.random.uniform((50,512))
# create_masks(i,s)
# # %%


#%%

# %%
