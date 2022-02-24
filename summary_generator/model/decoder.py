import tensorflow as tf
import tensorflow.keras as keras
from summary_generator.model.decoder_layer import DecoderLayer
from summary_generator.model.positional_encoding import PositionalEncoding


class Decoder(keras.layers.Layer):
    def __init__(
        self,
        number_of_decoder: int,
        vocab_size: int,
        model_dim: int,
        seq_dim: int,
        number_of_heads: int,
        epsilon: float,
        dropout_rate: float,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.number_of_decoder = number_of_decoder
        self.model_dim = model_dim
        self.seq_dim = seq_dim
        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=seq_dim
        )
        self.pos_encoding = PositionalEncoding(
            model_dim=model_dim, seq_length=seq_dim
        )

        self.decoder_layers = [
            DecoderLayer(
                model_dim=model_dim,
                seq_dim=seq_dim,
                number_of_heads=number_of_heads,
                epsilon=epsilon,
                dropout_rate=dropout_rate,
            )
            for _ in range(number_of_decoder)
        ]

        self.dropout = keras.layers.Dropout(rate=dropout_rate)

    def call(
        self,
        input: tf.Tensor,
        encoder_value: tf.Tensor,
        padding_mask: tf.Tensor,
        look_ahead_mask: tf.Tensor,
    ):
        embedding = self.embedding(input)
        embedding *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        pos_encoded = self.pos_encoding(None)

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
