import tensorflow as tf
import tensorflow.keras as keras
from summary_generator.model.encoder import Encoder
from summary_generator.model.decoder import Decoder


class Transformer(keras.Model):
    def __init__(
        self,
        number_of_decoder=2,
        vocab_size=128000,
        model_dim=768,
        seq_dim=512,
        number_of_heads=12,
        epsilon=1e-6,
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder(
            number_of_decoder=2,
            vocab_size=128000,
            model_dim=768,
            seq_dim=512,
            number_of_heads=12,
            epsilon=1e-6,
            dropout_rate=0.1,
        )
        self.out = keras.layers.Dense(128000, "softmax")

    def call(self, text_token, summary_token):
        (
            enc_padding_mask,
            look_ahead_mask,
            dec_padding_mask,
        ) = self.create_masks(text_token, summary_token)
        x = self.encoder(text_token, tf.squeeze(enc_padding_mask,axis=[1,2]))
        x = self.decoder(summary_token, x, look_ahead_mask, dec_padding_mask)
        x = self.out(x)
        return x

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask
