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

    def call(self, text_token, text_mask, summary_token, summary_mask):
        x = self.encoder(text_token, text_mask)
        x = self.decoder(summary_token, x, summary_mask)
        x = self.out(x)
        return x
