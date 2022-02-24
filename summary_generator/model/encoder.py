import tensorflow as tf
import tensorflow.keras as keras
from transformers import TFAutoModel


class Encoder(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = TFAutoModel.from_pretrained(
            "dbmdz/bert-base-turkish-128k-uncased"
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
