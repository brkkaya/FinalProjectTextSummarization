import tensorflow as tf
import tensorflow.keras as keras


class Embedding(keras.layers.Layer):
    def __init__(
        self, vocab_size: int = 128000, model_dim: int = 768, **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=model_dim
        )
        self.model_dim = model_dim

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.embedding(inputs) * tf.sqrt(self.model_dim)
