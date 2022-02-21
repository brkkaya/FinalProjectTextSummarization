from base64 import encode
from statistics import mode
from time import time
from src.services.base_service import BaseService
from summary_generator.pre_process import PreProcess
from summary_generator.model import (
    BertTransformers,
    CustomSchedule,
    Decoder,
    Encoder,
    EncoderLayer,
)
from data_retrieve.data_reader import DataReader
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
import tensorflow.keras as keras
from transformers import TFAutoModel
import torch


class ModelTraining(BaseService):
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    def __init__(
        self,
        pre_process: PreProcess,
        data_reader: DataReader,
        model: TFAutoModel,
    ) -> None:
        super().__init__()
        self.pre_process = pre_process
        self.model = model

        self.data_reader = data_reader
        self.loss_object = SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

    def train(self):
        # tf.config.run_functions_eagerly(True)
        d_model = 512
        # seq_dim = 768
        text_token, text_attention, summary_token = self.pre_process.pipeline()
        # train_batches = self.pre_process.pipeline()
        # learning_rate = CustomSchedule(d_model)

        # transformer = self.model()
        # hidden_states = transformer[1]  # get output_hidden_states

        # hidden_states_size = 4  # count of the last states
        # hiddes_states_ind = list(range(-hidden_states_size, 0, 1))

        # selected_hidden_states = tf.keras.layers.concatenate(
        #     tuple([hidden_states[i] for i in hiddes_states_ind])
        # )

        # decoder = Decoder(
        #     model_dim=d_model,
        #     seq_dim=seq_dim,
        #     number_of_decoder=2,
        #     number_of_heads=12,
        #     epsilon=1e-6,
        #     drop_out_rate=0.1,
        #     vocab_size=128000,
        # )(selected_hidden_states)
        # out = keras.layers.Dense(128000,activation='softmax')(decoder)
        # model = keras.models.Model()

        # self.optimizer = Adam(
        #     learning_rate,
        #     beta_1=0.9,
        #     beta_2=0.98,
        #     epsilon=1e-9,
        # )
        # self.bert_transformers: BertTransformers = BertTransformers(
        #     number_of_decoder=2,
        #     model_dim=512,
        #     seq_dim=768,
        #     epsilon=1e-6,
        #     drop_out_rate=0.1,
        #     vocab_size=128000,
        #     number_of_head=12,
        # )

        # )  # (batch_size, input_seq_len, d_model)
        # tf.keras.utils.plot_model(
        #     self.bert_transformers,
        #     to_file=f"{self.global_path_provider.logs_path}/model.png",
        # )
        # print(self.bert_transformers.summary())

        # checkpoint_path = (
        #     f"{self.global_path_provider.logs_path}/checkpoints/train"
        # )

        # ckpt = tf.train.Checkpoint(
        #     transformer=self.bert_transformers, optimizer=self.optimizer
        # )

        # ckpt_manager = tf.train.CheckpointManager(
        #     ckpt, checkpoint_path, max_to_keep=5
        # )

        # # if a checkpoint exists, restore the latest checkpoint.
        # if ckpt_manager.latest_checkpoint:
        #     ckpt.restore(ckpt_manager.latest_checkpoint)
        #     print("Latest checkpoint restored!!")

        # for epoch in range(20):  # EPOCH 20
        #     start = time()

        #     self.train_loss.reset_states()
        #     self.train_accuracy.reset_states()

        #     # inp -> portuguese, tar -> english
        #     for (batch, (inp, tar)) in enumerate(train_batches):

        #         self.train_step(inp, tar)

        #         if batch % 50 == 0:
        #             print(
        #                 f"Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}"
        #             )

        #     if (epoch + 1) % 5 == 0:
        #         ckpt_save_path = ckpt_manager.save()
        #         print(
        #             f"Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}"
        #         )

        #     print(
        #         f"Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}"
        #     )

        #     print(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = self.bert_transformers(
                inp,
                tar_inp,
            )
            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(
            loss, self.bert_transformers.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(gradients, self.bert_transformers.trainable_variables)
        )

        self.train_loss(loss)
        self.train_accuracy(self.accuracy_function(tar_real, predictions))

    def loss_function(self, real: tf.Tensor, pred: tf.Tensor):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(x=mask, dtype=loss_.dtype)
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def accuracy_function(real: tf.Tensor, pred: tf.Tensor):

        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    def test_classify(self):
        (
            text_token,
            text_attention,
            summary_token,
            summary_attention,
        ) = self.pre_process.pipeline()

        input_ids = tf.keras.Input(shape=(512,), dtype="int32")
        attention_mask = tf.keras.Input(shape=(512), dtype="int32")
        encoder = self.model([input_ids, attention_mask])
        hidden_states = encoder[2][-1]  # get output_hidden_states
        for layer in self.model.layers:
            layer.trainable = False
        hidden_states_size = 4  # count of the last states
        hiddes_states_ind = list(range(-hidden_states_size, 0, 1))
        selected_hiddes_states = tf.keras.layers.concatenate(
            tuple([hidden_states[i] for i in hiddes_states_ind])
        )
        x = keras.layers.LSTM(128, return_sequences=False)(hidden_states)
        x = keras.layers.Dense(1, activation="relu")(x)

        model = tf.keras.models.Model(
            inputs=[input_ids, attention_mask], outputs=x
        )
        model.compile(
            tf.keras.optimizers.Adam(lr=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        import numpy as np

        history = model.fit(
            [text_token, text_attention],
            np.ones((50, 1)),
            batch_size=2,
            epochs=2,
            # validation_split=0.2,
        )
