from statistics import mode
from time import time
from src.services.base_service import BaseService
from transformers.pre_process import PreProcess
from transformers.model import BertTransformers, CustomSchedule
from data_retrieve.data_reader import DataReader
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf


class ModelTraining(BaseService):
    train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        ]
    def __init__(
        self,
        pre_process: PreProcess,
        data_reader: DataReader,
    ) -> None:
        super().__init__()
        self.pre_process = pre_process
        self.data_reader = data_reader
        self.loss_object = SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

    def train(self):
        d_model = 512
        learning_rate = CustomSchedule(d_model)
        self.bert_transformers: BertTransformers = BertTransformers(
            number_of_decoder=2,
            model_dim=512,
            seq_dim=768,
            epsilon=1e-6,
            drop_out_rate=0.1,
            vocab_size=128000,
            number_of_head=12,
        )
        self.optimizer = Adam(
            learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9,
        )
        checkpoint_path = f"{self.global_path_provider}/checkpoints/train"

        ckpt = tf.train.Checkpoint(
            transformer=bert_transformers, optimizer=optimizer
        )

        ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_path, max_to_keep=5
        )

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!!")

        

        for epoch in range(20):  # EPOCH 20
            start = time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            # inp -> portuguese, tar -> english
            for (batch, (inp, tar)) in enumerate(train_batches):
                self.train_step(inp, tar)

                if batch % 50 == 0:
                    print(
                        f"Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}"
                    )

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(
                    f"Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}"
                )

            print(
                f"Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}"
            )

            print(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = self.bert_transformers(
                [inp, tar_inp], training=True
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
