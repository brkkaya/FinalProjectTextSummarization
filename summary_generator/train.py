import tensorflow as tf
import tensorflow.keras as keras
from time import time
from data_retrieve.data_reader import DataReader
from src.services.base_service import BaseService
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from transformers import TFAutoModel
from summary_generator.model.decoder import Decoder
from summary_generator.model.encoder import Encoder
from summary_generator.pre_process import PreProcess
from summary_generator.model.transformer import Transformer
from transformers import AutoTokenizer


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


class ModelTraining(BaseService):
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]
    val_step_signature = [
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
        tf.get_logger().setLevel("ERROR")
        self.pre_process = pre_process
        self.model = model

        self.data_reader = data_reader
        self.loss_object = SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        self.train_loss = keras.metrics.Mean(name="train_loss")
        self.train_accuracy = keras.metrics.Mean(name="train_accuracy")
        self.val_loss = keras.metrics.Mean(name="val_loss")
        self.val_accuracy = keras.metrics.Mean(name="val_accuracy")
        learning_rate = CustomSchedule(512)

        self.optimizer = Adam(
            learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9,
        )
        self.results1 = {"precision": [], "recall": [], "fmeasure": []}
        self.resultsL = {"precision": [], "recall": [], "fmeasure": []}

    def train(self):
        # tf.config.run_functions_eagerly(True)
        d_model = 512
        # seq_dim = 768
        (
            text_token,
            summary_token,
            # val_text_token,
            # val_summary_token,
        ) = self.pre_process.pipeline(is_eval=False)
        train_batches = (
            tf.data.Dataset.from_tensor_slices(
                (
                    text_token,
                    summary_token,
                )
            )
            .shuffle(buffer_size=1024, seed=42353)
            .batch(batch_size=2)
        )
        self.bert_transformers: Transformer = Transformer(
            number_of_decoder=self.decoder_number,
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
            seq_dim=self.seq_dim,
            number_of_heads=self.number_of_heads,
            epsilon=self.epsilon,
            dropout_rate=self.dropout_rate,
        )

        # train_batches = self.pre_process.pipeline()

        # )  # (batch_size, input_seq_len, d_model)
        # tf.keras.utils.plot_model(
        #     self.bert_transformers,
        #     to_file=f"{self.global_path_provider.logs_path}/model.png",
        # )
        # print(self.bert_transformers.summary())

        checkpoint_path = f"/content/drive/MyDrive/transformers_model/FinalProjectTextSummarization/logs/checkpoints/train_vanilla"

        ckpt = tf.train.Checkpoint(
            transformer=self.bert_transformers, optimizer=self.optimizer
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
                    self.log.info(
                        f"Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}"
                    )

            ckpt_save_path = ckpt_manager.save()
            self.log.info(
                f"Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}"
            )
            # self.validate(val_batches=val_batches)

            self.log.info(
                f"Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f} \t Val Loss {self.val_loss.result():.4f} Val Accuracy {self.val_accuracy.result():.4f}"
            )

            self.log.info(
                f"Time taken for 1 epoch: {time() - start:.2f} secs\n"
            )

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):

        tar_inp = tar[:, :-1]

        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions = self.bert_transformers(
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

    @tf.function(input_signature=val_step_signature)
    def validate(self, val_batches, model: Transformer):
        preds_val = []

        for (batch, (val_inp, val_tar)) in enumerate(val_batches):
            val_tar_inp = val_tar[:, :-1]
            val_tar_real = val_tar[:, 1:]
            predictions_val = model(val_inp, val_tar_inp, training=False)
            val_loss = self.loss_function(val_tar_real, predictions_val)
            self.val_loss(val_loss)
            self.val_accuracy(
                self.accuracy_function(val_tar_real, predictions_val)
            )
            preds_val.append(self.pre_process.decode_tokens(predictions_val))
            if batch % 50 == 0:
                self.log.info(
                    f"Val Loss Batch {batch} : {self.val_loss.result():.4f}"
                )
                self.log.info(
                    f"Val Accuracy Batch {batch} : {self.val_accuracy.result():.4f}"
                )
        return preds_val

    def loss_function(self, real: tf.Tensor, pred: tf.Tensor):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(x=mask, dtype=loss_.dtype)
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def accuracy_function(self, real: tf.Tensor, pred: tf.Tensor):

        accuracies = tf.equal(
            real, tf.cast(tf.argmax(pred, axis=2), dtype=tf.int32)
        )

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    def test_classify(self):
        tf.config.run_functions_eagerly(True)
        (
            text_token,
            text_attention,
            summary_token,
            summary_attention,
        ) = self.pre_process.pipeline()
        # tf.config.run_functions_eagerly(True)
        input_ids = keras.Input(shape=(512,), dtype="int32")
        attention_mask = keras.Input(shape=(512,), dtype="int32")
        summary_ids = keras.Input(shape=(512,), dtype="int32")
        summary_mask = keras.Input(shape=(512,), dtype="int32")
        # self.model(input_ids,attention_mask)
        enc = Encoder()
        encoder = enc(input_ids, attention_mask)
        # hidden_states = encoder[2][-1]  # get output_hidden_states
        # for layer in self.model.layers:
        #     layer.trainable = False

        x = Decoder(
            number_of_decoder=2,
            vocab_size=128000,
            model_dim=768,
            seq_dim=512,
            number_of_heads=12,
            epsilon=1e-6,
            dropout_rate=0.1,
        )(summary_ids, encoder, summary_mask)
        x = keras.layers.Dense(128000, activation="softmax")(x)

        model = keras.models.Model(
            inputs=[input_ids, attention_mask, summary_ids, summary_mask],
            outputs=x,
        )
        model.compile(
            Adam(lr=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # tf.keras.utils.plot_model(
        #     model,
        #     to_file=f"{self.global_path_provider.logs_path}/model.png",
        # )
        import numpy as np

        history = model.fit(
            [text_token, text_attention],
            [summary_token, summary_attention],
            batch_size=2,
            epochs=2,
            # validation_split=0.2,
        )

    def calculate_rouge_score(self, preds: tf.Tensor, reals: tf.Tensor):
        """preds: are the token output of model
        reals: real validation summary data."""
        from rouge_score import rouge_scorer
        import numpy as np

        scorer = rouge_scorer.RougeScorer(["rouge1", "rougleL"])
        results1 = {"precision": [], "recall": [], "fmeasure": []}
        resultsL = {"precision": [], "recall": [], "fmeasure": []}
        preds = self.pre_process.decode_tokens(preds)
        for pred, real in zip(preds, reals):
            score = scorer.score(pred, real)
            precision, recall, fmeasure = score["rouge1"]
            results1["precision"].append(precision)
            results1["recall"].append(recall)
            results1["fmeasure"].append(fmeasure)
            precision, recall, fmeasure = score["rougeL"]

            resultsL["precision"].append(precision)
            resultsL["recall"].append(recall)
            resultsL["fmeasure"].append(fmeasure)

        self.log.info(
            f"Rouge-1 Score Precision:{np.mean(np.asarray(results1['precision']))}"
        )
        self.log.info(
            f"Rouge-1 Score Recall:{np.mean(np.asarray(results1['recall']))}"
        )
        self.log.info(
            f"Rouge-1 Score fmeasure:{np.mean(np.asarray(results1['fmeasure']))}"
        )
        self.log.info(
            f"Rouge-L Score Precision:{np.mean(np.asarray(resultsL['precision']))}"
        )
        self.log.info(
            f"Rouge-L Score Recall:{np.mean(np.asarray(resultsL['recall']))}"
        )
        self.log.info(
            f"Rouge-L Score fmeasure:{np.mean(np.asarray(resultsL['fmeasure']))}"
        )

    def evaluate_model(self, model_path: str):
        import numpy as np

        tf.config.run_functions_eagerly(True)
        model = Transformer(
            number_of_decoder=self.decoder_number,
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
            seq_dim=self.seq_dim,
            number_of_heads=self.number_of_heads,
            epsilon=self.epsilon,
            dropout_rate=self.dropout_rate,
        )
        checkpoint_path = "/content/drive/MyDrive/transformers_model/FinalProjectTextSummarization/logs/checkpoints/train_vanilla"

        ckpt = tf.train.Checkpoint(transformer=model, optimizer=self.optimizer)

        ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_path, max_to_keep=5
        )

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!!")

        val_text, val_summary, val_raw_summary = self.pre_process.pipeline(
            is_eval=True
        )
        val_batches = (
            tf.data.Dataset.from_tensor_slices(
                (val_text, val_summary, val_raw_summary)
            )
            .shuffle(buffer_size=1024, seed=42353)
            .batch(batch_size=1)
        )
        # preds = self.validate(val_batches, model=model)
        for (batch, (val_inp, _, raw_summary)) in enumerate(val_batches):

            self.generate_text(
                val_inp, raw_summary, model, self.pre_process.tokenizer
            )
            if batch % 50 == 0:
                self.log.info(f"Batch Size == {batch}")

        self.log.info(
            f"Rouge-1 Score Precision:{np.mean(np.asarray(self.results1['precision']))}"
        )
        self.log.info(
            f"Rouge-1 Score Recall:{np.mean(np.asarray(self.results1['recall']))}"
        )
        self.log.info(
            f"Rouge-1 Score fmeasure:{np.mean(np.asarray(self.results1['fmeasure']))}"
        )
        self.log.info(
            f"Rouge-L Score Precision:{np.mean(np.asarray(self.resultsL['precision']))}"
        )
        self.log.info(
            f"Rouge-L Score Recall:{np.mean(np.asarray(self.resultsL['recall']))}"
        )
        self.log.info(
            f"Rouge-L Score fmeasure:{np.mean(np.asarray(self.resultsL['fmeasure']))}"
        )

    @tf.function(input_signature=val_step_signature)
    def generate_text(
        self,
        text_token: tf.Tensor,
        val_summary: str,
        model: Transformer,
        tokenizer: AutoTokenizer,
    ):
        from rouge_score import rouge_scorer

        start_token = tf.constant(2, dtype=tf.int64)[tf.newaxis]
        end_token = tf.constant(3, dtype=tf.int64)[tf.newaxis]
        output_array = tf.TensorArray(
            dtype=tf.int64, size=0, dynamic_size=True
        )
        output_array = output_array.write(0, start_token)
        flag_break = False
        for i in tf.range(512):
            output = tf.transpose(output_array.stack())
            # s = tf.random.uniform((1, 56, 100))
            preds = model(text_token, output, training=False)
            preds = preds[:, -1:, :]
            predicted_id = tf.argmax(preds, axis=-1)
            output_array = output_array.write(i + 1, predicted_id[0])
            if predicted_id == end_token:
                flag_break = True
                break
        output = tf.transpose(output_array.stack())
        if flag_break:
            gen_token_list = output[0, 1:-1].numpy().tolist()
        else:
            gen_token_list = output[0, 1:].numpy().tolist()
        pred_text = tokenizer.decode(gen_token_list)
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougleL"])
        score = scorer.score(pred_text, val_summary)
        precision, recall, fmeasure = score["rouge1"]
        self.results1["precision"].append(precision)
        self.results1["recall"].append(recall)
        self.results1["fmeasure"].append(fmeasure)
        precision, recall, fmeasure = score["rougeL"]

        self.resultsL["precision"].append(precision)
        self.resultsL["recall"].append(recall)
        self.resultsL["fmeasure"].append(fmeasure)
