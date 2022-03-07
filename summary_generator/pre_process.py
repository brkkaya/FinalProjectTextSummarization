from src.services.base_service import BaseService
from transformers import AutoTokenizer, TFAutoModel
from data_retrieve.data_reader import DataReader
from typing import Dict, List
from torch.nn.functional import cosine_similarity
from nltk.tokenize import sent_tokenize
from torch import nn, Tensor


class PreProcess(BaseService):
    def __init__(self, data_reader: DataReader, model: TFAutoModel) -> None:
        super().__init__()
        self.data_reader = data_reader
        self.tokenizer = AutoTokenizer.from_pretrained(
            "dbmdz/bert-base-turkish-128k-cased"
        )
        self.model = model

    def tokenizer_text(
        self,
        texts: List[str],
    ):
        return self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=texts,
            return_tensors="tf",
            padding="max_length",
            max_length=self.seq_dim,
            # return_attention_mask=False,
            truncation=True,
        )["input_ids"]

    def batchify(data: Tensor, mask: Tensor, bsz: int) -> Tensor:
        """Divides the data into bsz separate sequences, removing extra elements
        that wouldn't cleanly fit.

        Args:
            data: Tensor, shape [N]
            bsz: int, batch size

        Returns:
            Tensor of shape [N // bsz, bsz]
        """
        seq_len = data.size(0) // bsz
        data = data[: seq_len * bsz]
        mask = mask[: seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        mask = mask.view(bsz, seq_len).t().contiguous()
        return data, mask
        # return data.to(device)

    def split_tokenizer_out(self, tokens: List[Dict]):
        import tensorflow as tf

        input_ids = []
        attention_mask = []
        token_type_ids = []

        for token in tokens:
            input_ids.append(token["input_ids"])
            attention_mask.append(token["attention_mask"])
            token_type_ids.append(token["token_type_ids"])
        return (
            (tf.squeeze(tf.stack(input_ids), axis=1)),
            (tf.squeeze(tf.stack(attention_mask), axis=1)),
            (tf.squeeze(tf.stack(token_type_ids), axis=1)),
        )

    def split_texts(self, texts: List[str], summaries: List[str]):
        texts = self.split_text_by_sentences(texts)  # list of list of str
        summaries = self.split_text_by_sentences(summaries)
        # list of list of str
        texts_chunks_list: List = []
        texts_chunks_sentences: List = []
        summaries_chunks_list: List = []
        summaries_chunks_sentences: List = []
        for text, summary in zip(texts, summaries):
            text_chunks_list: List = []
            text_chunks_sentences: List = []
            summary_chunks_list: List = []
            summary_chunks_sentences: List = []
            text_length = 0
            for sentence in text:
                sentence_length = len(sentence.split(" "))
                if text_length + sentence_length < 512:
                    text_length += sentence_length
                    text_chunks_sentences.append(sentence)
                else:
                    text_chunks_list.append(" ".join(text_chunks_sentences))
                    text_length = 0
                    text_chunks_sentences.clear()
            if text_length != 0:
                text_chunks_list.append(" ".join(text_chunks_sentences))

            summary_chunk_length = len(" ".join(summary).split(" ")) // len(
                text_chunks_list
            )
            if summary_chunk_length > 1:
                summary_length = 0
                for sentence in summary:
                    sentence_length = len(sentence.split(" "))
                    if (
                        summary_length + sentence_length
                        <= summary_chunk_length
                    ):
                        summary_length += sentence_length
                        summary_chunks_sentences.append(sentence)
                    else:
                        summary_chunks_list.append(
                            " ".join(summary_chunks_sentences)
                        )
                        summary_length = 0
                        summary_chunks_sentences.clear()
                if summary_length != 0:
                    summary_chunks_list.append(
                        " ".join(summary_chunks_sentences)
                    )
            else:
                summary_chunks_list.append(" ".join(summary))

            [texts_chunks_list.append(text) for text in text_chunks_list]
            [
                summaries_chunks_list.append(summary)
                for summary in summary_chunks_list
            ]
            # add every element one by one to main lists

    def pipeline(self, is_eval: bool):
        import tensorflow as tf

        if is_eval == False:

            raw_text = self.data_reader.df_train["text"].values.tolist()
            raw_summary = self.data_reader.df_train["summary"].values.tolist()
            val_raw_text = self.data_reader.df_val["text"].values.tolist()
            val_raw_summary = self.data_reader.df_val[
                "summary"
            ].values.tolist()

            # self.split_texts(raw_text.tolist(), raw_summary.tolist())
            text_token = self.tokenizer_text(raw_text)
            summary_token = self.tokenizer_text(raw_summary)
            val_text_token = self.tokenizer_text(val_raw_text)
            val_summary_token = self.tokenizer_text(val_raw_summary)
        else:
            val_raw_text = self.data_reader.df_val["text"].values.tolist()
            val_raw_summary = self.data_reader.df_val[
                "summary"
            ].values.tolist()
            val_text_token = self.tokenizer_text(val_raw_text)
            val_summary_token = self.tokenizer_text(val_raw_summary)
        # s = self.model(text_token[0:20], text_attention[0:20])
        # print(s)
        """Tokenized sentences even a sentence occurs, it type is List[List[int]]"""
        if is_eval == False:
            return (
                text_token,
                summary_token,
                val_text_token,
                val_summary_token,
            )
        else:
            return (val_text_token,val_summary_token)
            

    def decode_tokens(self, input_ids):
        return self.tokenizer.batch_decode(input_ids)

    def vectorizer(self, text: List[str]):
        # tokens:List[List[str]] = [sentence.split(' ') for sentence in text]
        # return count_vectorizer.fit_transform(text)
        return 0

    def split_text_by_sentences(self, texts: List[str]) -> List[List[str]]:
        # f = []
        # for x in texts:
        #     f.append(sent_tokenize(x))
        return [sent_tokenize(text) for text in texts]
        # return f

    def split_text_into_n_size_chunks(self, text: List[List[str]]):
        """text: List of sentences belongs to a text"""
        total_size: int = 0
        chunks_list: List = []
        chunk_sentences: List = []
        for sentence in text[0]:
            len_sentence = len(sentence.split(" "))
            if total_size + len_sentence <= 512:
                chunk_sentences.append(sentence)
                total_size += len_sentence
            else:
                chunks_list.append(chunk_sentences)
                chunk_sentences.clear()
                total_size = 0
        if total_size != 0:
            chunks_list.append(chunk_sentences)
        return chunks_list

    # def encode_chunks(
    #     self, chunks_list: List[List[str]], summary=List[List[str]]
    # ):
    #     """It should return text sentence with pair of summary sentences"""
    #     len_sentences_of_summary = len(summary[0][0])
    #     len_size_of_chunks = len(chunks_list)

    #     sentences_per_chunk = math.ceil(
    #         len_size_of_chunks / len_sentences_of_summary
    #     )

    #     tokenized_summary = self.tokenizer_text(summary)
    #     vector_summary = self.model(tokenized_summary["input_ids"])[
    #         "pooler_output"
    #     ]
    #     test = self.model(**tokenized_summary)

    #     for i in range(sentences_per_chunk):
    #         tokenized_text = self.tokenizer_text(chunks_list[i])
    #         vector_chunk = self.model.forward(tokenized_text["input_ids"])[
    #             "pooler_output"
    #         ]

    #         sorted_list = self.__cos_sim_n2(
    #             vector_chunk=vector_chunk,
    #             vector_summary=vector_summary,
    #             sentence_per_chunk=sentences_per_chunk,
    #         )
    #         self.log.info(sorted_list)
    #         self.log.info("f")

    def __cos_sim_n2(self, vector_chunk, vector_summary, sentence_per_chunk):
        cos_list = []
        for i in range(len(vector_chunk)):
            for j in range(len(vector_summary)):

                cos_list.append(
                    cosine_similarity(
                        vector_chunk[i], vector_summary[j], dim=-1
                    )
                )
        return list(sorted(cos_list))[-sentence_per_chunk:]
