import numpy as np
from src.services.base_service import BaseService
import pandas as pd
import re
from transformers import AutoTokenizer, TFAutoModel
from data_retrieve.data_reader import DataReader
from typing import Dict, List
import math
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
        texts: List[List[str]],
    ):

        return self.split_tokenizer_out(
            [
                self.tokenizer.encode_plus(
                    text=text,
                    return_tensors="tf",
                    padding="max_length",
                    max_length=self.seq_dim,
                    return_attention_mask=True,
                    truncation=True,
                )
                for text in texts
            ]
        )

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

    def pipeline(self):
        import tensorflow as tf

        raw_text = self.data_reader.df_train["text"].values
        raw_summary = self.data_reader.df_train["summary"].values
        text_token, text_attention, text_ids = self.tokenizer_text(raw_text)
        summary_token, summary_attention, summary_ids = self.tokenizer_text(
            raw_summary
        )

        # s = self.model(text_token[0:20], text_attention[0:20])
        # print(s)
        """Tokenized sentences even a sentence occurs, it type is List[List[int]]"""

        return (
            text_token,
            text_attention,
            summary_token,
            summary_attention,
        )

    def decode_tokens(self, input_ids):
        return [
            self.log.info(self.tokenizer.batch_decode(tokens))
            for tokens in input_ids
        ]

    def vectorizer(self, text: List[str]):
        # tokens:List[List[str]] = [sentence.split(' ') for sentence in text]
        # return count_vectorizer.fit_transform(text)
        return 0

    def split_text_by_sentences(self, texts: List[str]) -> List[List[str]]:
        f = []
        for x in texts:
            f.append(sent_tokenize(x))
        # return [sent_tokenize(text) for text in texts]
        return f

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

    def encode_chunks(
        self, chunks_list: List[List[str]], summary=List[List[str]]
    ):
        """It should return text sentence with pair of summary sentences"""
        len_sentences_of_summary = len(summary[0])
        len_size_of_chunks = len(chunks_list)

        sentences_per_chunk = math.ceil(
            len_size_of_chunks / len_sentences_of_summary
        )

        tokenized_summary = self.tokenizer_text(summary)
        vector_summary = self.model(tokenized_summary["input_ids"])[
            "pooler_output"
        ]
        test = self.model(**tokenized_summary)

        for i in range(sentences_per_chunk):
            tokenized_text = self.tokenizer_text(chunks_list[i])
            vector_chunk = self.model.forward(tokenized_text["input_ids"])[
                "pooler_output"
            ]

            sorted_list = self.__cos_sim_n2(
                vector_chunk=vector_chunk,
                vector_summary=vector_summary,
                sentence_per_chunk=sentences_per_chunk,
            )
            self.log.info(sorted_list)
            self.log.info("f")

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
