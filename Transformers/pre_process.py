import numpy as np
from src.services.base_service import BaseService
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
from DataRetrieve.data_reader import DataReader
from typing import List
import math
from torch.nn.functional import cosine_similarity
from nltk.tokenize import sent_tokenize


class PreProcess(BaseService):
    def __init__(self, data_reader: DataReader, model: AutoModel) -> None:
        super().__init__()
        self.data_reader = data_reader
        self.tokenizer = AutoTokenizer.from_pretrained(
            "dbmdz/bert-base-turkish-128k-uncased"
        )
        self.model = model

    def tokenizer_text(
        self,
        text: List[str],
    ):
        return self.tokenizer(text=text, return_tensors="pt", padding=True)

    def tokenizer_summary(self, summary: List[str]):
        return self.tokenizer(
            text=summary,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

    def pipeline(self):
        # f = self.tokenizer_text(self.data_reader.df_train["text"].tolist())
        s = [self.data_reader.df_train["text"].tolist()[2]]
        s2 = [self.data_reader.df_train["summary"].tolist()[2]]
        f = self.split_text_by_sentences(s)
        f2 = self.split_text_by_sentences(s2)
        p = self.split_text_into_n_size_chunks(f)
        d = self.encode_chunks(p, f2)

        self.log.info("Tokenizer takes list of str")

    def pipeline2(self):
        x_train = [self.data_reader.df_train["text"].tolist()]
        x_test = [self.data_reader.df_train["summary"].tolist()]
        
    def vectorizer(self, text: List[str]):
        # tokens:List[List[str]] = [sentence.split(' ') for sentence in text]
        # return count_vectorizer.fit_transform(text)
        return 0

    def split_text_by_sentences(self, texts: List[str]) -> List[List[str]]:
        return [sent_tokenize(text) for text in texts]

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

        tokenized_summary = self.tokenizer_text(summary[0])
        vector_summary = self.model.forward(tokenized_summary["input_ids"])[
            "pooler_output"
        ]

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


## if longer than 768 to around

##It is not possible to find similarity between two sentences by cosine

# Use pooled output to
#%%
