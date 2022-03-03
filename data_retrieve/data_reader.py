from src.services.base_service import BaseService
import pandas as pd
import numpy as np
import json


class DataReader(BaseService):
    def __init__(self) -> None:
        self.log.info("Test Parser init")
        np.random.seed(42)
        with open(
            f"{self.global_path_provider.path}/turkish_train.jsonl", "r"
        ) as jsonl_file:

            self._df_train = pd.DataFrame(
                [json.loads(json_line) for json_line in jsonl_file],
                columns=["text", "summary"],
            )
            if self.config.params.truncate:
                mask = (
                    self._df_train.iloc[:, 0].str.split(" ").str.len() >= 512
                )
                self._df_train = self._df_train.loc[mask, :]

        with open(
            f"{self.global_path_provider.path}/turkish_test.jsonl", "r"
        ) as jsonl_file:

            self._df_test = pd.DataFrame(
                [json.loads(json_line) for json_line in jsonl_file],
                columns=["text", "summary"],
            )
            if self.config.params.truncate:
                mask = self._df_val.iloc[:, 0].str.split(" ").str.len() >= 512
                self._df_val = self._df_val.loc[mask, :]
        with open(
            f"{self.global_path_provider.path}/turkish_val.jsonl", "r"
        ) as jsonl_file:

            self._df_val = pd.DataFrame(
                [json.loads(json_line) for json_line in jsonl_file],
                columns=["text", "summary"],
            )
            if self.config.params.truncate:
                mask = self._df_val.iloc[:, 0].str.split(" ").str.len() >= 512
                self._df_val = self._df_val.loc[mask, :]

    @property
    def df_train(self) -> pd.DataFrame:
        return self._df_train

    @property
    def df_val(self) -> pd.DataFrame:
        return self._df_val

    @property
    def df_test(self) -> pd.DataFrame:
        return self._df_test
