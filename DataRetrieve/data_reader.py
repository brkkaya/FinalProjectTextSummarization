from src.services.base_service import BaseService
import pandas as pd
import numpy as np
import json


class DataReader(BaseService):
    def __init__(self) -> None:
        self.log.info("Test Parser init")
        np.random.seed(42)
        with open(
            f"{self.global_path_provider.path}/tu_train.jsonl", "r"
        ) as jsonl_file:

            self._df_train = pd.DataFrame(
                [json.loads(json_line) for json_line in jsonl_file],
                columns=["text", "summary"],
            )[:50]
            if self.config.params.truncate:
                self._df_train = self._df_train.iloc[
                    len(self._df_train.iloc[:, 0]) <= 512, :
                ]

        with open(
            f"{self.global_path_provider.path}/tu_test.jsonl", "r"
        ) as jsonl_file:

            self._df_test = pd.DataFrame(
                [json.loads(json_line) for json_line in jsonl_file],
                columns=["text", "summary"],
            )
            if self.config.params.truncate:
                self._df_test = self._df_test.iloc[
                    len(self._df_test.iloc[:, 0]) <= 512, :
                ]
        with open(
            f"{self.global_path_provider.path}/tu_val.jsonl", "r"
        ) as jsonl_file:

            self._df_val = pd.DataFrame(
                [json.loads(json_line) for json_line in jsonl_file],
                columns=["text", "summary"],
            )
            if self.config.params.truncate:
                self._df_val = self._df_val.iloc[
                    len(self._df_val.iloc[:, 0]) <= 512, :
                ]

    @property
    def df_train(self) -> pd.DataFrame:
        return self._df_train

    @property
    def df_val(self) -> pd.DataFrame:
        return self._df_val

    @property
    def df_test(self) -> pd.DataFrame:
        return self._df_test
