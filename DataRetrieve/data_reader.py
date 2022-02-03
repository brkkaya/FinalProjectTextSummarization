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
            ).sample(150000)

        with open(
            f"{self.global_path_provider.path}/tu_test.jsonl", "r"
        ) as jsonl_file:

            self._df_test = pd.DataFrame(
                [json.loads(json_line) for json_line in jsonl_file],
                columns=["text", "summary"],
            ).sample(150000)

        with open(
            f"{self.global_path_provider.path}/tu_val.jsonl", "r"
        ) as jsonl_file:

            self._df_val = pd.DataFrame(
                [json.loads(json_line) for json_line in jsonl_file],
                columns=["text", "summary"],
            ).sample(150000)

    @property
    def df_train(self) -> pd.DataFrame:
        return self._df_train

    @property
    def df_val(self) -> pd.DataFrame:
        return self._df_val

    @property
    def df_test(self) -> pd.DataFrame:
        return self._df_test
