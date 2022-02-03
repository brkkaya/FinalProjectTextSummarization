import numpy as np
from src.services.base_service import BaseService
import pandas as pd
import re


class PreProcess(BaseService):
    def __init__(self) -> None:
        super().__init__()

    def import_data(self):
        pass

    def change_commas(self, text: str) -> str:
        return self.change_commas_in_text.sub("'", text)

    def pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def compute_cosine_similarity(self, text: str, summary: str):
        pass

    def split_text(self, text: str):
        pass
