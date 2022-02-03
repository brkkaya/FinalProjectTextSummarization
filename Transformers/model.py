from configparser import ConfigParser
import transformers
from src.services.base_service import BaseService
from DataRetrieve.data_reader import DataReader
import torch

class SummarizationModel(BaseService):
    def __init__(self, data_reader: DataReader) -> None:
        super().__init__()
        self.data_reader = data_reader
