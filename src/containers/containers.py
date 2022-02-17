from data_retrieve.data_parser import DataParser
from data_retrieve.data_reader import DataReader

from data_retrieve.data_scraper import DataScraper
from summary_generator.pre_process import PreProcess
from transformers import TFAutoModel, AutoConfig


class Container:

    parser = DataParser()
    data_retriever = DataScraper(parser=parser)
    data_reader = DataReader()
    model_config = AutoConfig.from_pretrained(
        "dbmdz/bert-base-turkish-128k-cased", output_hidden_states=True
    )
    model = TFAutoModel.from_pretrained(
        "dbmdz/bert-base-turkish-128k-cased", config=model_config
    )

    process = PreProcess(
        data_reader=data_reader,
        model=model,
    )
