from DataRetrieve.data_parser import DataParser
from DataRetrieve.data_reader import DataReader

from DataRetrieve.data_scraper import DataScraper
from Transformers.pre_process import PreProcess
from transformers import AutoModel


class Container:

    parser = DataParser()
    data_retriever = DataScraper(parser=parser)
    data_reader = DataReader()
    model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
    
    process = PreProcess(
        data_reader=data_reader,
        model=model,
    )
