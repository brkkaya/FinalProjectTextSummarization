from DataRetrieve.data_parser import DataParser
from DataRetrieve.data_reader import DataReader

from DataRetrieve.data_scraper import DataScraper
from Transformers.pre_process import PreProcess

class Container:

    parser = DataParser()
    data_retriever = DataScraper(parser=parser)
    data_reader = DataReader()
    process = PreProcess()

    
    

