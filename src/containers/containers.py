from DataRetrieve.wiki_data_parser import DataParser
from DataRetrieve.wiki_data_reader import DataReader

from DataRetrieve.wiki_data_scraper import DataScraper


class Container:

    parser = DataParser()
    data_retriever = DataScraper(parser=parser)
    data_reader = DataReader()
    
