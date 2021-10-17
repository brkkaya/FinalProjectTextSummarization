from DataRetrieve.wiki_data_parser import DataParser
from src.services.base_service import BaseService


class DataScraper(BaseService):
    def __init__(self, parser=DataParser) -> None:
        self.parser = parser
        self.log.info("Test Scraper Init")

    def _run(self):
        self.parser._run()
        self.log.info("Test Scraper Run")
