from src.services.base_service import BaseService
import pandas as pd

class DataReader(BaseService):
    def __init__(self) -> None:
        self.log.info("Test Parser init")
        
    def _run(self):
        self.log.info("Test Parser Run")
        df = pd.read_csv('wiki_data.csv')
        print('h')

# yonca avm mamak