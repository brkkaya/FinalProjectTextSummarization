from src.services.base_service import BaseService

#TODO Remove Non latin chars from https://stackoverflow.com/questions/23680976/python-removing-non-latin-characters

#TODO get the statistics of headers of sub topics. Beware to remove them.
class DataParser(BaseService):
    def __init__(self) -> None:
        self.log.info("Test Parser init")

    def _run(self):
        self.log.info("Test Parser Run")