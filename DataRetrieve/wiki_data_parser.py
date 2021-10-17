from src.services.base_service import BaseService


class DataParser(BaseService):
    def __init__(self) -> None:
        self.log.info("Test Parser init")

    def _run(self):
        self.log.info("Test Parser Run")