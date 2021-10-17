

from src.services.logger import Logger


class BaseService:
    def __init__(self) -> None:
        self.log = Logger.log