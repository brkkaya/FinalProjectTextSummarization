import graypy
import logging

class BaseService:
    def __init__(self) -> None:
        self.log = logging.getLogger()