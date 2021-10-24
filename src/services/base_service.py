from src.services.logger.logger import Logger
from src.services.yaml_loader import YamlLoader


class BaseService:

    log = Logger().log
    config = YamlLoader().config
    