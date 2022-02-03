from src.services.logger.logger import Logger
from src.services.path_provider import GLobalPathProvider
from src.services.yaml_loader import YamlLoader


class BaseService:

    log = Logger().log
    config = YamlLoader().config
    global_path_provider = GLobalPathProvider()
