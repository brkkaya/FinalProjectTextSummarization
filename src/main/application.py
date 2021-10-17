from src.services.Path_provider import GLobalPathProvider
from src.services.logger import Logger
from src.services.yaml_loader import YamlLoader


class Application(Logger):
    def __init__(self):
        self.log.info("Hello world")
        path_provider = GLobalPathProvider(file_name="hello")
        self.log.info((path_provider.path))
        YamlLoader()


if __name__ == "__main__":
    Application()
