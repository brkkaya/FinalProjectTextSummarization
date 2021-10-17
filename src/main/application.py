from src.services.logger import Logger
from src.services.yaml_loader import YamlLoader


class Application(Logger):
    def __init__(self):
        self.log.info("Hello world")
        YamlLoader()


if __name__ == "__main__":
    Application()
