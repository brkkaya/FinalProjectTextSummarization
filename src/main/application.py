from src.services.Path_provider import GLobalPathProvider
from src.services.logger import Logger


class Application(Logger):
    def __init__(self):
        self.log.info("Hello world")
        path_provider = GLobalPathProvider(file_name="hello")
        self.log.info((path_provider.path))


if __name__ == "__main__":
    Application()
