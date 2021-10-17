from src.services.logger import Logger


class Application(Logger):
    def __init__(self):
        self.log.info("Hello world")


if __name__ == "__main__":
    Application()
