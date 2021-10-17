from src.services.base_service import BaseService


class Application(BaseService):
    def __init__(self):
        self.log.info("Hello world")
        
        self.log.info(self.config.param)
        


if __name__ == "__main__":
    Application()
