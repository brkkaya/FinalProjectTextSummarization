from src.containers.containers import Container

# from src.services.base_service import BaseService


class Application:
    def __init__(self):
        # self.log.info("Hello world")

        # self.log.info(self.config.param)
        app = Container()
        app.data_retriever._run()
        # app.wire([app])


if __name__ == "__main__":
    Application()
