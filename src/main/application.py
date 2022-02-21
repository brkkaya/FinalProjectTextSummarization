from src.containers.containers import Container


class Application:
    def __init__(self):
        app = Container()
        # app.data_retriever._run()
        # app.process.pipeline()
        app.train.test_classify()


if __name__ == "__main__":
    Application()
