from src.containers.containers import Container


class Application:
    def __init__(self):
        app = Container()
        app.data_retriever._run()
        # app.data_reader._run()


if __name__ == "__main__":
    Application()
