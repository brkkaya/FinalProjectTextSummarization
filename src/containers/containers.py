from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    def __init__(self) -> None:
        super().__init__()
        
        self.config_service =providers.Configuration()
        