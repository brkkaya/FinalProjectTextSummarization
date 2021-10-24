from box import Box

from src.services.path_provider import GLobalPathProvider


class YamlLoader:
    __path_provider = GLobalPathProvider(file_name="hello")
    config = Box.from_yaml(filename=__path_provider.path + "/src/main/config.yaml")
