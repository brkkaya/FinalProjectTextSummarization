from box import Box

class YamlLoader:
    config = Box.from_yaml(filename=Globals().path +'config.yaml')
