from src.services.logger.logger import Logger
from src.services.path_provider import GLobalPathProvider
from src.services.yaml_loader import YamlLoader


class BaseService:

    log = Logger().log
    config = YamlLoader().config
    global_path_provider = GLobalPathProvider()

    @property
    def number_of_heads(self):
        return self.config.params.model.number_of_heads

    @property
    def vocab_size(self):
        return self.config.params.model.vocab_size

    @property
    def model_dim(self):
        return self.config.params.model.model_dim

    @property
    def seq_dim(self):
        return self.config.params.model.seq_dim

    @property
    def decoder_number(self):
        return self.config.params.model.number_of_decoder

    @property
    def epsilon(self):
        return self.config.params.model.epsilon

    @property
    def dropout_rate(self):
        return self.config.params.model.dropout_rate
