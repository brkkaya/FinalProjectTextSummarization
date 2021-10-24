import logging
from src.services.logger.custom_file_handler import CustomFileHandler
from src.services.logger.custom_log_formatter import CustomFormatter
from src.services.path_provider import GLobalPathProvider


class Logger:
    import re

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    file_handler = CustomFileHandler(
        filename=GLobalPathProvider(file_name=None).log_path,
        mode="w",
        encoding="utf-8",
    )
    # file_handler = logging.FileHandler(
    #     GLobalPathProvider(file_name="testing").log_path
    # )
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    stream_handler.setFormatter(CustomFormatter())
    file_formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d] [%(levelname)-s] [%(module)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    #  date_format =
    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    log.info("Logger initialized")
