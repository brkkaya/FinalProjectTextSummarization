import logging
from src.services.custom_logger import CustomFormatter


class Logger:
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    handler.setFormatter(CustomFormatter())

    log.addHandler(handler)
    log.info("Logger initialized")
