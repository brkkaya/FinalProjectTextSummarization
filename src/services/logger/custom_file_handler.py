from logging import FileHandler
from typing import Union


class CustomFileHandler(FileHandler):
    def __init__(
        self,
        filename,
        mode: str = None,
        encoding: Union[str, None] = None,
        delay: bool = None,
        errors: Union[str, None] =None,
    ) -> None:

        super().__init__(
            filename=filename,
            mode=mode,
            encoding=encoding,
            delay=delay,
            errors=errors,
        )
