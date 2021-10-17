import logging
import typing
from colorlog import ColoredFormatter
from colorlog.formatter import LogColors, SecondaryLogColors


class CustomFormatter(ColoredFormatter):
    def __init__(
        self,
        fmt: typing.Optional[str] = None,
        datefmt: typing.Optional[str] = None,
        style: str = "%",
        log_colors: typing.Optional[LogColors] = None,
        reset: bool = True,
        secondary_log_colors: typing.Optional[SecondaryLogColors] = None,
        validate: bool = True,
        stream: typing.Optional[typing.IO] = None,
        no_color: bool = False,
    ) -> None:

        format_str = "%(log_color)s[%(asctime)s]%(reset)s %(cyan)s[%(levelname)-s]%(reset)s %(yellow)s[%(module)s]%(reset)s: %(green)s%(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

        colors = {
            "DEBUG": "green",
            "INFO": "cyan",
            "WARNING": "bold_yellow",
            "ERROR": "bold_red",
            "CRITICAL": "bold_purple",
        }
        super().__init__(
            fmt=format_str,
            datefmt=date_format,
            style=style,
            log_colors=colors,
            reset=reset,
            secondary_log_colors=secondary_log_colors,
            validate=validate,
            stream=stream,
            no_color=no_color,
        )
