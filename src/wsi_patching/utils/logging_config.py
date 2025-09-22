import logging
import sys

from typing_extensions import Literal

LogLevel = int | Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]


def init_logging(verbosity_level: LogLevel) -> None:
    logging.basicConfig(
        level=verbosity_level,
        format="[%(asctime)s] [%(processName)s] [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
