import logging
import sys
from typing import Union

from typing_extensions import Literal

LogLevel = Union[int, Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]]


def init_logging(verbosity_level: LogLevel) -> None:
    logging.basicConfig(
        level=verbosity_level,
        format="[%(asctime)s] [%(levelname)s] [%(processName)s] [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
