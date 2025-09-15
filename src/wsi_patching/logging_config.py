import logging
import sys


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(processName)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
