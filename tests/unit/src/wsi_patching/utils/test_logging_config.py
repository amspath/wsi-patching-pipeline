import logging
import re
import sys

import pytest

from wsi_patching.utils.logging_config import init_logging


def _reset_root_logging():
    # Ensure basicConfig() won't no-op due to existing handlers
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.NOTSET)


@pytest.mark.parametrize("level", [logging.INFO, "INFO"])
def test_init_logging_sets_level_and_stdout_handler(level, capsys):
    _reset_root_logging()
    init_logging(level)

    root = logging.getLogger()
    # exactly one handler, a StreamHandler to sys.stdout
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stdout

    # emits to stdout with our format (contains [processName] [logger name] message)
    logger = logging.getLogger("test_logger")
    logger.info("hello world")
    out = capsys.readouterr().out
    # timestamp + process + name + message
    # Example: [2025-09-23 12:34:56,789] [MainProcess] [test_logger] hello world
    assert "hello world" in out
    assert "[MainProcess]" in out
    assert "[test_logger]" in out
    # crude check that it starts with a timestamp in brackets
    assert re.match(r"^\[\d{4}-\d{2}-\d{2}", out)


def test_init_logging_is_idempotent_does_not_duplicate_handlers():
    _reset_root_logging()
    init_logging("DEBUG")
    first_handlers = list(logging.getLogger().handlers)
    # Call again: basicConfig should not add another handler if one exists
    init_logging("DEBUG")
    second_handlers = list(logging.getLogger().handlers)
    assert len(first_handlers) == 1
    assert len(second_handlers) == 1
    assert first_handlers[0] is second_handlers[0]
