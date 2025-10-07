import logging

from wsi_patching.core.types.util_types import EndOfQueue, EndOfStream
from wsi_patching.writers.writer_base import WriterBase


class FakeQueue:
    """Simple FIFO with a .get() API compatible enough for start_writer()."""

    def __init__(self, items):
        self._items = list(items)

    def get(self):
        return self._items.pop(0)


# ---- basic subclass to observe calls ----
class Patch:
    pass


class DummyWriter(WriterBase):
    def __init__(self):
        super().__init__()
        self.open_count = 0
        self.closed = False
        self.eos_count = 0
        self.written = []

    def open(self) -> None:
        self.open_count += 1

    def write(self, sample: Patch):
        self.written.append(sample)

    def on_end_of_stream(self) -> None:
        self.eos_count += 1

    def close(self) -> None:
        self.closed = True


def test_logger_is_attached_via___init_subclass__():
    class MyWriter(WriterBase):
        pass

    # class-level logger named after the class
    assert isinstance(MyWriter.log, logging.Logger)
    assert MyWriter.log.name == "MyWriter"


def test__ensure_open_opens_once_only():
    w = DummyWriter()
    assert w.open_count == 0
    w._ensure_open()
    w._ensure_open()
    assert w.open_count == 1  # only the first call opens


def test_start_writer_happy_path_processes_messages_and_logs(caplog):
    caplog.set_level(logging.INFO)
    w = DummyWriter()
    messages = [
        1,
        2,
        EndOfStream(),  # triggers on_end_of_stream
        3,
        EndOfQueue(),  # triggers shutdown
    ]
    q = FakeQueue(messages)

    w.start_writer(q, verbosity_level="INFO")

    # wrote all non-control messages
    assert w.written == [1, 2, 3]
    # open called once, close called
    assert w.open_count == 1
    assert w.closed is True
    # EndOfStream counted exactly once
    assert w.eos_count == 1

    # logging assertions via caplog (robust, handler-agnostic)
    txt = caplog.text
    assert "Writer process started." in txt
    assert "Writer received EndOfQueue (shutdown)." in txt


def test_start_writer_logs_exception_from_write_and_still_closes(caplog):
    caplog.set_level(logging.INFO)

    class BoomWriter(DummyWriter):
        def write(self, sample: str):
            if sample == "boom":
                raise RuntimeError("kaboom")
            super().write(sample)

    w = BoomWriter()
    q = FakeQueue(["ok", "boom", EndOfQueue()])

    # should not raise; exception is caught and logged, then close() in finally
    w.start_writer(q, verbosity_level="INFO")

    # wrote only the first item; after exception loop exits to finally
    assert w.written == ["ok"]
    assert w.closed is True

    assert "Unhandled exception in writer process:" in caplog.text


def test_start_writer_logs_exception_from_close_but_does_not_raise(caplog):
    caplog.set_level(logging.INFO)

    class CloseBoomWriter(DummyWriter):
        def close(self) -> None:
            # mark that we attempted to close, then raise
            self.closed = True
            raise RuntimeError("close-fail")

    w = CloseBoomWriter()
    q = FakeQueue([EndOfQueue()])

    # should not raise even though close() fails
    w.start_writer(q, verbosity_level="INFO")
    assert w.closed is True  # attempted

    assert "Writer.close() raised during shutdown." in caplog.text
