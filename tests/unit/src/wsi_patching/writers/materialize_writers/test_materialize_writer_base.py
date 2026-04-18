from queue import Queue
from threading import Event
from typing import Any

from wsi_patching.core.types.util_types import EndOfQueue, EndOfStream
from wsi_patching.writers.materialize_writers.materialize_writer_base import MaterializeWriterBase


class _ProbeWriter(MaterializeWriterBase):
    """Test double that records lifecycle calls and data written."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.written = []
        self.open_count = 0
        self.closed = False
        self.eos_count = 0

    def open(self):
        self.calls.append("open")
        self.open_count += 1

    def write(self, batch: Any):
        self.calls.append(("write", batch))
        self.written.append(batch)

    def on_end_of_stream(self):
        self.calls.append("eos")
        self.eos_count += 1

    def close(self):
        self.calls.append("close")
        self.closed = True


class _BoomOnWriteWriter(_ProbeWriter):
    """Raises from write() once to test exception handling; still ensure close() is called."""

    def write(self, batch: Any):
        super().write(batch)  # record the attempt
        raise RuntimeError("boom in write")


def _drain_start(writer, queue, **kwargs):
    """Run start_writer() and return after it finishes."""
    # start_writer yields nothing; it returns None
    return writer.start_writer(queue, **kwargs)


def test_init_logging_called(monkeypatch):
    called = {}

    def fake_init(level):
        called["level"] = level

    monkeypatch.setattr(
        "wsi_patching.writers.materialize_writers.materialize_writer_base.init_logging", fake_init, raising=True
    )

    q = Queue()
    q.put(EndOfQueue())

    w = _ProbeWriter()
    _drain_start(w, q, verbosity_level="DEBUG", poll_s=0.05)

    assert called["level"] == "DEBUG"


def test_open_once_write_batches_then_close(caplog):
    q = Queue()
    q.put(["a", "b"])  # batch #1
    q.put(["c"])  # batch #2
    q.put(EndOfQueue())  # stop

    w = _ProbeWriter()
    with caplog.at_level("INFO"):
        _drain_start(w, q, verbosity_level="INFO", poll_s=0.05)

    # open happens exactly once, before any writes
    assert w.open_count == 1
    assert w.calls[0] == "open"

    # write called for each batch element (batch object is the list itself per code)
    assert ("write", ["a", "b"]) in w.calls
    assert ("write", ["c"]) in w.calls
    assert w.written == [["a", "b"], ["c"]]

    # closed in finally
    assert w.closed
    assert w.calls[-1] == "close"
    # saw start + shutdown logs
    assert any("Writer process started." in rec.message for rec in caplog.records)
    assert any("Writer received EndOfQueue (shutdown)." in rec.message for rec in caplog.records)


def test_end_of_stream_triggers_on_end_of_stream_without_write():
    q = Queue()
    q.put(EndOfStream())  # should call on_end_of_stream, not write
    q.put(["x"])  # then a real batch
    q.put(EndOfQueue())

    w = _ProbeWriter()
    _drain_start(w, q, verbosity_level="INFO", poll_s=0.05)

    # ensure eos recorded
    assert "eos" in w.calls
    # ensure write happened for the batch only once
    assert ("write", ["x"]) in w.calls
    # no write for the eos marker
    assert not any(call == ("write", EndOfStream()) for call in w.calls)


def test_stop_event_causes_clean_exit_when_queue_empty(caplog):
    q = Queue()
    evt = Event()
    evt.set()  # simulate external stop while queue empty

    w = _ProbeWriter()
    with caplog.at_level("INFO"):
        _drain_start(w, q, verbosity_level="INFO", stop_event=evt, poll_s=0.05)

    # no writes, but open then close should occur
    assert w.open_count == 1
    assert w.closed
    assert any("Stop event set; writer exiting." in rec.message for rec in caplog.records)


def test_exception_in_write_is_logged_and_close_still_called(caplog):
    q = Queue()
    q.put(["boom"])  # will cause write to raise
    q.put(EndOfQueue())  # eventual shutdown

    w = _BoomOnWriteWriter()
    with caplog.at_level("ERROR"):
        _drain_start(w, q, verbosity_level="INFO", poll_s=0.05)

    # The writer should have attempted to write once
    assert ("write", ["boom"]) in w.calls
    # Error logged by exception handler
    assert any("Unhandled exception in writer process:" in rec.message for rec in caplog.records)
    # Must still close in finally
    assert w.closed


def test_close_exception_is_logged_not_raised(monkeypatch, caplog):
    class _CloseBoomWriter(_ProbeWriter):
        def close(self):
            super().close()  # mark closed True to observe call
            raise RuntimeError("boom in close")

    q = Queue()
    q.put(EndOfQueue())

    w = _CloseBoomWriter()
    with caplog.at_level("ERROR"):
        _drain_start(w, q, verbosity_level="INFO", poll_s=0.05)

    # close was attempted (flag set by super().close()), and exception logged
    assert w.closed
    assert any("Writer.close() raised during shutdown." in rec.message for rec in caplog.records)
