from queue import Queue
from threading import Event
from typing import Any, Iterable

import pytest

from wsi_patching.core.types.util_types import EndOfQueue, EndOfStream
from wsi_patching.writers.stream_writers.stream_writer_base import StreamWriterBase


class _PassThroughWriter(StreamWriterBase):
    """Streams each item from the received batch, prefixing for easy verification."""

    def stream(self, batch: Iterable[Any]):
        for x in batch:
            yield f"Y:{x}"


class _BoomWriter(StreamWriterBase):
    """Raises when asked to stream, to test exception propagation."""

    def stream(self, batch: Iterable[Any]):
        raise RuntimeError("boom")


def _drain(gen):
    """Collect all items from a generator until it finishes."""
    return list(gen)


def test_init_logging_called(monkeypatch):
    called = {}

    def fake_init_logging(level):
        called["level"] = level

    monkeypatch.setattr(
        "wsi_patching.writers.stream_writers.stream_writer_base.init_logging", fake_init_logging, raising=True
    )

    q = Queue()
    q.put(EndOfQueue())
    w = _PassThroughWriter()
    _ = _drain(w.start_writer(q, verbosity_level="DEBUG", poll_s=0.05))
    assert called["level"] == "DEBUG"


def test_streams_batches_until_end_of_queue(caplog):
    q = Queue()
    w = _PassThroughWriter()

    # put a batch, then EndOfQueue
    q.put([1, 2, 3])
    q.put(EndOfQueue())

    with caplog.at_level("DEBUG"):
        out = _drain(w.start_writer(q, verbosity_level="INFO", poll_s=0.05))

    assert out == ["Y:1", "Y:2", "Y:3"]
    # sanity: debug log about batch size present
    assert any("Received batch of size 3" in rec.message for rec in caplog.records)
    # finished log present
    assert any("Writer finished." in rec.message for rec in caplog.records)


def test_ignores_end_of_stream_and_continues():
    q = Queue()
    w = _PassThroughWriter()

    q.put(EndOfStream())  # should be ignored
    q.put(["a"])  # actual batch to stream
    q.put(EndOfQueue())  # finish

    out = _drain(w.start_writer(q, verbosity_level="INFO", poll_s=0.05))
    assert out == ["Y:a"]


def test_stop_event_causes_clean_exit_without_items(caplog):
    q = Queue()
    evt = Event()
    evt.set()  # simulate external stop request when queue is empty

    w = _PassThroughWriter()
    with caplog.at_level("INFO"):
        out = _drain(w.start_writer(q, verbosity_level="INFO", stop_event=evt, poll_s=0.05))

    assert out == []
    # confirm the informative log is emitted
    assert any("Stop event set; writer exiting." in rec.message for rec in caplog.records)
    assert any("Writer finished." in rec.message for rec in caplog.records)


def test_exception_in_stream_is_propagated(caplog):
    q = Queue()
    q.put(["anything"])  # will trigger stream
    # ensure we terminate loop after exception (not strictly necessary; it will raise first)
    q.put(EndOfQueue())

    w = _BoomWriter()
    with pytest.raises(RuntimeError, match="boom"):
        _ = _drain(w.start_writer(q, verbosity_level="INFO", poll_s=0.05))

    # optional: ensure error was logged
    assert any("Exception in writer:" in rec.message for rec in caplog.records)
