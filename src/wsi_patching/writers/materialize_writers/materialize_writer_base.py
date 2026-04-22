import logging
from queue import Empty, Queue
from threading import Event
from typing import Any, Optional

from wsi_patching.core.types.util_types import EndOfQueue, EndOfStream
from wsi_patching.utils.logging_config import LogLevel, init_logging
from wsi_patching.utils.meta_typing import ContextAware, WriterMeta


class MaterializeWriterBase(ContextAware, metaclass=WriterMeta):
    """
    Base class for sink stages (writers). It hides multiprocessing queue handling and
    special control messages (EndOfStream / EndOfQueue).

    Implementers override:
      - open(self) -> None               # allocate resources; runs in writer process
      - write(self, batch: Any) -> None  # write a batch of items
      - on_end_of_stream(self) -> None   # optional per-slide finalization
      - close(self) -> None              # flush/close resources

    Single-process usage: writer(it) will consume the iterable and write everything.
    Multi-process usage: writer.start_writer(queue) will block and consume messages from producers.
    """

    def __init__(self) -> None:
        self._is_open = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.log = logging.getLogger(f"{cls.__name__}")

    # ----- lifecycle hooks for subclasses -----
    def open(self) -> None:
        """Allocate resources. Called in the writer process (or current process for single-process)."""
        pass

    def write(self, batch: Any) -> None:
        """Write a batch of items. Must be implemented by subclass."""
        raise NotImplementedError

    def on_end_of_stream(self) -> None:
        """Called when a producer finishes a slide. Default: no-op."""
        pass

    def close(self) -> None:
        """Flush and close resources."""
        pass

    # ----- helpers -----
    def _ensure_open(self) -> None:
        if not self._is_open:
            self.open()
            self._is_open = True

    # ----- Threaded consumer entrypoint -----
    def start_writer(
        self, queue: Queue, verbosity_level: LogLevel, stop_event: Optional[Event] = None, poll_s: float = 0.5
    ) -> None:
        """
        Multi-process: consume from a queue. Handles EndOfStream/EndOfQueue for you.
        Subclasses SHOULD NOT override this; implement open/write/on_end_of_stream/close instead.
        """
        init_logging(verbosity_level)
        self.log.info("Writer process started.")
        self._ensure_open()
        try:
            while True:
                try:
                    msg = queue.get(timeout=poll_s)
                except Empty:
                    if stop_event is not None and stop_event.is_set():
                        self.log.info("Stop event set; writer exiting.")
                        break
                    continue

                if isinstance(msg, EndOfQueue):
                    self.log.info("Writer received EndOfQueue (shutdown).")
                    break
                if isinstance(msg, EndOfStream):
                    self.on_end_of_stream()
                    continue
                self.write(msg)
        except KeyboardInterrupt:
            pass
        except Exception:
            self.log.exception("Unhandled exception in writer process:", exc_info=True)
        finally:
            try:
                self.close()
            except Exception:
                self.log.exception("Writer.close() raised during shutdown.", exc_info=True)
