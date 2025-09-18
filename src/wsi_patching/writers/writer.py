import logging
from multiprocessing.queues import Queue as MPQueue
from typing import Any, Union

from wsi_patching.utils.logging_config import init_logging
from wsi_patching.utils.meta_typing import WriterMeta
from wsi_patching.utils.types import EndOfQueue, EndOfStream, Patch


class WriterBase(metaclass=WriterMeta):
    """
    Base class for sink stages (writers). It hides multiprocessing queue handling and
    special control messages (EndOfStream / EndOfQueue).

    Implementers override:
      - open(self) -> None               # allocate resources; runs in writer process
      - write(self, sample: Any) -> None # write a single item
      - on_end_of_stream(self) -> None   # optional per-slide finalization
      - close(self) -> None              # flush/close resources

    Single-process usage: writer(it) will consume the iterable and write everything.
    Multi-process usage: writer.start_writer(queue) will block and consume messages from producers.
    """

    # By default, accept anything; concrete writers should annotate __call__ to set input_type
    input_type: Any = object
    output_type: Any = object

    def __init__(self) -> None:
        self._is_open = False

    # ----- lifecycle hooks for subclasses -----
    def open(self) -> None:
        """Allocate resources. Called in the writer process (or current process for single-process)."""
        pass

    def write(self, sample: Any) -> None:
        """Write one item. Must be implemented by subclass."""
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

    # ----- Multiprocess consumer entrypoint -----
    def start_writer(self, queue: MPQueue) -> None:
        """
        Multi-process: consume from a queue. Handles EndOfStream/EndOfQueue for you.
        Subclasses SHOULD NOT override this; implement open/write/on_end_of_stream/close instead.
        """
        init_logging()
        logging.info("Writer process started.")
        self._ensure_open()
        try:
            while True:
                msg: Union[Patch, EndOfStream, EndOfQueue] = queue.get()
                if isinstance(msg, EndOfQueue):
                    logging.info("Writer received EndOfQueue (shutdown).")
                    break
                if isinstance(msg, EndOfStream):
                    self.on_end_of_stream()
                    continue
                self.write(msg)
        except KeyboardInterrupt:
            pass
        except Exception:
            logging.exception("Unhandled exception in writer process:", exc_info=True)
        finally:
            try:
                self.close()
            except Exception:
                logging.exception("Writer.close() raised during shutdown.", exc_info=True)
