import logging
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Event as MpEvent
from queue import Empty
from typing import Any, Iterable, Optional

from wsi_patching.core.types.util_types import EndOfQueue, EndOfStream
from wsi_patching.utils.logging_config import LogLevel, init_logging
from wsi_patching.utils.meta_typing import ContextAware, WriterMeta


class StreamWriterBase(ContextAware, metaclass=WriterMeta):
    """
    Base class for generator sink stages (writers).
    To be used for pipelines that are written to memory, instead of to disk.
    By using a generator writer, you can process your patches in a streaming fashion.
    This is useful for large datasets that do not fit into memory, or for real-time processing

    Implementers override:
      - write(self, batch: Any) -> None  # write a batch of items

    Single-process usage: writer(it) will consume the iterable and write everything.
    Multi-process usage: writer.start_writer(queue) will block and consume messages from producers.
    """

    def __init__(self) -> None:
        self._is_open = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.log = logging.getLogger(f"{cls.__name__}")

    def stream(self, batch: Iterable[Any]) -> Iterable[Any]:
        """Stream a batch of items. Must be implemented by subclass."""
        raise NotImplementedError

    def start_writer(
        self, queue: MPQueue, verbosity_level: LogLevel, stop_event: Optional[MpEvent] = None, poll_s: float = 0.5
    ) -> Iterable[Any]:
        """
        Start the writer process. This will block and consume messages from the queue.
        It will yield items as they are written.
        """
        if not isinstance(queue, MPQueue):
            raise ValueError("queue must be a multiprocessing.Queue")

        init_logging(verbosity_level)

        self.log.info("Starting generator writer...")
        try:
            while True:
                try:
                    msg = queue.get(timeout=poll_s)
                except Empty:
                    if stop_event is not None and stop_event.is_set():
                        self.log.info("Stop event set; writer exiting.")
                        break
                    continue

                if isinstance(msg, EndOfStream):
                    self.log.info("Received EndOfStream signal.")
                    continue
                elif isinstance(msg, EndOfQueue):
                    self.log.info("Received EndOfQueue signal. Finishing writer.")
                    break
                else:
                    self.log.debug(f"Received batch of size {len(msg)}")
                    for item in self.stream(msg):
                        yield item

        except Exception as e:
            self.log.exception(f"Exception in writer: {e}")
            raise
        finally:
            self.log.info("Writer finished.")
