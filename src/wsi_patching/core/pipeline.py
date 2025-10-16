import logging
import multiprocessing as mp
import sys
import threading
from dataclasses import dataclass, field
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from queue import Full
from threading import Thread
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

from pyparsing import Callable

from wsi_patching.core.types.util_types import EndOfQueue, EndOfStream
from wsi_patching.utils.logging_config import LogLevel, init_logging
from wsi_patching.utils.meta_typing import ContextAware, PipelineContext, StageMeta
from wsi_patching.utils.profiling import PipelineProfileAggregator, Profiler, get_current_profiler, set_current_profiler
from wsi_patching.writers.materialize_writers.materialize_writer_base import MaterializeWriterBase
from wsi_patching.writers.stream_writers.stream_writer_base import StreamWriterBase

try:
    # Consistent across 3.8/3.9 and supports typing_extensions constructs
    from typing_extensions import get_args, get_origin
except ImportError:  # 3.10+ or environments without typing_extensions
    from typing import get_args, get_origin


def _type_options(t: Any) -> Tuple[type, ...]:
    """Normalize to tuple of types; (object,) means 'anything'."""
    if t is object or t is None:
        return (object,)
    if isinstance(t, tuple):
        return t
    origin = get_origin(t)
    if origin in (Union,) or str(origin).endswith("types.UnionType"):
        return tuple(a for a in get_args(t) if a is not type(None))
    return (t,)


def _is_compatible(prod_out: Any, cons_in: Any) -> bool:
    outs, ins = _type_options(prod_out), _type_options(cons_in)
    for o in outs:
        for i in ins:
            try:
                if i is object or o is object or issubclass(o, i):
                    return True
            except TypeError:
                pass
    return False


def _tname(t: Any) -> str:
    try:
        return t.__name__
    except Exception:
        return str(t)


class Stage(ContextAware, metaclass=StageMeta):
    input_type: Any = object
    output_type: Any = object

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.log = logging.getLogger(f"{cls.__name__}")

    def __call__(self, it: Iterable[Any]) -> Iterable[Any]:
        raise NotImplementedError

    def then(self, nxt: "Stage") -> "Pipeline":
        return Pipeline([self, nxt])

    def for_slide(self, slide_path: str) -> "Stage":
        return self

    def get_current_profiler(self) -> Optional[Profiler]:
        return get_current_profiler()


# -------- Pipeline --------
@dataclass
class Pipeline(Stage):
    stages: List[Stage]
    writer: Optional[Union[MaterializeWriterBase, StreamWriterBase]] = None
    prof_agg: Optional["PipelineProfileAggregator"] = None
    _context: PipelineContext = field(default_factory=PipelineContext)
    _runtime_type_asserts: bool = True

    def __init__(
        self,
        stages: List[Stage],
        writer: Optional[Union[MaterializeWriterBase, StreamWriterBase]] = None,
        prof_agg: Optional["PipelineProfileAggregator"] = None,
        context: Optional[PipelineContext] = None,
    ):
        self.stages = stages
        self.writer = writer
        self.prof_agg = prof_agg
        self._context = context or PipelineContext()
        self._preflight_types()

    @property
    def context(self) -> PipelineContext:
        return self._context

    def __call__(self, it: Iterable[Any]) -> Iterable[Any]:
        stream: Iterable[Any] = it
        for s in self.stages:
            stream = self._wrap_runtime_asserts(s, stream) if self._runtime_type_asserts else s(stream)
        return stream

    def then(self, nxt: Stage) -> "Pipeline":
        if self.writer is not None:
            raise RuntimeError("Cannot add stages after a writer. A writer is the last stage.")

        return Pipeline(self.stages + [nxt], prof_agg=self.prof_agg, context=self._context)

    def to(self, writer: Union[MaterializeWriterBase, StreamWriterBase]) -> "Pipeline":
        return Pipeline(stages=self.stages, writer=writer, prof_agg=self.prof_agg, context=self._context)

    def _preflight_types(self) -> None:
        errors: List[str] = []
        for i in range(len(self.stages) - 1):
            a, b = self.stages[i], self.stages[i + 1]
            if not _is_compatible(getattr(a, "output_type", object), getattr(b, "input_type", object)):
                errors.append(
                    f"Type mismatch: {type(a).__name__}.out={_tname(a.output_type)} "
                    f"-> {type(b).__name__}.in={_tname(b.input_type)}"
                )

        if self.writer is not None:
            if not _is_compatible(
                getattr(self.stages[-1], "output_type", object), getattr(self.writer, "input_type", object)
            ):
                errors.append(
                    f"Type mismatch: {type(self.stages[-1]).__name__}.out={_tname(self.stages[-1].output_type)} "
                    f"-> {type(self.writer).__name__}.in={_tname(self.writer.input_type)}"
                )
        if errors:
            raise TypeError("Pipeline type preflight failed:\n  - " + "\n  - ".join(errors))

    def _wrap_runtime_asserts(self, stage: Stage, it: Iterable[Any]) -> Iterable[Any]:
        out_types = _type_options(stage.output_type)

        def gen() -> Iterator[Any]:
            for item in stage(it):
                if out_types != (object,) and not isinstance(item, out_types):
                    exp = " | ".join(_tname(t) for t in out_types)
                    raise TypeError(f"{type(stage).__name__} yielded {_tname(type(item))}, expected {exp}")
                yield item

        return gen()

    def get_profile(self) -> dict:
        if self.prof_agg is None:
            return {"by_stage": {}, "by_slide": {}}
        return self.prof_agg.get_profile()

    def print_profile(self) -> None:
        if self.prof_agg is None:
            print("[profile] No profile data (did you run with profile=True?)")
            return
        self.prof_agg.print_profile()

    def _ensure_prof_agg(self) -> None:
        if self.prof_agg is None:
            self.prof_agg = PipelineProfileAggregator()

    def stream(
        self,
        cpu_processes: int = 4,
        writer_prefetch_factor: int = 2,
        profile: bool = False,
        verbosity_level: LogLevel = "WARNING",
        gracefully_handle_producer_errors: bool = False,
    ):
        """Streaming mode: returns an iterator from the StreamWriterBase."""
        raise RuntimeError()
        if not isinstance(self.writer, StreamWriterBase):
            raise RuntimeError("Pipeline writer is not a StreamWriterBase; use .materialize() instead.")

        sink_runner: Callable[[MPQueue], Iterable[Any]] = lambda q, v, e: self.writer.start_writer(q, v, stop_event=e)

        yield from self._orchestrate(
            cpu_processes=cpu_processes,
            writer_prefetch_factor=writer_prefetch_factor,
            profile=profile,
            verbosity=verbosity_level,
            gracefully_handle_producer_errors=gracefully_handle_producer_errors,
            sink_runner=sink_runner,
            streaming=True,
        )

    def materialize(
        self,
        cpu_processes: int = 4,
        writer_prefetch_factor: int = 2,
        profile: bool = False,
        verbosity_level: LogLevel = "WARNING",
        gracefully_handle_producer_errors: bool = False,
    ):
        """
        Materialize mode: runs a sink that writes to disk (e.g., WebDatasetWriter).
        Returns whatever the writer exposes as its "result" (optional).
        """
        if isinstance(self.writer, StreamWriterBase):
            raise RuntimeError("Pipeline has a StreamWriterBase; use .stream() instead of .materialize().")

        sink_runner: Callable[[MPQueue], Iterable[Any] | Any] = lambda q, v, e: self.writer.start_writer(
            q, v, stop_event=e
        )

        return self._orchestrate(
            cpu_processes=cpu_processes,
            writer_prefetch_factor=writer_prefetch_factor,
            profile=profile,
            verbosity=verbosity_level,
            gracefully_handle_producer_errors=gracefully_handle_producer_errors,
            sink_runner=sink_runner,
            streaming=False,
        )

    def _orchestrate(
        self,
        *,
        cpu_processes: int,
        writer_prefetch_factor: int,
        profile: bool,
        verbosity: LogLevel,
        gracefully_handle_producer_errors: bool,
        sink_runner: Callable[[MPQueue], Iterable[Any] | Any],
        streaming: bool,
    ):
        """
        Shared core:
        - prepares stages/contexts
        - sets up mp + bounded queue
        - spawns producers & supervisor
        - runs the sink (writer) either as iterator (streaming) or in a thread (materialize)
        """
        slides = self._prep_and_validate(verbosity=verbosity, profile=profile)
        if not slides:
            self.log.info("[WARN] No slides provided. Nothing to do.")
            return

        producer_queue, profiler_queue, stop_event = self._init_mp(
            queue_maxsize=writer_prefetch_factor * cpu_processes, profile=profile
        )

        sup_thread, stop_all, fail_box, failed_slides = self._spawn_supervisor(
            slides=slides,
            cpu_processes=cpu_processes,
            queue=producer_queue,
            profiler_queue=profiler_queue,
            profile=profile,
            verbosity=verbosity,
            gracefully_handle_producer_errors=gracefully_handle_producer_errors,
            stop_event=stop_event,
        )

        sink_result_box: List[Any] = []
        sink_exc_box: List[BaseException] = []

        if streaming:
            try:
                for item in sink_runner(producer_queue, verbosity, stop_event):  # <-- pass stop_event
                    if fail_box:
                        raise RuntimeError(fail_box[0])
                    yield item
                sup_thread.join(timeout=5.0)
                if fail_box:
                    raise RuntimeError(fail_box[0])
                if failed_slides:
                    self.log.warning(f"Streaming completed with errors on slides (skipped): {', '.join(failed_slides)}")
                else:
                    self.log.info("Streaming completed.")
            except GeneratorExit:
                self.log.info("Consumer stopped early; terminating producers.")
                stop_event.set()
                try:
                    producer_queue.put(EndOfQueue(), block=True, timeout=0.5)
                except Exception:
                    pass
                stop_all()
                raise
            except Exception:
                stop_event.set()
                try:
                    producer_queue.put(EndOfQueue(), block=True, timeout=0.5)
                except Exception:
                    pass
                stop_all()
                raise
            finally:
                sup_thread.join(timeout=5.0)
        else:

            def _run_sink():
                try:
                    out = sink_runner(producer_queue, verbosity, stop_event)
                    if isinstance(out, Iterable):
                        for _ in out:
                            if fail_box:
                                raise RuntimeError(fail_box[0])
                    else:
                        sink_result_box.append(out)
                except BaseException as e:
                    sink_exc_box.append(e)

            t = Thread(target=_run_sink, name="writer", daemon=False)
            t.start()

            try:
                t.join()
                sup_thread.join(timeout=5.0)
                if sink_exc_box:
                    raise sink_exc_box[0]
                if fail_box:
                    raise RuntimeError(fail_box[0])
                if failed_slides:
                    self.log.warning(
                        "Materialization completed with errors on slides (skipped): %s.", ", ".join(failed_slides)
                    )
                else:
                    self.log.info("Materialization completed.")
                if sink_result_box:
                    return sink_result_box[0]
                return getattr(self.writer, "result", None)
            except Exception:
                stop_event.set()
                try:
                    producer_queue.put(EndOfQueue(), block=True, timeout=0.5)
                except Exception:
                    pass
                stop_all()
                raise
            finally:
                sup_thread.join(timeout=5.0)

    def _prep_and_validate(self, *, verbosity: LogLevel, profile: bool) -> List[str]:
        """
        - init logging
        - ensure writer present
        - collect slides from first stage (grid)
        - export/attach/validate all stages + writer
        - reset profile aggregator if requested
        """
        init_logging(verbosity)
        if self.writer is None:
            raise RuntimeError("Pipeline has no writer; add one via .to(writer)")

        self.log.info("Preparing pipeline (profile=%s).", profile)

        grid = self.stages[0]
        slides = list(getattr(grid, "slides", []))
        # Do not cast to list() again later; preserve this order

        for s in self.stages + [self.writer]:
            s.export_context(self._context)
        for s in self.stages + [self.writer]:
            s.attach_context(self._context)
            s.validate()

        if profile:
            self._ensure_prof_agg()
            if self.prof_agg is not None:
                self.prof_agg.reset()

        return slides

    def _init_mp(self, *, queue_maxsize: int, profile: bool) -> Tuple[MPQueue, Optional[MPQueue], MpEvent]:
        if mp.get_start_method(allow_none=True) != "spawn":
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

        ctx = mp.get_context("spawn")
        producer_queue: MPQueue = ctx.Queue(maxsize=queue_maxsize)
        profiler_queue: Optional[MPQueue] = ctx.Queue() if profile else None
        stop_event: MpEvent = ctx.Event()
        return producer_queue, profiler_queue, stop_event

    def _spawn_supervisor(
        self,
        *,
        slides: List[str],
        cpu_processes: int,
        queue: MPQueue,
        profiler_queue: Optional[MPQueue],
        profile: bool,
        verbosity: LogLevel,
        gracefully_handle_producer_errors: bool,
        stop_event: MpEvent,
    ) -> Tuple[Thread, Callable[[], None], List[str], List[str]]:
        """
        Spawns a supervisor thread that:
        - launches up to `cpu_processes` producer processes
        - respawns producers until slides are exhausted
        - collects profiling messages (optional)
        - on exit or error, ALWAYS sends EndOfQueue to the writer
        Returns:
        - supervisor thread
        - stop_all() callable to abort producers + signal EndOfQueue
        - fail_box (list with reason string if fatal error)
        - failed_slides list (if skipping is enabled)
        """
        pending = list(slides)
        active: List[mp.Process] = []
        failed_slides: List[str] = []
        fail_box: List[str] = []  # single-element list carrying error reason
        kill_lock = threading.Lock()

        def spawn_for(path: str) -> mp.Process:
            p = mp.Process(
                target=_producer_worker,
                args=(
                    path,
                    self.stages,
                    queue,
                    profile,
                    profiler_queue,
                    gracefully_handle_producer_errors,
                    verbosity,
                    stop_event,
                ),
                name=f"producer-{Path(path).stem}",
            )
            p.start()
            return p

        for _ in range(min(cpu_processes, len(pending))):
            active.append(spawn_for(pending.pop(0)))

        def stop_all():
            with kill_lock:
                stop_event.set()
                for p in active:
                    try:
                        if p.is_alive():
                            p.terminate()
                    except Exception:
                        pass
                try:
                    queue.put(EndOfQueue(), block=True, timeout=0.5)
                except Exception:
                    pass

        def supervisor():
            try:
                while active:
                    for p in list(active):
                        p.join(timeout=0.1)
                        if not p.is_alive():
                            active.remove(p)
                            if p.exitcode and p.exitcode != 0:
                                slide_name = p.name.replace("producer-", "")
                                failed_slides.append(slide_name)
                                if not gracefully_handle_producer_errors:
                                    stop_event.set()
                                    fail_box.append(f"Producer failed on slide '{slide_name}'")
                                    for r in active:
                                        try:
                                            if r.is_alive():
                                                r.terminate()
                                        except Exception:
                                            pass
                                    return

                    while pending and len(active) < cpu_processes:
                        active.append(spawn_for(pending.pop(0)))
                        self.log.info("Spawning new producer... %d slides left.", len(pending))

                if profile and profiler_queue is not None and getattr(self, "prof_agg", None) is not None:
                    received, expected = 0, len(slides)
                    while received < expected:
                        try:
                            msg = profiler_queue.get(timeout=1.0)
                        except Exception:
                            break
                        if isinstance(msg, dict) and msg.get("_profile"):
                            self.prof_agg.ingest_msg(msg)
                            received += 1
            finally:
                # Always signal writer; avoid blocking forever
                try:
                    queue.put(EndOfQueue(), block=True, timeout=0.5)
                except Exception:
                    pass

        sup = Thread(target=supervisor, name="supervisor", daemon=True)
        sup.start()
        return sup, stop_all, fail_box, failed_slides


def _producer_worker(
    slide_path: str,
    stage_specs: List[Stage],
    producer_queue: MPQueue,
    profile: bool,
    profiler_queue: Optional[MPQueue],
    gracefully_handle_producer_errors: bool,
    verbosity_level: LogLevel = "WARNING",
    stop_event: Optional[MpEvent] = None,
):
    init_logging(verbosity_level)
    profiler: Optional[Profiler] = None
    try:
        slide_id = Path(slide_path).stem
        profiler = Profiler(enabled=profile, slide_id=slide_id)
        set_current_profiler(profiler)

        local_stages = [st.for_slide(slide_path) for st in stage_specs]
        pipe = Pipeline(local_stages)

        # sources ignore input
        for out in pipe(iter(())):
            if stop_event is not None and stop_event.is_set():
                break
            # bounded put with stop-aware retry
            while True:
                try:
                    producer_queue.put(out, block=True, timeout=0.5)
                    break
                except Full:
                    if stop_event is not None and stop_event.is_set():
                        break
                    continue

    except Exception as e:
        if not gracefully_handle_producer_errors:
            logging.exception(f"Producer error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if profile and profiler_queue is not None and profiler is not None:
            try:
                profiler_queue.put({"_profile": True, **profiler.serialize()})
            except Exception:
                logging.info("Failed to send profile message.", file=sys.stderr)
        set_current_profiler(None)

        # best-effort stream boundary
        try:
            producer_queue.put(EndOfStream(), block=True, timeout=0.5)
        except Exception:
            pass
