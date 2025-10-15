import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass, field
from multiprocessing.queues import Queue as MPQueue
from pathlib import Path
from threading import Thread
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

from wsi_patching.core.types.util_types import EndOfQueue, EndOfStream
from wsi_patching.utils.logging_config import LogLevel, init_logging
from wsi_patching.utils.meta_typing import ContextAware, PipelineContext, StageMeta
from wsi_patching.utils.profiling import PipelineProfileAggregator, Profiler, get_current_profiler, set_current_profiler
from wsi_patching.writers.generator_writer_base import GeneratorWriterBase
from wsi_patching.writers.writer_base import WriterBase

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
    writer: Optional[WriterBase] = None
    prof_agg: Optional["PipelineProfileAggregator"] = None
    _context: PipelineContext = field(default_factory=PipelineContext)
    _runtime_type_asserts: bool = True

    def __init__(
        self,
        stages: List[Stage],
        writer: Optional[Union[WriterBase, GeneratorWriterBase]] = None,
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

    def to(self, writer: Union[WriterBase, GeneratorWriterBase]) -> "Pipeline":
        if isinstance(writer, WriterBase):
            return WriterPipeline(stages=self.stages, writer=writer, prof_agg=self.prof_agg, context=self._context)
        elif isinstance(writer, GeneratorWriterBase):
            return GeneratorPipeline(stages=self.stages, writer=writer, prof_agg=self.prof_agg, context=self._context)
        else:
            raise TypeError("writer must be a WriterBase or GeneratorWriterBase instance")

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

    def materialize(self, **kwargs) -> Any:
        if self.writer is None:
            raise RuntimeError("Pipeline has no writer; add one via .to(writer)")

        if isinstance(self.writer, GeneratorWriterBase):
            raise RuntimeError("materialize() can only be used with WriterBase writers. Use stream instead.")

        raise NotImplementedError("materialize() not yet implemented for this WriterPipeline class.")

    def stream(self, **kwargs) -> Iterable[Any]:
        if self.writer is None:
            raise RuntimeError("Pipeline has no writer; add one via .to(writer)")

        if isinstance(self.writer, WriterBase):
            raise RuntimeError("stream() can only be used with GeneratorWriterBase writers. Use materialize() instead.")

        raise RuntimeError("stream() not yet implemented for this GeneratorPipeline class.")


class WriterPipeline(Pipeline):
    writer: WriterBase

    def materialize(
        self,
        cpu_processes: int = 4,
        queue_maxsize: int = 4000,
        profile: bool = False,
        verbosity_level: LogLevel = "WARNING",
        gracefully_handle_producer_errors: bool = False,
    ):
        init_logging(verbosity_level)
        self.log.info(f"Starting pipeline with {cpu_processes} processes (profile={profile}).")

        if self.writer is None:
            raise RuntimeError("Pipeline has no writer; add one via .to(writer)")

        grid = self.stages[0]
        slides = list(getattr(grid, "slides", []))
        if not slides:
            self.log.info("[WARN] No slides provided. Nothing to do.")
            return

        # export/attach/validate
        for s in self.stages + [self.writer]:
            s.export_context(self._context)
        for s in self.stages + [self.writer]:
            s.attach_context(self._context)
            s.validate()

        # processes for producers; thread for writer
        if mp.get_start_method(allow_none=True) != "spawn":
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

        q: MPQueue = mp.Queue(maxsize=queue_maxsize)
        prof_q: Optional[MPQueue] = mp.Queue() if profile else None

        if profile:
            self._ensure_prof_agg()
            self.prof_agg.reset()

        # --- writer as THREAD ---
        # Uses thread to avoid CUDA context issues on in memory output
        writer_thread = Thread(
            target=self.writer.start_writer,
            args=(q, verbosity_level),
            name="writer",
            daemon=False,  # ensure clean join
        )
        writer_thread.start()

        # --- spawn producer PROCESSES ---
        pending = list(slides)
        active: List[mp.Process] = []

        def spawn_for(path: str):
            p = mp.Process(
                target=_producer_worker,
                args=(path, self.stages, q, profile, prof_q, gracefully_handle_producer_errors, verbosity_level),
                name=f"producer-{Path(path).stem}",
            )
            p.start()
            return p

        for _ in range(min(cpu_processes, len(pending))):
            active.append(spawn_for(pending.pop(0)))

        failed_slides = []

        def abort_all(reason: str):
            for p in active:
                if p.is_alive():
                    p.terminate()
            # Best-effort: tell writer we're done if it's still alive
            try:
                q.put(EndOfQueue())
            except Exception:
                pass
            writer_thread.join(timeout=5.0)
            self.log.error(reason)
            raise RuntimeError(reason)

        while active:
            # 1) Writer crash → abort immediately
            if not writer_thread.is_alive():
                abort_all("Writer thread crashed; pipeline aborted.")

            # 2) Reap producers; handle failures
            for p in list(active):
                p.join(timeout=0.1)
                if not p.is_alive():
                    active.remove(p)
                    # Non-zero exit means producer failed
                    if p.exitcode and p.exitcode != 0:
                        slide_name = p.name.replace("producer-", "")
                        failed_slides.append(slide_name)
                        if not gracefully_handle_producer_errors:
                            abort_all(f"Producer failed on slide '{slide_name}'")

            # 3) Keep queue full
            while pending and len(active) < cpu_processes:
                active.append(spawn_for(pending.pop(0)))
                self.log.info(f"Spawning new producer... {len(pending)} slides left.")

        # collect profiling
        if profile and prof_q is not None and self.prof_agg is not None:
            received, expected = 0, len(slides)
            while received < expected:
                try:
                    msg = prof_q.get(timeout=1.0)
                except Exception:
                    break
                if isinstance(msg, dict) and msg.get("_profile"):
                    self.prof_agg.ingest_msg(msg)
                    received += 1

        if not writer_thread.is_alive():
            raise RuntimeError("Writer crashed during shutdown.")

        # signal end of stream and join writer thread
        q.put(EndOfQueue())
        writer_thread.join()

        if failed_slides and gracefully_handle_producer_errors:
            self.log.warning(f"Pipeline completed with errors on slides (skipped): {', '.join(failed_slides)}.")

        # now self.writer is the SAME object that consumed data
        return self.writer.get_output()


class GeneratorPipeline(Pipeline):
    writer: GeneratorWriterBase

    def stream(
        self,
        cpu_processes: int = 4,
        queue_maxsize: int = 4000,
        profile: bool = False,
        verbosity_level: LogLevel = "WARNING",
        gracefully_handle_producer_errors: bool = False,
    ) -> Iterable[Any]: ...


def _producer_worker(
    slide_path: str,
    stage_specs: List[Stage],
    queue: MPQueue,
    profile: bool,
    prof_queue: Optional[MPQueue],
    gracefully_handle_producer_errors: bool,
    verbosity_level: LogLevel = "WARNING",
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
            queue.put(out)
    except Exception as e:
        if not gracefully_handle_producer_errors:
            logging.exception(f"Producer error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if profile and prof_queue is not None and profiler is not None:
            try:
                prof_queue.put({"_profile": True, **profiler.serialize()})
            except Exception:
                logging.info("Failed to send profile message.", file=sys.stderr)
        set_current_profiler(None)

        try:
            queue.put(EndOfStream())
        except Exception:
            pass
