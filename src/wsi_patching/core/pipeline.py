from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass, field
from multiprocessing.queues import Queue as MPQueue
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Union, get_args, get_origin

from wsi_patching.utils.logging_config import init_logging
from wsi_patching.utils.profiling import PipelineProfileAggregator, Profiler, set_current_profiler
from wsi_patching.utils.stage_meta import StageMeta
from wsi_patching.utils.types import EndOfQueue, EndOfStream, Patch
from wsi_patching.writers.writer import WriterBase


# -------- Context --------
class PipelineContext(dict):
    def require_key(self, key: str):
        if key not in self:
            raise KeyError(f"Missing required context key: '{key}'")


def _type_options(t: Any) -> tuple[type, ...]:
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


class Stage(metaclass=StageMeta):
    input_type: Any = object
    output_type: Any = object

    def __call__(self, it: Iterable[Any]) -> Iterable[Any]:
        raise NotImplementedError

    def then(self, nxt: "Stage") -> "Pipeline":
        return Pipeline([self, nxt])

    def for_slide(self, slide_path: str) -> "Stage":
        return self

    def attach_context(self, ctx: PipelineContext) -> None:
        self._ctx = ctx  # type: ignore[attr-defined]

    def export_context(self, ctx: PipelineContext) -> None:
        pass

    def validate(self) -> None:
        pass

    @property
    def ctx(self) -> PipelineContext:
        return getattr(self, "_ctx", PipelineContext())


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
        writer: Optional[WriterBase] = None,
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

    def to(self, writer: WriterBase) -> "Pipeline":
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

    def run(self, cpu_processes: int = 4, queue_maxsize: int = 4000, profile: bool = False):
        init_logging()
        logging.info(f"Starting pipeline with {cpu_processes} processes (profile={profile}).")

        if self.writer is None:
            raise RuntimeError("Pipeline has no writer; add one via .to(writer)")

        grid = self.stages[0]
        slides = list(getattr(grid, "slides", []))
        if not slides:
            logging.info("[WARN] No slides provided. Nothing to do.")
            return

        for s in self.stages:
            s.export_context(self._context)
        for s in self.stages:
            s.attach_context(self._context)
            s.validate()

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

        writer_proc = mp.Process(target=self.writer.start_writer, args=(q,), name="writer")
        writer_proc.start()

        pending = list(slides)
        active: List[mp.Process] = []

        def spawn_for(path: str):
            p = mp.Process(
                target=_producer_worker,
                args=(path, self.stages, q, profile, prof_q),
                name=f"producer-{Path(path).stem}",
            )
            p.start()
            return p

        for _ in range(min(cpu_processes, len(pending))):
            active.append(spawn_for(pending.pop(0)))

        while active:
            for p in list(active):
                p.join(timeout=0.1)
                if not p.is_alive():
                    active.remove(p)
            while pending and len(active) < cpu_processes:
                active.append(spawn_for(pending.pop(0)))

        # Collect profiling
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

        q.put(EndOfQueue())
        writer_proc.join()


def _producer_worker(
    slide_path: str, stage_specs: List[Stage], queue: MPQueue, profile: bool, prof_queue: Optional[MPQueue]
):
    init_logging()
    logging.info("Starting processing.")
    profiler: Optional[Profiler] = None
    try:
        slide_id = Path(slide_path).stem
        profiler = Profiler(enabled=profile, slide_id=slide_id)
        set_current_profiler(profiler)

        local_stages = [st.for_slide(slide_path) for st in stage_specs]
        pipe = Pipeline(local_stages)

        # sources ignore input
        for out in pipe(iter(())):
            if isinstance(out, Patch):
                queue.put(out)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.exception(f"Producer error: {e}", exc_info=True)
    finally:
        if profile and prof_queue is not None and profiler is not None:
            try:
                prof_queue.put({"_profile": True, **profiler.serialize()})
            except Exception:
                logging.info("Failed to send profile message.", file=sys.stderr)
        set_current_profiler(None)
        queue.put(EndOfStream())
