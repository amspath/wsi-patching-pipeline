import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass
from multiprocessing.queues import Queue as MPQueue
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from wsi_patching.logging_config import init_logging
from wsi_patching.profiling import PipelineProfileAggregator, Profiler, set_current_profiler
from wsi_patching.webdatasetwriter import WebDatasetWriter

Sample = Dict[str, Any]


class PipelineContext(dict):
    """Lightweight, picklable context for cross-stage config & checks."""

    def require_key(self, key: str):
        if key not in self:
            raise KeyError(f"Missing required context key: '{key}'")


class Stage:
    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        raise NotImplementedError

    def then(self, nxt: "Stage") -> "Pipeline":
        return Pipeline([self, nxt])

    def for_slide(self, slide_path: str) -> "Stage":
        return self

    # --- new hooks ---
    def attach_context(self, ctx: PipelineContext) -> None:
        self._ctx = ctx  # type: ignore[attr-defined]

    def export_context(self, ctx: PipelineContext) -> None:
        """Optional: seed/override context keys (called before preflight)."""
        pass

    def validate(self) -> None:
        """Optional: validate config using self._ctx (called once before run)."""
        pass

    @property
    def ctx(self) -> PipelineContext:
        return getattr(self, "_ctx", PipelineContext())


@dataclass
class Pipeline(Stage):
    stages: List[Stage]
    prof_agg: Optional["PipelineProfileAggregator"] = None  # new

    def __init__(
        self,
        stages: List[Stage],
        prof_agg: Optional["PipelineProfileAggregator"] = None,
        context: Optional[PipelineContext] = None,
    ):
        self.stages = stages
        self.prof_agg = prof_agg
        self._context = context or PipelineContext()

    @property
    def context(self) -> PipelineContext:
        return self._context

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for s in self.stages:
            it = s(it)
        return it

    def then(self, nxt: Stage) -> "Pipeline":
        return Pipeline(self.stages + [nxt], prof_agg=self.prof_agg)

    def get_profile(self) -> Dict[str, Any]:
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
        writer_stage: WebDatasetWriter = self.stages[-1]  # type: ignore
        producer_stages: List[Stage] = self.stages[:-1]

        grid = self.stages[0]  # type: ignore
        slides = list(grid.slides)
        if not slides:
            logging.info("[WARN] No slides provided. Nothing to do.")
            return

        # 1) Let stages contribute to context
        for s in self.stages[:-1]:
            s.export_context(self._context)

        # 2) Attach context then run preflight validations
        for s in self.stages[:-1]:
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
            self.prof_agg.reset()  # type: ignore[union-attr]

        writer_proc = mp.Process(target=writer_stage.start_writer, args=(q,), name="webdataset-writer")
        writer_proc.start()

        pending = list(slides)
        active: List[mp.Process] = []

        def spawn_for(path: str):
            p = mp.Process(
                target=_producer_worker,
                args=(path, producer_stages, q, profile, prof_q),
                name=f"producer-{Path(path).stem}",
            )
            p.start()
            return p

        for _ in range(min(cpu_processes, len(pending))):
            slide_path = pending.pop(0)
            active.append(spawn_for(slide_path))

        while active:
            for p in list(active):
                p.join(timeout=0.1)
                if not p.is_alive():
                    active.remove(p)
            while pending and len(active) < cpu_processes:
                slide_path = pending.pop(0)
                active.append(spawn_for(slide_path))

        if profile and prof_q is not None and self.prof_agg is not None:
            received = 0
            expected = len(slides)
            while received < expected:
                try:
                    msg = prof_q.get(timeout=1.0)
                except Exception:
                    break
                if isinstance(msg, dict) and msg.get("_profile"):
                    self.prof_agg.ingest_msg(msg)  # << delegate
                    received += 1

        q.put(None)
        writer_proc.join()


def _producer_worker(
    slide_path: str, stage_specs: List[Stage], queue: MPQueue, profile: bool, prof_queue: Optional[MPQueue]
):
    init_logging()
    logging.info("Starting processing.")
    try:
        slide_id = Path(slide_path).stem
        profiler = Profiler(enabled=profile, slide_id=slide_id)
        set_current_profiler(profiler)

        # Generic per-slide adaptation: each stage decides if/how it needs to specialize
        local_stages: List[Stage] = [st.for_slide(slide_path) for st in stage_specs]

        # Run per-slide pipeline up to (but excluding) the writer sink.
        pipe = Pipeline(local_stages)
        it = pipe(iter(()))  # sources ignore input

        for out in it:
            if out is None or out.get("_eos"):
                continue
            if "__key__" not in out or "png_bytes" not in out or "json_bytes" not in out:
                continue
            queue.put(out)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.info(f"Error: {e}", file=sys.stderr)
    finally:
        if profile and prof_queue is not None:
            try:
                prof_queue.put({"_profile": True, **profiler.serialize()})
            except Exception:
                logging.info("Failed to send profile message.", file=sys.stderr)
                pass
        set_current_profiler(None)
        queue.put({"_eos": True})
