import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from wsi_patching.queue_plumbing import (
    _make_emitter,
    _sample_bytes,
    execute_stages_locally,
    gpu_process_main,
    metrics_aggregator_main,
    writer_process_main,
)
from wsi_patching.typing import RuntimeCtx, Sample


class Stage:
    placement: str = "producer"  # "producer" | "gpu" | "writer"

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        raise NotImplementedError

    def then(self, nxt: "Stage") -> "Pipeline":
        return Pipeline([self, nxt])


@dataclass
class Pipeline(Stage):
    stages: List[Stage]

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for s in self.stages:
            it = s(it)
        return it

    def then(self, nxt: Stage) -> "Pipeline":
        return Pipeline(self.stages + [nxt])

    # Entry point: spawns processes and wires queues according to placements
    def run(
        self,
        max_producers: int = 4,
        gpu_devices: Optional[List[int]] = None,
        prod_queue_size: int = 5000,
        writer_queue_size: int = 5000,
    ):
        gpu_devices = gpu_devices or [0]

        # Partition by placement
        prod_stages: List[Stage] = []
        gpu_stages: List[Stage] = []
        writer_stages: List[Stage] = []

        seen_gpu = False
        seen_writer = False
        for s in self.stages:
            if s.placement == "producer" and not seen_gpu and not seen_writer:
                prod_stages.append(s)
            elif s.placement == "gpu" and not seen_writer:
                seen_gpu = True
                gpu_stages.append(s)
            else:
                seen_writer = True
                writer_stages.append(s)

        if not writer_stages or writer_stages[-1].placement != "writer":
            raise RuntimeError("Pipeline must end with a writer-stage (e.g., ToWebDataset).")

        # Build shared queues
        ctx = RuntimeCtx()
        ctx.prod_to_gpu = mp.Queue(maxsize=prod_queue_size)
        ctx.gpu_to_writer = mp.Queue(maxsize=writer_queue_size)
        ctx.metrics_q = mp.Queue(maxsize=10000)

        # Discover slides from the first stage (WSIGrid) in prod_stages
        slides = discover_slides_from_pipeline(prod_stages)

        # Start aggregator (expect EOS from: all producer procs + all GPU procs + writer)
        expected_eos = len(slides) + len(gpu_devices) + 1
        metrics_p = mp.Process(
            target=metrics_aggregator_main, args=(ctx.metrics_q, expected_eos), name="metrics", daemon=True
        )
        metrics_p.start()

        # Writer process (single)
        writer_p = mp.Process(target=writer_process_main, args=(writer_stages, ctx), name="writer", daemon=True)
        writer_p.start()

        # GPU processes (one per device)
        gpu_ps: List[mp.Process] = []
        for dev in gpu_devices:
            gpu_p = mp.Process(
                target=gpu_process_main, args=(gpu_stages, ctx, len(slides), dev), name=f"gpu:{dev}", daemon=True
            )
            gpu_p.start()
            gpu_ps.append(gpu_p)

        # Producer processes (one per WSI, up to max_producers concurrently)
        # We will launch up to max_producers at a time to avoid oversubscribing.
        # Each producer runs prod_stages for its single WSI and emits to prod_to_gpu.
        active: List[mp.Process] = []
        launched = 0
        for slide in slides:
            while len(active) >= max_producers:
                # Wait for any producer to finish
                for p in list(active):
                    if not p.is_alive():
                        p.join()
                        active.remove(p)
                if len(active) >= max_producers:
                    time.sleep(0.05)

            p = mp.Process(
                target=producer_process_main,
                args=(prod_stages, ctx, slide),
                name=f"producer:{Path(slide).stem}",
                daemon=True,
            )
            p.start()
            active.append(p)
            launched += 1

        # Wait for all producers to finish
        for p in active:
            p.join()

        # Signal GPU processes that all producers finished (one EOS per producer)
        for _ in range(len(slides)):
            ctx.prod_to_gpu.put({"_eos": True})

        # Wait for GPU to finish and pass EOS downstream
        for p in gpu_ps:
            p.join()

        # Wait for writer to close shards & exit
        writer_p.join()
        metrics_p.join()


class WSIGrid(Stage):
    """
    Source. Yields one Sample per WSI, carrying config to downstream stages.
    We *don't* enumerate tiles here; RegionReadAndBatch will slice per region.
    """

    placement = "producer"

    def __init__(
        self, slides: List[str], tile_size: int, stride: int, level: int = 0, align_origin: Tuple[int, int] = (0, 0)
    ):
        self.slides = list(slides)
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.level = int(level)
        self.align_origin = tuple(align_origin)

    def __call__(self, _it: Iterable[Sample]) -> Iterable[Sample]:
        for path in self.slides:
            yield {
                "wsi_id": Path(path).stem,
                "meta": {"path": path, "backend": "cucim"},
                "level": self.level,
                "tile_size": self.tile_size,
                "stride": self.stride,
                "align_origin": self.align_origin,
                # FilterByROI may add "roi_rects": List[Rect]
            }


# ------------------------------
# Runtime: process/queue plumbing
# ------------------------------
def discover_slides_from_pipeline(prod_stages: List[Stage]) -> List[str]:
    """
    Execute just the WSIGrid stage to list slides. We detect it by type.
    """
    grid = next((s for s in prod_stages if isinstance(s, WSIGrid)), None)
    if grid is None:
        raise RuntimeError("WSIGrid stage is required in the producer pipeline.")
    # We don't want to run everything; just extract slide paths
    return list(grid.slides)


def producer_process_main(prod_stages: List[Stage], ctx: RuntimeCtx, slide_path: str):
    emit = _make_emitter(ctx, "producer")
    try:
        first = prod_stages[0]
        if not isinstance(first, WSIGrid):
            raise RuntimeError("First producer stage must be WSIGrid.")
        grid = WSIGrid(
            slides=[slide_path],
            tile_size=first.tile_size,
            stride=first.stride,
            level=first.level,
            align_origin=first.align_origin,
        )
        local_stages = [grid] + prod_stages[1:]

        it: Iterable[Sample] = iter(())
        it = execute_stages_locally(local_stages, it=it, emit_metric=emit)

        q = ctx.prod_to_gpu
        assert q is not None
        items = 0
        bytes_tot = 0
        t_queue = 0.0
        for s in it:
            items += 1
            bytes_tot += _sample_bytes(s)
            t0 = time.perf_counter()
            q.put(s)  # may block (backpressure)
            t_queue += time.perf_counter() - t0

        # report queue time
        emit({"type": "queue_put", "queue": "prod→gpu", "items": items, "bytes": bytes_tot, "time_s": t_queue})

    except Exception:
        traceback.print_exc()
    finally:
        # EOS is sent by parent to GPU procs; but the metrics aggregator expects an EOS per producer
        emit({"type": "eos"})
