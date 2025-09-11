from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING, Iterable, List

import numpy as np

from wsi_patching.typing import RuntimeCtx, Sample

if TYPE_CHECKING:
    from wsi_patching.core import Stage


def execute_stages_locally(stages: List["Stage"], it: Iterable["Sample"], emit_metric) -> Iterable["Sample"]:
    """
    Run stages synchronously in this process, timing each stage.
    """
    for s in stages:
        stage_name = s.__class__.__name__
        placement = getattr(s, "placement", "producer")

        t0 = time.perf_counter()
        out = s(it)  # call the stage
        call_time = time.perf_counter() - t0

        def stage_iter(out_iter):
            items_out = 0
            bytes_out = 0
            t_iter = 0.0
            for item in out_iter:
                t1 = time.perf_counter()
                yield item
                t_iter += time.perf_counter() - t1
                items_out += 1
                bytes_out += _sample_bytes(item)
            emit_metric(
                {
                    "type": "stage",
                    "stage": stage_name,
                    "placement": placement,
                    "items_out": items_out,
                    "bytes_out": bytes_out,
                    "time_s": call_time + t_iter,
                }
            )

        it = stage_iter(out)
    return it


def gpu_process_main(gpu_stages: List[Stage], ctx: RuntimeCtx, num_producers: int, device_id: int):
    emit = _make_emitter(ctx, "gpu")

    inQ = ctx.prod_to_gpu
    outQ = ctx.gpu_to_writer
    assert inQ is not None and outQ is not None

    gpu_ops = next((s for s in gpu_stages if hasattr(s, "function")), None)  # TODO: better detection
    batch_size = gpu_ops.batch_size if gpu_ops else 200
    timeout_ms = gpu_ops.batch_timeout_ms if gpu_ops else 75

    def run_gpu_pipeline(batch: List[Sample]) -> List[Sample]:
        # Run the GPU stages with per-stage timing
        out = [{"batch": batch}]
        for s in gpu_stages:
            t0 = time.perf_counter()
            # stage may yield multiple items; collect them
            tmp = []
            for item in s(out):
                tmp.append(item)
            dt = time.perf_counter() - t0
            emit(
                {
                    "type": "stage",
                    "stage": s.__class__.__name__,
                    "items_out": len(tmp),
                    "bytes_out": sum(_sample_bytes(x) for x in tmp),
                    "time_s": dt,
                }
            )
            out = tmp
        # PNGEncoder produces individual samples
        return out

    eos_seen = 0
    buffer: List[Sample] = []
    last_flush = time.time()
    items_put = 0
    bytes_put = 0
    t_put = 0.0

    def flush_if_ready(force: bool = False):
        nonlocal buffer, last_flush, items_put, bytes_put, t_put
        now = time.time()
        if not buffer:
            last_flush = now
            return
        age_ms = (now - last_flush) * 1000.0
        if force or (len(buffer) >= batch_size) or (age_ms >= timeout_ms):
            encoded_samples = run_gpu_pipeline(buffer)
            for s in encoded_samples:
                t0 = time.perf_counter()
                outQ.put(s)
                t_put += time.perf_counter() - t0
                items_put += 1
                bytes_put += _sample_bytes(s)
            buffer = []
            last_flush = time.time()

    try:
        while True:
            try:
                item = inQ.get(timeout=0.1)
            except queue.Empty:
                item = None

            if item is None:
                flush_if_ready(False)
                continue

            if isinstance(item, dict) and item.get("_eos"):
                eos_seen += 1
                flush_if_ready(True)
                if eos_seen >= num_producers:
                    outQ.put({"_eos": True})
                    break
                continue

            buffer.append(item)
            flush_if_ready(False)
    except Exception:
        traceback.print_exc()
        try:
            outQ.put({"_eos": True})
        except Exception:
            pass
    finally:
        emit({"type": "queue_put", "queue": "gpu→writer", "items": items_put, "bytes": bytes_put, "time_s": t_put})
        emit({"type": "eos"})


def writer_process_main(writer_stages: List[Stage], ctx: RuntimeCtx):
    emit = _make_emitter(ctx, "writer")
    inQ = ctx.gpu_to_writer
    assert inQ is not None

    class QIter:
        def __iter__(self):
            return self

        def __next__(self):
            item = inQ.get()
            if isinstance(item, dict) and item.get("_eos"):
                raise StopIteration
            return item

    it: Iterable[Sample] = QIter()
    try:
        for _ in execute_stages_locally(writer_stages, it, emit_metric=emit):
            pass
    except StopIteration:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        emit({"type": "eos"})


def metrics_aggregator_main(q: mp.Queue, expected_eos: int):  # noqa: F821
    # aggregate by (placement, stage)
    agg = defaultdict(lambda: {"time_s": 0.0, "items_out": 0, "bytes_out": 0})
    eos = 0
    while True:
        m = q.get()
        if m.get("type") == "eos":
            eos += 1
            if eos >= expected_eos:
                break
            continue
        if m.get("type") == "stage":
            key = (m["placement"], m["stage"])
            a = agg[key]
            a["time_s"] += m.get("time_s", 0.0)
            a["items_out"] += m.get("items_out", 0)
            a["bytes_out"] += m.get("bytes_out", 0)
        elif m.get("type") == "queue_put":
            key = (m["placement"], f"QueuePut@{m.get('queue')}")
            a = agg[key]
            a["time_s"] += m.get("time_s", 0.0)
            a["items_out"] += m.get("items", 0)
            a["bytes_out"] += m.get("bytes", 0)

    # Print summary
    rows = []
    for (placement, stage), v in agg.items():
        t = v["time_s"]
        n = v["items_out"]
        b = v["bytes_out"]
        ips = (n / t) if t > 0 else 0.0
        mbps = (b / (1024 * 1024)) / t if t > 0 else 0.0
        rows.append((t, placement, stage, n, b, ips, mbps))
    rows.sort(reverse=True)  # by time

    print("\n=== Pipeline profile (aggregated) ===")
    print(f"{'time_s':>9}  {'where':<9}  {'stage':<28}  {'items':>10}  {'MB_out':>10}  {'items/s':>10}  {'MB/s':>10}")
    for t, pl, st, n, b, ips, mbps in rows:
        print(f"{t:9.2f}  {pl:<9}  {st:<28}  {n:10d}  {b / (1024 * 1024):10.2f}  {ips:10.1f}  {mbps:10.2f}")
    print("====================================\n")


def _make_emitter(ctx: RuntimeCtx, placement: str):
    q = ctx.metrics_q

    def emit(m: dict):
        if q is not None:
            m["placement"] = placement
            q.put(m, block=True)

    return emit


def _sample_bytes(s: Sample) -> int:
    if "png" in s and isinstance(s["png"], (bytes, bytearray)):
        return len(s["png"])
    p = s.get("patch")
    if p is None:
        return 0
    if hasattr(p, "nbytes"):
        return int(p.nbytes)
    try:
        return int(np.asarray(p).nbytes)
    except Exception:
        return 0


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b
