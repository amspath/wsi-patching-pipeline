from typing import Any, Dict, Optional, Union


class Profiler:
    """Per-process profiler, but only used for isolated timings (manual calls)."""

    def __init__(self, enabled: bool, slide_id: str):
        self.enabled = bool(enabled)
        self.slide_id = slide_id
        self._stats: Dict[str, Dict[str, Union[float, int]]] = {}

    def _ensure(self, stage_name: str):
        if stage_name not in self._stats:
            self._stats[stage_name] = {"wall_time_sec": 0.0, "yields": 0}

    def add_time(self, stage_name: str, dt: float, yielded: bool):
        """Record time spent. If `yielded` is True, increment yield count.

        Yield represents whether the computation eventually produced an output item for the stage generator.
        """
        if not self.enabled:
            return
        self._ensure(stage_name)
        self._stats[stage_name]["wall_time_sec"] = float(self._stats[stage_name]["wall_time_sec"]) + float(dt)
        if yielded:
            self._stats[stage_name]["yields"] = int(self._stats[stage_name]["yields"]) + 1

    def serialize(self) -> Dict[str, Any]:
        return {"slide_id": self.slide_id, "stages": self._stats}


class PipelineProfileAggregator:
    """
    Collects per-process _Profiler summaries and exposes an aggregated view
    with the same shape you had before:
      - get_profile(): {"by_stage": {...}, "by_slide": {...}}
      - print_profile(): pretty console output
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._agg: Dict[str, Any] = {"by_stage": {}, "by_slide": {}}

    def ingest_msg(self, msg: Dict[str, Any]) -> None:
        """Ingest a single producer summary message."""
        slide_id = msg.get("slide_id", "<unknown>")
        stages: Dict[str, Dict[str, Union[float, int]]] = msg.get("stages", {})

        self._agg["by_slide"][slide_id] = {}
        for stage_name, stats in stages.items():
            wall = float(stats.get("wall_time_sec", 0.0))
            n = int(stats.get("yields", 0))
            avg = (wall / n * 1000.0) if n > 0 else 0.0

            # Per-slide
            self._agg["by_slide"][slide_id][stage_name] = {"wall_time_sec": wall, "yields": n, "avg_ms_per_yield": avg}

            # Aggregate by stage
            agg = self._agg["by_stage"].setdefault(stage_name, {"wall_time_sec": 0.0, "yields": 0})
            agg["wall_time_sec"] = float(agg["wall_time_sec"]) + wall
            agg["yields"] = int(agg["yields"]) + n

    def get_profile(self) -> Dict[str, Any]:
        out = {"by_stage": {}, "by_slide": self._agg.get("by_slide", {})}
        for stage_name, agg in self._agg.get("by_stage", {}).items():
            wall = float(agg["wall_time_sec"])
            n = int(agg["yields"])
            out["by_stage"][stage_name] = {
                "wall_time_sec": wall,
                "yields": n,
                "avg_ms_per_yield": (wall / n * 1000.0) if n > 0 else 0.0,
            }
        return out

    def print_profile(self) -> None:
        prof = self.get_profile()
        if not prof["by_stage"]:
            return

        def fmt(stats: Dict[str, Union[float, int]]) -> str:
            return (
                f"{int(stats['yields']):10d} {float(stats['wall_time_sec']):12.3f} "
                f"{float(stats['avg_ms_per_yield']):16.3f}"
            )

        print("\n=== Pipeline Profile (isolated timings only) ===")
        print(f"{'Stage':30} {'Yields':>10} {'Wall (s)':>12} {'Avg (ms/yield)':>16}")
        for name, stats in sorted(prof["by_stage"].items(), key=lambda kv: kv[1]["wall_time_sec"], reverse=True):
            print(f"{name:30} {fmt(stats)}")

        print("\n--- Per slide breakdown ---")
        for slide_id, stages in prof["by_slide"].items():
            print(f"[{slide_id}]")
            for name, stats in sorted(stages.items(), key=lambda kv: kv[1]["wall_time_sec"], reverse=True):
                print(
                    f"  {name:28} yields={int(stats['yields']):6d} "
                    f"wall={float(stats['wall_time_sec']):8.3f}s avg={float(stats['avg_ms_per_yield']):8.3f}ms"
                )


# Global per-process profiler handle
CURRENT_PROFILER: Optional["Profiler"] = None


def set_current_profiler(p: Optional["Profiler"]) -> None:
    global CURRENT_PROFILER
    CURRENT_PROFILER = p


def get_current_profiler() -> Optional["Profiler"]:
    return CURRENT_PROFILER
