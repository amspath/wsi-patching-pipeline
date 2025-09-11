import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

Sample = Dict[str, Any]
Rect = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class RuntimeCtx:
    prod_to_gpu: Optional[mp.Queue] = None
    gpu_to_writer: Optional[mp.Queue] = None
    metrics_q: Optional[mp.Queue] = None
