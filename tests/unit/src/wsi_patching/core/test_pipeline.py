from typing import Any, Iterable, List, Union

import pytest

from wsi_patching.core.pipeline import Pipeline, Stage, _producer_worker
from wsi_patching.utils.types import EndOfStream


# ----------------- helpers & fakes -----------------
class SourceStage(Stage):
    """Source that ignores input, yields a fixed sequence. Pretend it has slides (used by Pipeline.run)."""

    def __init__(self, values: List[int], slides: Union[List[str], None] = None):
        self.values = values
        self.slides = slides or []

    def __call__(self, it: Iterable[Any]) -> Iterable[int]:
        return iter(self.values)


class MapToStr(Stage):
    """Maps ints to strings. Used to trigger type mismatch / runtime asserts."""

    input_type = int
    output_type = str

    def __call__(self, it: Iterable[int]) -> Iterable[str]:
        for x in it:
            yield str(x)


class PassThrough(Stage):
    """Pass-through stage with explicit types for compatibility tests."""

    def __init__(self, input_t=object, output_t=object):
        self.input_type = input_t
        self.output_type = output_t

    def __call__(self, it: Iterable[Any]) -> Iterable[Any]:
        for x in it:
            yield x


class DummyWriter:
    """Minimal writer-like object (only used for type preflight on .to())."""

    def __init__(self, input_type=object):
        self.input_type = input_type


class FakeQueue:
    def __init__(self):
        self.items: List[Any] = []

    def put(self, x: Any):
        self.items.append(x)

    def get(self, timeout: Union[float, None] = None):
        if not self.items:
            raise RuntimeError("Queue empty")
        return self.items.pop(0)


# ----------------- tests (unchanged below) -----------------
def test_pipeline_type_preflight_ok():
    p = Pipeline([SourceStage([1, 2]), MapToStr()])
    out = list(p(iter(())))
    assert out == ["1", "2"]


def test_pipeline_type_preflight_mismatch_raises():
    s1 = PassThrough(input_t=object, output_t=int)
    s2 = PassThrough(input_t=int, output_t=int)
    s3 = PassThrough(input_t=float, output_t=float)
    with pytest.raises(TypeError) as ei:
        _ = Pipeline([s1, s2, s3])
    msg = str(ei.value)
    assert "Pipeline type preflight failed" in msg
    assert "Type mismatch" in msg


def test_pipeline_then_disallows_after_writer():
    p = Pipeline([SourceStage([1])]).to(DummyWriter(input_type=int))
    with pytest.raises(RuntimeError):
        p.then(PassThrough())


def test_pipeline_to_keeps_stages_and_sets_writer():
    ok_writer = DummyWriter(input_type=str)
    p2 = Pipeline([SourceStage([1, 2]), MapToStr()]).to(ok_writer)
    out = list(p2(iter(())))
    assert out == ["1", "2"]


def test_runtime_type_assertion_catches_wrong_yield_type():
    class BadStage(Stage):
        output_type = int

        # IMPORTANT: annotate return so StageMeta sets output_type=int
        def __call__(self, it: Iterable[Any]) -> Iterable[int]:
            yield "oops"  # wrong type

    p = Pipeline([BadStage()])
    with pytest.raises(TypeError) as ei:
        _ = list(p(iter(())))
    assert "yielded str, expected int" in str(ei.value)


def test_get_and_print_profile_without_aggregator(capsys):
    p = Pipeline([SourceStage([1])])
    assert p.get_profile() == {"by_stage": {}, "by_slide": {}}
    p.print_profile()
    out = capsys.readouterr().out
    assert "[profile] No profile data (did you run with profile=True?)" in out


def test_producer_worker_puts_items_eos_and_profile(monkeypatch):
    stages = [SourceStage([10, 20])]
    q = FakeQueue()
    prof_q = FakeQueue()

    _producer_worker(
        slide_path="slide_a.svs",
        stage_specs=stages,
        queue=q,
        profile=True,
        prof_queue=prof_q,
        gracefully_handle_producer_errors=False,
        verbosity_level="INFO",
    )

    assert q.items[-1].__class__ is EndOfStream or isinstance(q.items[-1], EndOfStream)
    payload = q.items[:-1]
    assert payload == [10, 20]

    assert len(prof_q.items) == 1
    msg = prof_q.items[0]
    assert isinstance(msg, dict)
    assert msg.get("_profile") is True
    assert msg.get("slide_id") == "slide_a"
