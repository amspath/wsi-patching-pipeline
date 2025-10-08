from typing import Iterable, Optional, Union

import pytest

from wsi_patching.utils.meta_typing import ContextAware, PipelineContext, StageMeta, WriterMeta


# -------- StageMeta --------
def test_stage_meta_infers_iterable_payload_and_return_scalar():
    class S(metaclass=StageMeta):
        def __call__(self, it: Iterable[int]) -> str:
            return "x"

    assert S.input_type is int
    assert S.output_type is str


def test_stage_meta_infers_union_payload_and_union_return():
    class S(metaclass=StageMeta):
        def __call__(self, it: Iterable[Union[int, str]]) -> Union[bytes, str]:
            return b"x"

    # input becomes a tuple of the union args
    assert S.input_type == (int, str)
    # output becomes a tuple of the union args
    assert S.output_type == (bytes, str)


def test_stage_meta_ignores_none_in_union_optional():
    class S(metaclass=StageMeta):
        def __call__(self, it: Iterable[Optional[Union[int, str]]]) -> Optional[int]:
            return 1

    assert S.input_type == (int, str)
    assert S.output_type == (int,)


def test_stage_meta_defaults_to_object_when_no_annotations():
    with pytest.raises(ValueError):

        class S(metaclass=StageMeta):
            def __call__(self, it):
                return None


# -------- WriterMeta --------
def test_writer_meta_infers_iterable_payload():
    class W(metaclass=WriterMeta):
        def write(self, batch: Iterable[float]) -> None:
            pass

    assert W.input_type is float


def test_writer_meta_keeps_existing_input_type_when_no_write_annotations():
    with pytest.raises(ValueError):

        class W(metaclass=WriterMeta):
            def write(self, batch):
                pass


# -------- PipelineContext & ContextAware --------
def test_pipeline_context_require_key_present():
    ctx = PipelineContext({"a": 1})
    # should not raise
    ctx.require_key("a")


def test_pipeline_context_require_key_missing_raises():
    ctx = PipelineContext()
    with pytest.raises(KeyError):
        ctx.require_key("missing")


def test_context_aware_ctx_property_and_attach():
    ctx = PipelineContext({"k": "v"})

    class C(ContextAware):
        pass

    c = C()
    # default ctx is a PipelineContext (empty)
    assert isinstance(c.ctx, PipelineContext) and not c.ctx

    # after attach, property returns the attached instance
    c.attach_context(ctx)
    assert c.ctx is ctx
