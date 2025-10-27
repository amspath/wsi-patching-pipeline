import collections.abc as cabc
from typing import Any, Iterable, Union, get_args, get_origin


class StageMeta(type):
    def __new__(mcls, name, bases, ns, **kw):
        cls = super().__new__(mcls, name, bases, ns)
        in_t = getattr(cls, "input_type", object)
        out_t = getattr(cls, "output_type", object)
        in_t, out_t = _infer_types_from_annotations(
            ns.get("__call__"), in_key="it", out_key="return", default_in=in_t, default_out=out_t
        )
        cls.input_type, cls.output_type = in_t, out_t
        if cls.input_type is object or cls.output_type is object:
            raise ValueError(f"{name}: __call__ must have type annotations for both 'it' and return")
        return cls


class WriterMeta(type):
    def __new__(mcls, name, bases, ns, **kw):
        cls = super().__new__(mcls, name, bases, ns)
        in_t = getattr(cls, "input_type", object)
        in_t, _ = _infer_types_from_annotations(
            ns.get("write") or ns.get("stream"), in_key="batch", default_in=in_t, default_out=None
        )
        cls.input_type = in_t
        if cls.input_type is object:
            raise ValueError(f"{name}: write() must annotate its 'batch' parameter type")
        return cls


def _infer_types_from_annotations(func, in_key=None, out_key=None, default_in=object, default_out=object):
    in_t, out_t = default_in, default_out
    if func and hasattr(func, "__annotations__"):
        ann = func.__annotations__
        if in_key:
            in_t = _iter_payload(ann.get(in_key)) or in_t
        if out_key:
            out_t = _iter_payload(ann.get(out_key)) or out_t
    return in_t, out_t


class PipelineContext(dict):
    def require_key(self, key: str):
        if key not in self:
            raise KeyError(f"Missing required context key: '{key}'")


class ContextAware:
    """Mixin for pipeline components that want a PipelineContext."""

    def attach_context(self, ctx: PipelineContext) -> None:
        self._ctx = ctx

    def export_context(self, ctx: PipelineContext) -> None:
        pass

    def validate(self) -> None:
        pass

    @property
    def ctx(self) -> PipelineContext:
        return getattr(self, "_ctx", PipelineContext())


# -------- Annotation utilities --------
def _iter_payload(t: Any) -> Any:
    """Extract T from Iterable[T]; support Union[T1, T2] and | unions."""
    if t is None:
        return object
    origin = get_origin(t)

    # Iterable[T]
    if origin in (list, tuple, set, frozenset, iter, Iterable, cabc.Iterable):
        args = get_args(t)
        if not args:
            return object
        payload = args[0]
        return _unwrap_union(payload)

    # Already a Union or plain class
    return _unwrap_union(t)


def _unwrap_union(t: Any) -> Any:
    origin = get_origin(t)
    if origin is Union or str(origin).endswith("types.UnionType"):
        return tuple(a for a in get_args(t) if a is not type(None))
    return t
