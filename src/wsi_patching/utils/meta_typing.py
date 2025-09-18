import collections.abc as cabc
from typing import Any, Iterable, Union, get_args, get_origin


class StageMeta(type):
    def __new__(mcls, name, bases, ns, **kw):
        cls = super().__new__(mcls, name, bases, ns)
        # Defaults
        in_t = getattr(cls, "input_type", object)
        out_t = getattr(cls, "output_type", object)
        call = ns.get("__call__")
        if call and hasattr(call, "__annotations__"):
            ann = call.__annotations__
            in_t = _iter_payload(ann.get("it", None)) or in_t
            out_t = _iter_payload(ann.get("return", None)) or out_t
        cls.input_type = in_t
        cls.output_type = out_t
        return cls


class WriterMeta(type):
    def __new__(mcls, name, bases, ns, **kw):
        cls = super().__new__(mcls, name, bases, ns)
        # Defaults
        in_t = getattr(cls, "input_type", object)
        write_function = ns.get("write")
        if write_function and hasattr(write_function, "__annotations__"):
            ann = write_function.__annotations__
            in_t = _iter_payload(ann.get("sample", None)) or in_t
        cls.input_type = in_t
        return cls


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
