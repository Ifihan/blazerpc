"""TensorInput, TensorOutput, and the BlazeRPC type system."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Generic, TypeVar, get_type_hints

import numpy as np

DType = TypeVar("DType", bound=np.generic)
Shape = TypeVar("Shape")

# Mapping from numpy dtypes to protobuf type strings.
DTYPE_MAP: dict[type, str] = {
    np.float32: "float",
    np.float64: "double",
    np.int32: "int32",
    np.int64: "int64",
    np.uint32: "uint32",
    np.uint64: "uint64",
    np.bool_: "bool",
    np.bytes_: "bytes",
    np.str_: "string",
}

# Mapping from Python scalar types to protobuf type strings.
PYTHON_TYPE_MAP: dict[type, str] = {
    float: "float",
    int: "int64",
    str: "string",
    bool: "bool",
    bytes: "bytes",
}


class _TensorType:
    """Internal representation of a tensor type with shape and dtype info."""

    def __init__(self, dtype: type, shape: tuple[Any, ...], *, is_input: bool) -> None:
        self.dtype = dtype
        self.shape = shape
        self.is_input = is_input

    def proto_type(self) -> str:
        """Return the protobuf field type string for this tensor's dtype."""
        return DTYPE_MAP.get(self.dtype, "bytes")

    def __repr__(self) -> str:
        kind = "TensorInput" if self.is_input else "TensorOutput"
        return f"{kind}[{self.dtype.__name__}, {self.shape}]"


class TensorInput(Generic[DType, Shape]):
    """Type annotation for tensor inputs with shape info."""

    @classmethod
    def __class_getitem__(cls, params: tuple[Any, ...]) -> _TensorType:
        dtype, *shape = params
        return _TensorType(dtype, tuple(shape), is_input=True)


class TensorOutput(Generic[DType, Shape]):
    """Type annotation for tensor outputs with shape info."""

    @classmethod
    def __class_getitem__(cls, params: tuple[Any, ...]) -> _TensorType:
        dtype, *shape = params
        return _TensorType(dtype, tuple(shape), is_input=False)


def extract_type_info(func: Callable[..., Any]) -> dict[str, Any]:
    """Extract input parameter types, dependencies, and return type.

    Returns a dict with keys:
        ``"inputs"``: ``dict[str, Any]`` mapping parameter names to their
            type annotations (may be plain Python types or ``_TensorType``).
            Only includes parameters that become Protobuf request fields.
        ``"deps"``: ``dict[str, Depends]`` mapping parameter names to their
            ``Depends`` instances (injected at request time).
        ``"context_params"``: ``list[str]`` of parameter names annotated
            with :class:`~blazerpc.context.Context`.
        ``"output"``: the return type annotation, or ``None`` if absent.
    """
    from blazerpc.context import Context, Depends  # local import avoids circular

    hints = get_type_hints(func)
    sig = inspect.signature(func)

    inputs: dict[str, Any] = {}
    deps: dict[str, Any] = {}
    context_params: list[str] = []

    for name, param in sig.parameters.items():
        default = param.default
        annotation = hints.get(name, Any)

        if isinstance(default, Depends):
            deps[name] = default
        elif annotation is Context:
            context_params.append(name)
        else:
            inputs[name] = annotation

    output = hints.get("return")

    return {
        "inputs": inputs,
        "deps": deps,
        "context_params": context_params,
        "output": output,
    }
