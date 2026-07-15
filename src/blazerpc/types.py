"""TensorInput, TensorOutput, and the BlazeRPC type system."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Generic, Literal, TypeAlias, TypeVar
from typing import TYPE_CHECKING, get_args, get_origin

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


def validate_tensor_type(type_hint: _TensorType) -> None:
    """Validate that a tensor annotation can be represented on the wire."""
    if type_hint.dtype not in DTYPE_MAP:
        name = getattr(type_hint.dtype, "__name__", repr(type_hint.dtype))
        raise ValueError(f"Unsupported tensor annotation dtype: {name}")
    for dimension in type_hint.shape:
        if isinstance(dimension, bool) or not isinstance(dimension, (int, str)):
            raise ValueError(
                "Tensor shape dimensions must be nonnegative integers or symbols"
            )
        if isinstance(dimension, int) and dimension < 0:
            raise ValueError("Tensor shape dimensions must be nonnegative")
        if isinstance(dimension, str) and not dimension:
            raise ValueError("Tensor shape symbols must not be empty")


def _shape_dimensions(annotation: Any) -> tuple[Any, ...]:
    if get_origin(annotation) is not tuple:
        return (annotation,)
    dimensions: list[Any] = []
    for dimension in get_args(annotation):
        if get_origin(dimension) is Literal:
            dimensions.extend(get_args(dimension))
        else:
            dimensions.append(dimension)
    return tuple(dimensions)


if TYPE_CHECKING:
    # Static form: TensorInput[dtype, tuple[Literal[dim], ...]]. At runtime the
    # compatibility classes below also accept the original variadic syntax.
    class _ShapedArray(np.ndarray[Any, np.dtype[Any]], Generic[Shape]):
        pass

    TensorInput: TypeAlias = np.ndarray[Any, np.dtype[DType]] | _ShapedArray[Shape]
    TensorOutput: TypeAlias = np.ndarray[Any, np.dtype[DType]] | _ShapedArray[Shape]
else:

    class TensorInput(Generic[DType, Shape]):
        """Tensor input annotation.

        Type-checked code should pass shape dimensions as a tuple of
        ``Literal`` values. The original variadic shape form remains supported
        at runtime for compatibility.
        """

        @classmethod
        def __class_getitem__(cls, params: tuple[Any, ...]) -> _TensorType:
            dtype, *shape = params
            if len(shape) == 1:
                shape = list(_shape_dimensions(shape[0]))
            return _TensorType(dtype, tuple(shape), is_input=True)

    class TensorOutput(Generic[DType, Shape]):
        """Tensor output annotation.

        Type-checked code should pass shape dimensions as a tuple of
        ``Literal`` values. The original variadic shape form remains supported
        at runtime for compatibility.
        """

        @classmethod
        def __class_getitem__(cls, params: tuple[Any, ...]) -> _TensorType:
            dtype, *shape = params
            if len(shape) == 1:
                shape = list(_shape_dimensions(shape[0]))
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

    annotation_target: Any = func
    if not (inspect.isfunction(func) or inspect.ismethod(func)):
        annotation_target = getattr(func, "__call__")
    hints = inspect.get_annotations(annotation_target, eval_str=True)
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
    if output is None and "return" in hints:
        output = type(None)

    return {
        "inputs": inputs,
        "deps": deps,
        "context_params": context_params,
        "output": output,
    }
