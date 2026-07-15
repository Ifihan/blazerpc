"""Tensor <-> protobuf conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from blazerpc.exceptions import SerializationError
from blazerpc.types import DTYPE_MAP, _TensorType

# Reverse mapping: proto type string -> numpy dtype.
_PROTO_TO_NUMPY: dict[str, type[np.generic]] = {v: k for k, v in DTYPE_MAP.items()}
_PROTO_TO_NUMPY.update({np.dtype(dtype).name: dtype for dtype in DTYPE_MAP})

MAX_TENSOR_RANK = 32
MAX_TENSOR_BYTES = 2**31 - 1


@dataclass(slots=True)
class TensorProto:
    """Wire representation of a tensor."""

    shape: tuple[int, ...]
    dtype: str
    data: bytes


def _validate_shape(shape: Any, itemsize: int, data_length: int) -> tuple[int, ...]:
    if not isinstance(shape, (tuple, list)):
        raise SerializationError("Tensor shape must be a list or tuple")
    if len(shape) > MAX_TENSOR_RANK:
        raise SerializationError(f"Tensor rank exceeds maximum {MAX_TENSOR_RANK}")

    dimensions: list[int] = []
    zero_size = False
    element_count = 1
    max_elements = MAX_TENSOR_BYTES // itemsize
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise SerializationError("Tensor dimensions must be integers")
        if dimension < 0:
            raise SerializationError("Tensor dimensions must be nonnegative")
        if dimension > max_elements:
            raise SerializationError("Tensor shape is too large")
        dimensions.append(dimension)
        if dimension == 0:
            zero_size = True
        elif not zero_size:
            if element_count > max_elements // dimension:
                raise SerializationError("Tensor shape is too large")
            element_count *= dimension

    expected_length = 0 if zero_size else element_count * itemsize
    if expected_length != data_length:
        raise SerializationError(
            f"Tensor byte length mismatch: expected {expected_length}, got {data_length}"
        )
    return tuple(dimensions)


def _validate_contract(arr: np.ndarray, type_hint: _TensorType) -> None:
    expected_dtype = np.dtype(type_hint.dtype)
    if arr.dtype != expected_dtype:
        raise SerializationError(
            f"Tensor dtype mismatch: expected {expected_dtype.name}, got {arr.dtype.name}",
            dtype=arr.dtype.name,
        )
    if arr.ndim != len(type_hint.shape):
        raise SerializationError(
            f"Tensor rank mismatch: expected {len(type_hint.shape)}, got {arr.ndim}"
        )

    symbols: dict[str, int] = {}
    for index, (expected, actual) in enumerate(zip(type_hint.shape, arr.shape)):
        if isinstance(expected, int) and actual != expected:
            raise SerializationError(
                f"Tensor dimension {index} mismatch: expected {expected}, got {actual}"
            )
        if isinstance(expected, str):
            previous = symbols.setdefault(expected, actual)
            if previous != actual:
                raise SerializationError(
                    f"Tensor symbol '{expected}' mismatch: expected {previous}, got {actual}"
                )


def serialize_tensor(
    arr: np.ndarray, type_hint: _TensorType | None = None
) -> TensorProto:
    """Serialize a numpy array to a TensorProto."""
    if type_hint is not None:
        _validate_contract(arr, type_hint)
    dtype_str = DTYPE_MAP.get(arr.dtype.type)
    if dtype_str is None or arr.dtype != np.dtype(arr.dtype.type):
        raise SerializationError(
            f"Unsupported numpy dtype: {arr.dtype}", dtype=str(arr.dtype)
        )
    contiguous = np.ascontiguousarray(arr)
    _validate_shape(contiguous.shape, contiguous.dtype.itemsize, contiguous.nbytes)
    return TensorProto(
        shape=tuple(contiguous.shape),
        dtype=dtype_str,
        data=contiguous.tobytes(),
    )


def deserialize_tensor(
    proto: TensorProto, type_hint: _TensorType | None = None
) -> np.ndarray:
    """Deserialize a TensorProto back to a numpy array."""
    if not isinstance(proto.data, (bytes, bytearray, memoryview)):
        raise SerializationError("Tensor data must be bytes")
    np_dtype = _PROTO_TO_NUMPY.get(proto.dtype)
    if np_dtype is None:
        raise SerializationError(
            f"Unknown proto dtype: {proto.dtype}", dtype=proto.dtype
        )
    dtype = np.dtype(np_dtype)
    shape = _validate_shape(proto.shape, dtype.itemsize, len(proto.data))
    arr = np.frombuffer(proto.data, dtype=dtype).reshape(shape)
    if type_hint is not None:
        _validate_contract(arr, type_hint)
    return arr


def python_to_proto(value: Any, type_hint: Any) -> Any:
    """Convert a Python value to its proto-friendly representation.

    For scalars and lists of scalars, returns the value unchanged.
    For numpy arrays / _TensorType hints, returns a TensorProto.
    """
    if isinstance(type_hint, _TensorType):
        if not isinstance(value, np.ndarray):
            raise SerializationError(
                f"Expected numpy array for tensor field, got {type(value).__name__}"
            )
        return serialize_tensor(value, type_hint)

    if isinstance(value, np.ndarray):
        return serialize_tensor(value)

    return value


def proto_to_python(proto: Any, type_hint: Any) -> Any:
    """Convert a proto-friendly representation back to Python.

    For TensorProto values, deserializes to numpy arrays.
    For scalars, returns the value unchanged.
    """
    if isinstance(proto, TensorProto):
        hint = type_hint if isinstance(type_hint, _TensorType) else None
        return deserialize_tensor(proto, hint)

    if isinstance(type_hint, _TensorType):
        if isinstance(proto, TensorProto):
            return deserialize_tensor(proto, type_hint)
        raise SerializationError(
            f"Expected TensorProto for tensor field, got {type(proto).__name__}"
        )

    return proto
