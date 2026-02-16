"""Tensor <-> protobuf conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, get_args, get_origin

import numpy as np

from blazerpc.exceptions import SerializationError
from blazerpc.types import DTYPE_MAP, PYTHON_TYPE_MAP, _TensorType

# Reverse mapping: proto type string -> numpy dtype.
_PROTO_TO_NUMPY: dict[str, type[np.generic]] = {v: k for k, v in DTYPE_MAP.items()}


@dataclass(slots=True)
class TensorProto:
    """Wire representation of a tensor."""

    shape: tuple[int, ...]
    dtype: str
    data: bytes


def serialize_tensor(arr: np.ndarray) -> TensorProto:
    """Serialize a numpy array to a TensorProto."""
    dtype_str = DTYPE_MAP.get(arr.dtype.type)
    if dtype_str is None:
        raise SerializationError(
            f"Unsupported numpy dtype: {arr.dtype}", dtype=str(arr.dtype)
        )
    contiguous = np.ascontiguousarray(arr)
    return TensorProto(
        shape=tuple(contiguous.shape),
        dtype=dtype_str,
        data=contiguous.tobytes(),
    )


def deserialize_tensor(proto: TensorProto) -> np.ndarray:
    """Deserialize a TensorProto back to a numpy array."""
    np_dtype = _PROTO_TO_NUMPY.get(proto.dtype)
    if np_dtype is None:
        raise SerializationError(
            f"Unknown proto dtype: {proto.dtype}", dtype=proto.dtype
        )
    arr = np.frombuffer(proto.data, dtype=np_dtype)
    return arr.reshape(proto.shape)


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
        return serialize_tensor(value)

    if isinstance(value, np.ndarray):
        return serialize_tensor(value)

    return value


def proto_to_python(proto: Any, type_hint: Any) -> Any:
    """Convert a proto-friendly representation back to Python.

    For TensorProto values, deserializes to numpy arrays.
    For scalars, returns the value unchanged.
    """
    if isinstance(proto, TensorProto):
        return deserialize_tensor(proto)

    if isinstance(type_hint, _TensorType):
        if isinstance(proto, TensorProto):
            return deserialize_tensor(proto)
        raise SerializationError(
            f"Expected TensorProto for tensor field, got {type(proto).__name__}"
        )

    return proto
