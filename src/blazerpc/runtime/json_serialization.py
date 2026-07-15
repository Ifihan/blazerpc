"""JSON-compatible tensor serialization.

Provides base64-encoded tensor encoding for JSON-RPC transport,
analogous to the binary Protobuf encoding in :mod:`serialization`.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

import numpy as np

from blazerpc.exceptions import SerializationError
from blazerpc.runtime.serialization import _PROTO_TO_NUMPY
from blazerpc.types import _TensorType


def tensor_to_json(
    arr: np.ndarray, type_hint: _TensorType | None = None
) -> dict[str, Any]:
    """Serialize a numpy array to a JSON-safe dict.

    Returns ``{"shape": [...], "dtype": "float", "data": "<base64>"}``.
    """
    from blazerpc.runtime.serialization import serialize_tensor

    tensor = serialize_tensor(arr, type_hint)
    return {
        "shape": list(tensor.shape),
        "dtype": tensor.dtype,
        "data": base64.b64encode(tensor.data).decode("ascii"),
    }


def tensor_from_json(
    obj: dict[str, Any], type_hint: _TensorType | None = None
) -> np.ndarray:
    """Deserialize a JSON tensor dict back to a numpy array."""
    if not isinstance(obj, dict):
        raise SerializationError("Tensor JSON must be an object")
    dtype_str = obj.get("dtype")
    if not isinstance(dtype_str, str):
        raise SerializationError("Tensor JSON missing 'dtype' field")
    np_dtype = _PROTO_TO_NUMPY.get(dtype_str)
    if np_dtype is None:
        raise SerializationError(f"Unknown tensor dtype: {dtype_str}", dtype=dtype_str)
    data = obj.get("data")
    if not isinstance(data, str):
        raise SerializationError("Tensor JSON 'data' must be a base64 string")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SerializationError("Tensor JSON contains malformed base64 data") from exc
    shape = obj.get("shape")
    if not isinstance(shape, list):
        raise SerializationError("Tensor JSON 'shape' must be a list")

    from blazerpc.runtime.serialization import TensorProto, deserialize_tensor

    return deserialize_tensor(
        TensorProto(shape=tuple(shape), dtype=dtype_str, data=raw), type_hint
    )


def is_tensor_json(obj: Any) -> bool:
    """Check whether *obj* looks like a serialized tensor dict."""
    return isinstance(obj, dict) and "shape" in obj and "dtype" in obj and "data" in obj


def python_to_json(value: Any, type_hint: Any) -> Any:
    """Convert a Python value to its JSON-safe representation.

    Numpy arrays become tensor dicts; scalars pass through unchanged.
    """
    if isinstance(type_hint, _TensorType) or isinstance(value, np.ndarray):
        if not isinstance(value, np.ndarray):
            raise SerializationError(
                f"Expected numpy array for tensor field, got {type(value).__name__}"
            )
        hint = type_hint if isinstance(type_hint, _TensorType) else None
        return tensor_to_json(value, hint)
    return value


def json_to_python(value: Any, type_hint: Any) -> Any:
    """Convert a JSON value back to Python.

    Tensor dicts become numpy arrays; scalars pass through unchanged.
    """
    if isinstance(type_hint, _TensorType) or is_tensor_json(value):
        if is_tensor_json(value):
            hint = type_hint if isinstance(type_hint, _TensorType) else None
            return tensor_from_json(value, hint)
        raise SerializationError(
            f"Expected tensor dict for tensor field, got {type(value).__name__}"
        )
    return value
