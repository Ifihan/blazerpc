"""JSON-compatible tensor serialization.

Provides base64-encoded tensor encoding for JSON-RPC transport,
analogous to the binary Protobuf encoding in :mod:`serialization`.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from blazerpc.exceptions import SerializationError
from blazerpc.runtime.serialization import _PROTO_TO_NUMPY
from blazerpc.types import DTYPE_MAP, _TensorType


def tensor_to_json(arr: np.ndarray) -> dict[str, Any]:
    """Serialize a numpy array to a JSON-safe dict.

    Returns ``{"shape": [...], "dtype": "float", "data": "<base64>"}``.
    """
    dtype_str = DTYPE_MAP.get(arr.dtype.type)
    if dtype_str is None:
        raise SerializationError(
            f"Unsupported numpy dtype: {arr.dtype}", dtype=str(arr.dtype)
        )
    contiguous = np.ascontiguousarray(arr)
    return {
        "shape": list(contiguous.shape),
        "dtype": dtype_str,
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def tensor_from_json(obj: dict[str, Any]) -> np.ndarray:
    """Deserialize a JSON tensor dict back to a numpy array."""
    dtype_str = obj.get("dtype")
    if dtype_str is None:
        raise SerializationError("Tensor JSON missing 'dtype' field")
    np_dtype = _PROTO_TO_NUMPY.get(dtype_str)
    if np_dtype is None:
        raise SerializationError(f"Unknown tensor dtype: {dtype_str}", dtype=dtype_str)
    raw = base64.b64decode(obj["data"])
    arr = np.frombuffer(raw, dtype=np_dtype)
    shape = obj.get("shape")
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


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
        return tensor_to_json(value)
    return value


def json_to_python(value: Any, type_hint: Any) -> Any:
    """Convert a JSON value back to Python.

    Tensor dicts become numpy arrays; scalars pass through unchanged.
    """
    if isinstance(type_hint, _TensorType) or is_tensor_json(value):
        if is_tensor_json(value):
            return tensor_from_json(value)
        raise SerializationError(
            f"Expected tensor dict for tensor field, got {type(value).__name__}"
        )
    return value
