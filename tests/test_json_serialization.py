"""Tests for JSON tensor serialization."""

from __future__ import annotations

import numpy as np
import pytest

from blazerpc.exceptions import SerializationError
from blazerpc.runtime.json_serialization import (
    is_tensor_json,
    json_to_python,
    python_to_json,
    tensor_from_json,
    tensor_to_json,
)
from blazerpc.types import TensorInput


# ---------------------------------------------------------------------------
# tensor_to_json / tensor_from_json round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int32, np.int64, np.bool_]
)
def test_roundtrip_various_dtypes(dtype: type) -> None:
    arr = np.array([1, 2, 3], dtype=dtype)
    json_obj = tensor_to_json(arr)
    restored = tensor_from_json(json_obj)
    np.testing.assert_array_equal(arr, restored)


def test_roundtrip_multidimensional() -> None:
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    json_obj = tensor_to_json(arr)
    restored = tensor_from_json(json_obj)
    np.testing.assert_array_equal(arr, restored)
    assert restored.shape == (3, 4)


def test_roundtrip_scalar() -> None:
    arr = np.array(42.0, dtype=np.float64)
    json_obj = tensor_to_json(arr)
    restored = tensor_from_json(json_obj)
    np.testing.assert_array_equal(arr, restored)


def test_roundtrip_empty() -> None:
    arr = np.array([], dtype=np.float32)
    json_obj = tensor_to_json(arr)
    restored = tensor_from_json(json_obj)
    assert len(restored) == 0


def test_tensor_to_json_format() -> None:
    arr = np.array([1.0, 2.0], dtype=np.float32)
    result = tensor_to_json(arr)
    assert "shape" in result
    assert "dtype" in result
    assert "data" in result
    assert result["shape"] == [2]
    assert isinstance(result["data"], str)  # base64 string


def test_unsupported_dtype_raises() -> None:
    arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
    with pytest.raises(SerializationError):
        tensor_to_json(arr)


def test_unknown_dtype_from_json_raises() -> None:
    with pytest.raises(SerializationError):
        tensor_from_json({"shape": [2], "dtype": "complex128", "data": "AAAA"})


def test_missing_dtype_from_json_raises() -> None:
    with pytest.raises(SerializationError):
        tensor_from_json({"shape": [2], "data": "AAAA"})


# ---------------------------------------------------------------------------
# is_tensor_json
# ---------------------------------------------------------------------------


def test_is_tensor_json_true() -> None:
    assert is_tensor_json({"shape": [2], "dtype": "float", "data": "AAAA"})


def test_is_tensor_json_false_not_dict() -> None:
    assert not is_tensor_json("hello")
    assert not is_tensor_json(42)


def test_is_tensor_json_false_missing_fields() -> None:
    assert not is_tensor_json({"shape": [2], "dtype": "float"})
    assert not is_tensor_json({"dtype": "float", "data": "AAAA"})


# ---------------------------------------------------------------------------
# python_to_json / json_to_python
# ---------------------------------------------------------------------------


def test_python_to_json_tensor() -> None:
    arr = np.array([1.0], dtype=np.float32)
    tensor_type = TensorInput[np.float32, 1]
    result = python_to_json(arr, tensor_type)
    assert isinstance(result, dict)
    assert "shape" in result


def test_python_to_json_scalar() -> None:
    assert python_to_json(42, int) == 42
    assert python_to_json("hello", str) == "hello"


def test_json_to_python_tensor() -> None:
    arr = np.array([1.0, 2.0], dtype=np.float32)
    json_obj = tensor_to_json(arr)
    tensor_type = TensorInput[np.float32, 2]
    restored = json_to_python(json_obj, tensor_type)
    np.testing.assert_array_equal(arr, restored)


def test_json_to_python_scalar() -> None:
    assert json_to_python(42, int) == 42
    assert json_to_python("hello", str) == "hello"
