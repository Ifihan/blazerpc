"""Tests for tensor serialization."""

from __future__ import annotations

import numpy as np
import pytest

from blazerpc.exceptions import SerializationError
from blazerpc.runtime.serialization import (
    TensorProto,
    deserialize_tensor,
    proto_to_python,
    python_to_proto,
    serialize_tensor,
)
from blazerpc.types import TensorInput, _TensorType


# -- round-trip --


@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.int32, np.int64, np.bool_],
)
def test_roundtrip_various_dtypes(dtype: type[np.generic]) -> None:
    arr = np.array([1, 2, 3], dtype=dtype)
    proto = serialize_tensor(arr)
    result = deserialize_tensor(proto)
    np.testing.assert_array_equal(arr, result)


def test_roundtrip_multidimensional() -> None:
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    proto = serialize_tensor(arr)
    result = deserialize_tensor(proto)
    np.testing.assert_array_equal(arr, result)
    assert result.shape == (2, 3, 4)


def test_roundtrip_scalar() -> None:
    arr = np.array(42.0, dtype=np.float64)
    proto = serialize_tensor(arr)
    result = deserialize_tensor(proto)
    np.testing.assert_array_equal(arr, result)


def test_roundtrip_empty() -> None:
    arr = np.array([], dtype=np.float32)
    proto = serialize_tensor(arr)
    result = deserialize_tensor(proto)
    assert result.shape == (0,)


# -- serialize errors --


def test_serialize_unsupported_dtype() -> None:
    arr = np.array([1 + 2j], dtype=np.complex128)
    with pytest.raises(SerializationError, match="Unsupported numpy dtype"):
        serialize_tensor(arr)


# -- deserialize errors --


def test_deserialize_unknown_dtype() -> None:
    proto = TensorProto(shape=(2,), dtype="unknown_type", data=b"\x00" * 8)
    with pytest.raises(SerializationError, match="Unknown proto dtype"):
        deserialize_tensor(proto)


# -- python_to_proto / proto_to_python --


def test_python_to_proto_tensor() -> None:
    tensor_type = TensorInput[np.float32, 3]
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = python_to_proto(arr, tensor_type)
    assert isinstance(result, TensorProto)


def test_python_to_proto_scalar() -> None:
    result = python_to_proto(42, int)
    assert result == 42


def test_python_to_proto_tensor_type_wrong_value() -> None:
    tensor_type = TensorInput[np.float32, 3]
    with pytest.raises(SerializationError, match="Expected numpy array"):
        python_to_proto("not an array", tensor_type)


def test_proto_to_python_tensor_proto() -> None:
    arr = np.array([1.0, 2.0], dtype=np.float32)
    proto = serialize_tensor(arr)
    result = proto_to_python(proto, float)
    np.testing.assert_array_equal(result, arr)


def test_proto_to_python_scalar() -> None:
    result = proto_to_python(42, int)
    assert result == 42
