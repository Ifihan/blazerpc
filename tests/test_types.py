"""Tests for the BlazeRPC type system."""

from __future__ import annotations

import numpy as np

from blazerpc.types import (
    DTYPE_MAP,
    PYTHON_TYPE_MAP,
    TensorInput,
    TensorOutput,
    _TensorType,
    extract_type_info,
)


# -- _TensorType --


def test_tensor_input_class_getitem() -> None:
    t = TensorInput[np.float32, "batch", 224, 224, 3]
    assert isinstance(t, _TensorType)
    assert t.dtype is np.float32
    assert t.shape == ("batch", 224, 224, 3)
    assert t.is_input is True


def test_tensor_output_class_getitem() -> None:
    t = TensorOutput[np.int64, "batch", 1000]
    assert isinstance(t, _TensorType)
    assert t.dtype is np.int64
    assert t.shape == ("batch", 1000)
    assert t.is_input is False


def test_tensor_type_repr() -> None:
    t = TensorInput[np.float32, 3, 3]
    assert "TensorInput" in repr(t)
    assert "float32" in repr(t)


def test_tensor_type_proto_type() -> None:
    t = TensorInput[np.float32, 10]
    assert t.proto_type() == "float"

    t2 = TensorOutput[np.int64, 5]
    assert t2.proto_type() == "int64"


def test_tensor_type_proto_type_unknown_dtype() -> None:
    t = _TensorType(dtype=object, shape=(1,), is_input=True)  # type: ignore[arg-type]
    assert t.proto_type() == "bytes"


# -- DTYPE_MAP --


def test_dtype_map_has_common_types() -> None:
    assert np.float32 in DTYPE_MAP
    assert np.float64 in DTYPE_MAP
    assert np.int32 in DTYPE_MAP
    assert np.int64 in DTYPE_MAP
    assert np.bool_ in DTYPE_MAP


def test_python_type_map_has_common_types() -> None:
    assert float in PYTHON_TYPE_MAP
    assert int in PYTHON_TYPE_MAP
    assert str in PYTHON_TYPE_MAP
    assert bool in PYTHON_TYPE_MAP


# -- extract_type_info --


def test_extract_type_info_simple() -> None:
    def predict(text: str) -> float:
        return 1.0

    info = extract_type_info(predict)
    assert "text" in info["inputs"]
    assert info["inputs"]["text"] is str
    assert info["output"] is float


def test_extract_type_info_multiple_params() -> None:
    def predict(text: list[str], count: int) -> list[float]:
        return [1.0]

    info = extract_type_info(predict)
    assert len(info["inputs"]) == 2
    assert "text" in info["inputs"]
    assert "count" in info["inputs"]


def test_extract_type_info_tensor_types() -> None:
    def predict(
        x: TensorInput[np.float32, "batch", 224, 224, 3],
    ) -> TensorOutput[np.float32, "batch", 1000]:
        ...

    info = extract_type_info(predict)
    assert isinstance(info["inputs"]["x"], _TensorType)
    assert isinstance(info["output"], _TensorType)


def test_extract_type_info_no_return() -> None:
    def predict(text: str) -> None:
        pass

    info = extract_type_info(predict)
    assert info["output"] is type(None)
