"""Unit tests for BlazeClient."""

from __future__ import annotations

import numpy as np
import pytest

from blazerpc.app import BlazeApp
from blazerpc.client import BlazeClient, _build_path, _encode_kwargs
from blazerpc.codegen.proto_types import _TensorProtoMsg
from blazerpc.exceptions import SerializationError
from blazerpc.types import TensorInput, TensorOutput


def test_build_path_simple() -> None:
    assert _build_path("echo") == "/blazerpc.InferenceService/PredictEcho"


def test_build_path_underscore() -> None:
    assert _build_path("my_model") == "/blazerpc.InferenceService/PredictMyModel"


def test_build_path_hyphen() -> None:
    assert _build_path("my-model") == "/blazerpc.InferenceService/PredictMyModel"


def test_build_path_versioned() -> None:
    assert _build_path("echo", "2") == "/blazerpc.InferenceService/PredictEchoV2"


def test_client_defaults() -> None:
    client = BlazeClient()
    assert client._host == "127.0.0.1"
    assert client._port == 50051
    assert client._channel is None


def test_client_close_without_connect() -> None:
    client = BlazeClient()
    client.close()  # should not raise


def test_encode_kwargs_preserves_scalar_and_list_values() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("mixed")
    def mixed(values: list[float], label: str) -> str:
        return label

    kwargs = {"values": [1.0, 2.0], "label": "sample"}
    encoded = _encode_kwargs(kwargs, app.registry.get("mixed"))

    assert encoded == kwargs
    assert encoded is not kwargs


def test_encode_kwargs_builds_dynamic_tensor_message() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("identity")
    def identity(
        values: TensorInput[np.float32, 2],
    ) -> TensorOutput[np.float32, 2]:
        return values

    values = np.array([1.0, 2.0], dtype=np.float32)
    encoded = _encode_kwargs({"values": values}, app.registry.get("identity"))

    assert isinstance(encoded["values"], _TensorProtoMsg)
    assert encoded["values"].shape == [2]
    assert encoded["values"].dtype == "float"
    assert encoded["values"].data == values.tobytes()


def test_encode_kwargs_rejects_non_numpy_tensor_with_field_context() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("identity")
    def identity(
        values: TensorInput[np.float32, 2],
    ) -> TensorOutput[np.float32, 2]:
        return values

    with pytest.raises(
        SerializationError,
        match="Invalid tensor input 'values' for model 'identity'.*got list",
    ):
        _encode_kwargs({"values": [1.0, 2.0]}, app.registry.get("identity"))


def test_encode_kwargs_rejects_missing_tensor_with_field_context() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("identity")
    def identity(
        values: TensorInput[np.float32, 2],
    ) -> TensorOutput[np.float32, 2]:
        return values

    with pytest.raises(
        SerializationError, match="Missing tensor input 'values' for model 'identity'"
    ):
        _encode_kwargs({}, app.registry.get("identity"))
