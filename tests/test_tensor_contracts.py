"""Tensor annotation contract validation across transport boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from grpclib.const import Status
from grpclib.exceptions import GRPCError

from blazerpc.app import BlazeApp
from blazerpc.client import _encode_kwargs
from blazerpc.codegen.jsonrpc_handler import INTERNAL_ERROR, INVALID_PARAMS, JsonRpcDispatcher
from blazerpc.codegen.proto_types import _TensorProtoMsg, build_message_classes
from blazerpc.codegen.servicer import _make_unary_handler
from blazerpc.exceptions import SerializationError, ValidationError
from blazerpc.jsonrpc_client import _prepare_params, _restore_result
from blazerpc.runtime.json_serialization import tensor_from_json, tensor_to_json
from blazerpc.runtime.serialization import TensorProto, deserialize_tensor, serialize_tensor
from blazerpc.types import TensorInput, TensorOutput


def test_symbolic_shape_and_repeated_symbols() -> None:
    contract = TensorInput[np.float32, "side", "side"]
    arr = np.ones((3, 3), dtype=np.float32)

    restored = deserialize_tensor(serialize_tensor(arr, contract), contract)
    np.testing.assert_array_equal(restored, arr)

    with pytest.raises(SerializationError, match="symbol 'side' mismatch"):
        serialize_tensor(np.ones((2, 3), dtype=np.float32), contract)


@pytest.mark.parametrize(
    ("arr", "message"),
    [
        (np.ones((2, 3), dtype=np.float64), "dtype mismatch"),
        (np.ones((6,), dtype=np.float32), "rank mismatch"),
        (np.ones((2, 4), dtype=np.float32), "dimension 1 mismatch"),
    ],
)
def test_array_contract_mismatches(arr: np.ndarray, message: str) -> None:
    contract = TensorInput[np.float32, "batch", 3]
    with pytest.raises(SerializationError, match=message):
        serialize_tensor(arr, contract)


@pytest.mark.parametrize(
    "proto",
    [
        TensorProto(shape=(-1,), dtype="float", data=b""),
        TensorProto(shape=(2,), dtype="float", data=b"\x00" * 4),
        TensorProto(shape=(2**31,), dtype="float", data=b""),
        TensorProto(shape=(1,) * 33, dtype="float", data=b"\x00" * 4),
    ],
)
def test_rejects_unsafe_proto_shapes_and_lengths(proto: TensorProto) -> None:
    with pytest.raises(SerializationError):
        deserialize_tensor(proto)


@pytest.mark.parametrize("data", ["%%%", "AAAA\n", "A", 123])
def test_json_rejects_malformed_base64(data: Any) -> None:
    with pytest.raises(SerializationError):
        tensor_from_json({"shape": [1], "dtype": "float", "data": data})


def test_json_accepts_numpy_dtype_name_but_emits_existing_wire_name() -> None:
    arr = np.array([1.0], dtype=np.float32)
    payload = tensor_to_json(arr)
    assert payload["dtype"] == "float"

    payload["dtype"] = "float32"
    np.testing.assert_array_equal(tensor_from_json(payload), arr)


@pytest.mark.parametrize("dtype", [np.str_, np.bytes_])
def test_variable_width_dtype_rejected_at_registration(dtype: type) -> None:
    app = BlazeApp(enable_batching=False)

    def model(value: Any) -> float:
        return 1.0

    model.__annotations__["value"] = TensorInput[dtype, 1]
    with pytest.raises(ValidationError, match="Unsupported tensor annotation dtype"):
        app.registry.register("unsupported", "1", model)


def _identity_square(
    value: TensorInput[np.float32, "side", "side"],  # noqa: F821
) -> TensorOutput[np.float32, "side", "side"]:  # noqa: F821
    return value


def _bad_output(
    value: TensorInput[np.float32, 2],
) -> TensorOutput[np.float32, 2]:
    return np.ones(3, dtype=np.float32)


class _FakeStream:
    metadata: dict[str, str] = {}
    peer = "test"

    def __init__(self, request: bytes) -> None:
        self.request = request
        self.response: bytes | None = None

    async def recv_message(self) -> bytes:
        return self.request

    async def send_message(self, response: bytes) -> None:
        self.response = response


async def test_grpc_maps_input_and_output_contract_failures() -> None:
    app = BlazeApp(enable_batching=False)
    app.registry.register("bad_output", "1", _bad_output)
    model = app.registry.get("bad_output")
    request_cls, response_cls = build_message_classes(model)
    handler = _make_unary_handler(model, request_cls, response_cls)

    wrong_dtype = np.ones(2, dtype=np.float64)
    bad_request = request_cls(
        value=_TensorProtoMsg(
            shape=[2], dtype="double", data=wrong_dtype.tobytes()
        )
    )
    with pytest.raises(GRPCError) as input_error:
        await handler(_FakeStream(bytes(bad_request)))
    assert input_error.value.status is Status.INVALID_ARGUMENT

    short_request = request_cls(
        value=_TensorProtoMsg(shape=[2], dtype="float", data=b"\x00" * 4)
    )
    with pytest.raises(GRPCError) as malformed_error:
        await handler(_FakeStream(bytes(short_request)))
    assert malformed_error.value.status is Status.INVALID_ARGUMENT

    valid = np.ones(2, dtype=np.float32)
    valid_request = request_cls(
        value=_TensorProtoMsg(shape=[2], dtype="float", data=valid.tobytes())
    )
    with pytest.raises(GRPCError) as output_error:
        await handler(_FakeStream(bytes(valid_request)))
    assert output_error.value.status is Status.INTERNAL

    app.registry.register("square", "1", _identity_square)
    square = app.registry.get("square")
    square_request_cls, square_response_cls = build_message_classes(square)
    square_handler = _make_unary_handler(
        square, square_request_cls, square_response_cls
    )
    square_value = np.ones((2, 2), dtype=np.float32)
    square_request = square_request_cls(
        value=_TensorProtoMsg(
            shape=[2, 2], dtype="float", data=square_value.tobytes()
        )
    )
    stream = _FakeStream(bytes(square_request))
    await square_handler(stream)
    assert stream.response is not None


async def test_json_maps_input_and_output_contract_failures() -> None:
    app = BlazeApp(enable_batching=False)
    app.registry.register("square", "1", _identity_square)
    app.registry.register("bad_output", "1", _bad_output)
    dispatcher = JsonRpcDispatcher(app.registry)

    symbolic = tensor_to_json(np.ones((2, 2), dtype=np.float32))
    valid_response = await dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "method": "predict.square",
            "params": {"value": symbolic},
            "id": 0,
        }
    )
    assert "error" not in valid_response

    malformed = {"shape": [1], "dtype": "float", "data": "%%%"}
    malformed_response = await dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "method": "predict.square",
            "params": {"value": malformed},
            "id": 1,
        }
    )
    assert malformed_response["error"]["code"] == INVALID_PARAMS

    wrong_rank = tensor_to_json(np.ones(3, dtype=np.float32))
    input_response = await dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "method": "predict.square",
            "params": {"value": wrong_rank},
            "id": 2,
        }
    )
    assert input_response["error"]["code"] == INVALID_PARAMS

    valid = tensor_to_json(np.ones(2, dtype=np.float32))
    output_response = await dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "method": "predict.bad_output",
            "params": {"value": valid},
            "id": 3,
        }
    )
    assert output_response["error"]["code"] == INTERNAL_ERROR


def test_clients_validate_tensor_contracts_with_registry() -> None:
    app = BlazeApp(enable_batching=False)
    app.registry.register("square", "1", _identity_square)
    model = app.registry.get("square")

    with pytest.raises(SerializationError, match="rank mismatch"):
        _encode_kwargs({"value": np.ones(3, dtype=np.float32)}, model)
    with pytest.raises(SerializationError, match="rank mismatch"):
        _prepare_params({"value": np.ones(3, dtype=np.float32)}, model)
    with pytest.raises(SerializationError, match="symbol 'side' mismatch"):
        _restore_result(
            tensor_to_json(np.ones((2, 3), dtype=np.float32)), model.output_type
        )
