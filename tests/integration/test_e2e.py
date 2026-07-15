"""End-to-end integration tests.

These tests verify full register → serve → call → response flows using
grpclib's in-process server with real binary Protobuf encoding via betterproto.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from grpclib.client import Channel
from grpclib.const import Cardinality
from grpclib.health.v1.health_grpc import HealthStub
from grpclib.health.v1.health_pb2 import HealthCheckRequest, HealthCheckResponse
from grpclib.reflection.v1.reflection_grpc import ServerReflectionStub
from grpclib.reflection.v1.reflection_pb2 import ServerReflectionRequest
from grpclib.server import Server
from google.protobuf.descriptor_pb2 import FieldDescriptorProto, FileDescriptorProto

from blazerpc.app import BlazeApp
from blazerpc.codegen.proto_types import _TensorProtoMsg, build_message_classes
from blazerpc.codegen.servicer import build_servicer
from blazerpc.context import Context, Depends
from blazerpc.server.grpc import RawCodec
from blazerpc.server.health import build_health_service
from blazerpc.server.reflection import build_reflection_service
from blazerpc.types import TensorInput, TensorOutput


@pytest.mark.asyncio
async def test_server_with_health_starts_and_stops() -> None:
    """A server with both the inference servicer and health can start."""
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    servicer = build_servicer(app.registry)
    health = build_health_service([servicer])

    server = Server([servicer, health], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_standard_health_client_over_wire() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    servicer = build_servicer(app.registry)
    health = build_health_service([servicer])
    server = Server([servicer, health], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    channel = Channel("127.0.0.1", _get_server_port(server))
    try:
        response = await HealthStub(channel).Check(
            HealthCheckRequest(service="blazerpc.InferenceService")
        )
        assert response.status == HealthCheckResponse.SERVING
    finally:
        channel.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_reflection_returns_inference_file_descriptor() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("score_model")
    def score_model(features: list[float], label: str) -> int:
        return len(features) + len(label)

    servicer = build_servicer(app.registry)
    reflection_handlers = build_reflection_service([servicer])
    server = Server(reflection_handlers, codec=RawCodec())
    await server.start("127.0.0.1", 0)
    channel = Channel("127.0.0.1", _get_server_port(server))
    try:
        stub = ServerReflectionStub(channel)
        async with stub.ServerReflectionInfo.open() as stream:
            await stream.send_message(
                ServerReflectionRequest(
                    file_containing_symbol="blazerpc.InferenceService"
                ),
                end=True,
            )
            response = await stream.recv_message()

        serialized = response.file_descriptor_response.file_descriptor_proto
        assert len(serialized) == 1
        descriptor = FileDescriptorProto.FromString(serialized[0])
        assert descriptor.name == "blaze_service.proto"

        service = next(s for s in descriptor.service if s.name == "InferenceService")
        method = next(m for m in service.method if m.name == "PredictScoreModel")
        assert method.input_type == ".blazerpc.ScoreModelRequest"
        assert method.output_type == ".blazerpc.ScoreModelResponse"

        messages = {message.name: message for message in descriptor.message_type}
        request_fields = {
            field.name: field for field in messages["ScoreModelRequest"].field
        }
        assert request_fields["features"].label == FieldDescriptorProto.LABEL_REPEATED
        assert request_fields["features"].type == request_fields["features"].TYPE_FLOAT
        assert request_fields["label"].type == request_fields["label"].TYPE_STRING
        assert (
            messages["ScoreModelResponse"].field[0].type
            == FieldDescriptorProto.TYPE_INT64
        )
    finally:
        channel.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_reflection_lists_inference_and_health_services_once() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    servicer = build_servicer(app.registry)
    health = build_health_service([servicer])
    handlers = build_reflection_service([servicer, health])
    mapping_paths = [path for handler in handlers for path in handler.__mapping__()]
    assert len(mapping_paths) == len(set(mapping_paths))

    server = Server(handlers, codec=RawCodec())
    await server.start("127.0.0.1", 0)
    channel = Channel("127.0.0.1", _get_server_port(server))
    try:
        stub = ServerReflectionStub(channel)
        async with stub.ServerReflectionInfo.open() as stream:
            await stream.send_message(
                ServerReflectionRequest(list_services=""), end=True
            )
            response = await stream.recv_message()

        services = [service.name for service in response.list_services_response.service]
        assert services.count("blazerpc.InferenceService") == 1
        assert services.count("grpc.health.v1.Health") == 1
    finally:
        channel.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_multiple_models_register() -> None:
    """Multiple models can be registered and produce a working servicer."""
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    @app.model("add")
    def add(a: float, b: float) -> float:
        return a + b

    servicer = build_servicer(app.registry)
    mapping = servicer.__mapping__()

    assert "/blazerpc.InferenceService/PredictEcho" in mapping
    assert "/blazerpc.InferenceService/PredictAdd" in mapping


@pytest.mark.asyncio
async def test_streaming_model_registers() -> None:
    """A streaming model creates a server-streaming RPC."""
    app = BlazeApp(enable_batching=False)

    @app.model("tokens", streaming=True)
    async def generate(prompt: str) -> str:
        for token in ["hello", " ", "world"]:
            yield token

    servicer = build_servicer(app.registry)
    mapping = servicer.__mapping__()

    path = "/blazerpc.InferenceService/PredictTokens"
    assert path in mapping
    handler = mapping[path]
    assert handler.cardinality == Cardinality.UNARY_STREAM


@pytest.mark.asyncio
async def test_async_model_execution() -> None:
    """Async model functions are awaited correctly via the servicer."""
    app = BlazeApp(enable_batching=False)

    @app.model("async_echo")
    async def async_echo(text: str) -> str:
        await asyncio.sleep(0.01)
        return f"async: {text}"

    servicer = build_servicer(app.registry)
    mapping = servicer.__mapping__()
    assert "/blazerpc.InferenceService/PredictAsyncEcho" in mapping


# ---------------------------------------------------------------------------
# Wire-level tests: send Protobuf bytes through grpclib and verify responses
# ---------------------------------------------------------------------------


def _get_server_port(server: Server) -> int:
    """Extract the OS-assigned port from a started server."""
    for sock in server._server.sockets:
        return sock.getsockname()[1]
    raise RuntimeError("Server has no sockets")


async def _unary_call(
    channel: Channel,
    path: str,
    request_bytes: bytes,
    response_cls: type,
) -> object:
    """Send a unary Protobuf request and return the parsed response message."""
    stream = channel.request(path, Cardinality.UNARY_UNARY, None, None)
    async with stream as s:
        await s.send_message(request_bytes, end=True)
        response_bytes = await s.recv_message()
    return response_cls().parse(response_bytes)


@pytest.mark.asyncio
async def test_unary_echo_over_wire() -> None:
    """Send a Protobuf request to an echo model and get Protobuf back."""
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return f"Echo: {text}"

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    model = app.registry.get("echo")
    req_cls, resp_cls = build_message_classes(model)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        request_bytes = bytes(req_cls(text="hello"))
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictEcho",
            request_bytes,
            resp_cls,
        )
        assert response.result == "Echo: hello"  # type: ignore[union-attr]
    finally:
        channel.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_unary_add_over_wire() -> None:
    """Send numeric inputs and verify the sum comes back."""
    app = BlazeApp(enable_batching=False)

    @app.model("add")
    def add(a: float, b: float) -> float:
        return a + b

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    model = app.registry.get("add")
    req_cls, resp_cls = build_message_classes(model)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        request_bytes = bytes(req_cls(a=2.5, b=3.5))
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictAdd",
            request_bytes,
            resp_cls,
        )
        assert abs(response.result - 6.0) < 1e-5  # type: ignore[union-attr]
    finally:
        channel.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_unary_list_over_wire() -> None:
    """A model that takes list[str] and returns list[float]."""
    app = BlazeApp(enable_batching=False)

    @app.model("sentiment")
    def predict(text: list[str]) -> list[float]:
        return [0.9] * len(text)

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    model = app.registry.get("sentiment")
    req_cls, resp_cls = build_message_classes(model)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        request_bytes = bytes(req_cls(text=["good", "bad"]))
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictSentiment",
            request_bytes,
            resp_cls,
        )
        assert len(response.result) == 2  # type: ignore[union-attr]
        assert all(abs(v - 0.9) < 1e-5 for v in response.result)  # type: ignore[union-attr]
    finally:
        channel.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_unary_tensor_over_wire() -> None:
    """Tensor inputs are Protobuf-encoded, processed, and decoded back."""
    app = BlazeApp(enable_batching=False)

    @app.model("double")
    def double(
        data: TensorInput[np.float32, 4],
    ) -> TensorOutput[np.float32, 4]:
        return (data * 2).astype(np.float32)

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    model = app.registry.get("double")
    req_cls, resp_cls = build_message_classes(model)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        tp = _TensorProtoMsg(shape=list(arr.shape), dtype="float", data=arr.tobytes())
        request_bytes = bytes(req_cls(data=tp))
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictDouble",
            request_bytes,
            resp_cls,
        )
        result_arr = np.frombuffer(
            response.result.data,
            dtype=np.float32,  # type: ignore[union-attr]
        )
        np.testing.assert_array_equal(result_arr, [2.0, 4.0, 6.0, 8.0])
    finally:
        channel.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_async_model_over_wire() -> None:
    """Async model functions work over the wire."""
    app = BlazeApp(enable_batching=False)

    @app.model("async_echo")
    async def async_echo(text: str) -> str:
        await asyncio.sleep(0.01)
        return f"async: {text}"

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    model = app.registry.get("async_echo")
    req_cls, resp_cls = build_message_classes(model)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        request_bytes = bytes(req_cls(text="world"))
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictAsyncEcho",
            request_bytes,
            resp_cls,
        )
        assert response.result == "async: world"  # type: ignore[union-attr]
    finally:
        channel.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_unary_with_batching_over_wire() -> None:
    """Compatible tensor RPCs use one model call and receive split results."""
    app = BlazeApp(enable_batching=True, max_batch_size=4, batch_timeout_ms=100)
    calls = 0

    @app.model("double_batched")
    def double_batched(
        values: TensorInput[np.float32, "batch", 2],  # noqa: F821
    ) -> TensorOutput[np.float32, "batch", 2]:  # noqa: F821
        nonlocal calls
        calls += 1
        return values * 2

    @app.model("add")
    def add(a: float, b: float) -> float:
        return a + b

    model = app.registry.get("double_batched")
    batchers = await app._create_batchers()
    assert set(batchers) == {"double_batched"}

    servicer = build_servicer(app.registry, batchers=batchers)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    req_cls, resp_cls = build_message_classes(model)
    add_req_cls, add_resp_cls = build_message_classes(app.registry.get("add"))

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        add_response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictAdd",
            bytes(add_req_cls(a=10.0, b=20.0)),
            add_resp_cls,
        )
        assert abs(add_response.result - 30.0) < 1e-5  # type: ignore[union-attr]

        arrays = [
            np.array([[1, 2]], dtype=np.float32),
            np.array([[3, 4], [5, 6]], dtype=np.float32),
        ]
        requests = [
            bytes(
                req_cls(
                    values=_TensorProtoMsg(
                        shape=list(array.shape),
                        dtype="float",
                        data=array.tobytes(),
                    )
                )
            )
            for array in arrays
        ]
        responses = await asyncio.gather(
            *[
                _unary_call(
                    channel,
                    "/blazerpc.InferenceService/PredictDoubleBatched",
                    request,
                    resp_cls,
                )
                for request in requests
            ]
        )
        results = [
            np.frombuffer(response.result.data, dtype=np.float32).reshape(  # type: ignore[union-attr]
                response.result.shape  # type: ignore[union-attr]
            )
            for response in responses
        ]
        assert calls == 1
        np.testing.assert_array_equal(results[0], arrays[0] * 2)
        np.testing.assert_array_equal(results[1], arrays[1] * 2)
    finally:
        channel.close()
        server.close()
        await server.wait_closed()
        for batcher in batchers.values():
            await batcher.stop()


@pytest.mark.asyncio
async def test_context_injection_over_wire() -> None:
    """A handler using Context receives the correct gRPC method path."""
    app = BlazeApp(enable_batching=False)
    app.state.prefix = "CTX"

    def get_prefix(ctx: Context) -> str:
        return ctx.app_state.prefix

    @app.model("ctx_echo")
    def ctx_echo(
        ctx: Context,
        text: str,
        prefix: str = Depends(get_prefix),
    ) -> str:
        return f"{prefix}:{ctx.method}:{text}"

    servicer = build_servicer(app.registry, app_state=app.state)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    model = app.registry.get("ctx_echo")
    req_cls, resp_cls = build_message_classes(model)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        request_bytes = bytes(req_cls(text="hello"))
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictCtxEcho",
            request_bytes,
            resp_cls,
        )
        result = response.result  # type: ignore[union-attr]
        assert result.startswith("CTX:")
        assert "/blazerpc.InferenceService/PredictCtxEcho" in result
        assert result.endswith(":hello")
    finally:
        channel.close()
        server.close()
        await server.wait_closed()
