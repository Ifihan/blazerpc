"""Async gRPC client for BlazeRPC services."""

from __future__ import annotations

from typing import Any, AsyncIterator

from grpclib.client import Channel
from grpclib.const import Cardinality

from blazerpc.codegen.proto import _sanitize_name
from blazerpc.codegen.proto_types import _TensorProtoMsg, build_message_classes
from blazerpc.exceptions import SerializationError
from blazerpc.runtime.registry import ModelInfo, ModelRegistry
from blazerpc.runtime.serialization import TensorProto, proto_to_python, python_to_proto
from blazerpc.server.grpc import RawCodec
from blazerpc.types import _TensorType

SERVICE_NAME = "blazerpc.InferenceService"


class BlazeClient:
    """Async gRPC client for calling BlazeRPC model endpoints.

    Usage::

        async with BlazeClient("127.0.0.1", 50051, registry=app.registry) as client:
            result = await client.predict("echo", text="hello")
            async for chunk in client.stream("tokens", prompt="hi"):
                print(chunk)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 50051,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._registry = registry
        self._channel: Channel | None = None

    def _ensure_channel(self) -> Channel:
        if self._channel is None:
            self._channel = Channel(self._host, self._port, codec=RawCodec())
        return self._channel

    async def __aenter__(self) -> BlazeClient:
        self._ensure_channel()
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None

    async def predict(self, model_name: str, **kwargs: Any) -> Any:
        """Make a unary prediction call to a model.

        Parameters
        ----------
        model_name:
            The registered model name (e.g. ``"echo"``, ``"add"``).
        **kwargs:
            Input fields matching the model function's parameters.

        Returns
        -------
        The model's return value, unwrapped from the Protobuf response.
        """
        channel = self._ensure_channel()
        path = _build_path(model_name)
        model = self._get_model(model_name)
        request_cls, response_cls = build_message_classes(model)

        request_bytes = bytes(request_cls(**_encode_kwargs(kwargs, model)))

        stream = channel.request(path, Cardinality.UNARY_UNARY, None, None)
        async with stream as s:
            await s.send_message(request_bytes, end=True)
            response_bytes = await s.recv_message()

        response_msg = response_cls().parse(response_bytes)
        return _decode_result(response_msg.result, model)

    async def stream(self, model_name: str, **kwargs: Any) -> AsyncIterator[Any]:
        """Make a server-streaming call to a model.

        Parameters
        ----------
        model_name:
            The registered model name.
        **kwargs:
            Input fields matching the model function's parameters.

        Yields
        ------
        Each chunk's unwrapped result value.
        """
        channel = self._ensure_channel()
        path = _build_path(model_name)
        model = self._get_model(model_name)
        request_cls, response_cls = build_message_classes(model)

        request_bytes = bytes(request_cls(**_encode_kwargs(kwargs, model)))

        stream = channel.request(path, Cardinality.UNARY_STREAM, None, None)
        async with stream as s:
            await s.send_message(request_bytes, end=True)
            async for response_bytes in s:
                response_msg = response_cls().parse(response_bytes)
                yield _decode_result(response_msg.result, model)

    def _get_model(self, model_name: str) -> ModelInfo:
        if self._registry is None:
            raise RuntimeError(
                "BlazeClient requires a 'registry' to build Protobuf message classes. "
                "Pass registry=app.registry when constructing BlazeClient."
            )
        return self._registry.get(model_name)

    def _get_message_classes(self, model_name: str) -> tuple[type, type]:
        """Return ``(RequestClass, ResponseClass)`` for *model_name*.

        Requires that a ``registry`` was supplied at construction time.
        """
        return build_message_classes(self._get_model(model_name))


def _encode_kwargs(kwargs: dict[str, Any], model: ModelInfo) -> dict[str, Any]:
    """Convert annotated tensor inputs to their dynamic BetterProto messages."""
    encoded = dict(kwargs)
    for field_name, type_hint in model.input_types.items():
        if not isinstance(type_hint, _TensorType):
            continue
        if field_name not in kwargs:
            raise SerializationError(
                f"Missing tensor input '{field_name}' for model '{model.name}'"
            )
        try:
            tensor = python_to_proto(kwargs[field_name], type_hint)
        except SerializationError as exc:
            raise SerializationError(
                f"Invalid tensor input '{field_name}' for model '{model.name}': {exc}",
                dtype=exc.dtype,
            ) from exc
        encoded[field_name] = _TensorProtoMsg(
            shape=list(tensor.shape), dtype=tensor.dtype, data=tensor.data
        )
    return encoded


def _decode_result(result: Any, model: ModelInfo) -> Any:
    """Convert an annotated tensor response to numpy, preserving other values."""
    if not isinstance(model.output_type, _TensorType):
        return result

    tensor = TensorProto(
        shape=tuple(result.shape), dtype=result.dtype, data=result.data
    )
    try:
        return proto_to_python(tensor, model.output_type)
    except SerializationError as exc:
        raise SerializationError(
            f"Invalid tensor output from model '{model.name}': {exc}",
            dtype=exc.dtype,
        ) from exc


def _build_path(model_name: str) -> str:
    """Build the gRPC method path for a model name."""
    return f"/{SERVICE_NAME}/Predict{_sanitize_name(model_name)}"
