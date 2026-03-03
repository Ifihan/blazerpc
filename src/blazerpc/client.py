"""Async gRPC client for BlazeRPC services."""

from __future__ import annotations

from typing import Any, AsyncIterator

from grpclib.client import Channel
from grpclib.const import Cardinality

from blazerpc.codegen.proto import _sanitize_name
from blazerpc.codegen.proto_types import build_message_classes
from blazerpc.runtime.registry import ModelRegistry
from blazerpc.server.grpc import RawCodec

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
        request_cls, response_cls = self._get_message_classes(model_name)

        request_bytes = bytes(request_cls(**kwargs))

        stream = channel.request(path, Cardinality.UNARY_UNARY, None, None)
        async with stream as s:
            await s.send_message(request_bytes, end=True)
            response_bytes = await s.recv_message()

        response_msg = response_cls().parse(response_bytes)
        return response_msg.result  # type: ignore[union-attr]

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
        request_cls, response_cls = self._get_message_classes(model_name)

        request_bytes = bytes(request_cls(**kwargs))

        stream = channel.request(path, Cardinality.UNARY_STREAM, None, None)
        async with stream as s:
            await s.send_message(request_bytes, end=True)
            async for response_bytes in s:
                response_msg = response_cls().parse(response_bytes)
                yield response_msg.result  # type: ignore[union-attr]

    def _get_message_classes(self, model_name: str) -> tuple[type, type]:
        """Return ``(RequestClass, ResponseClass)`` for *model_name*.

        Requires that a ``registry`` was supplied at construction time.
        """
        if self._registry is None:
            raise RuntimeError(
                "BlazeClient requires a 'registry' to build Protobuf message classes. "
                "Pass registry=app.registry when constructing BlazeClient."
            )
        model = self._registry.get(model_name)
        return build_message_classes(model)


def _build_path(model_name: str) -> str:
    """Build the gRPC method path for a model name."""
    return f"/{SERVICE_NAME}/Predict{_sanitize_name(model_name)}"
