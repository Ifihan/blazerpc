"""Async gRPC client for BlazeRPC services."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from grpclib.client import Channel
from grpclib.const import Cardinality

from blazerpc.codegen.proto import _sanitize_name
from blazerpc.server.grpc import RawCodec

SERVICE_NAME = "blazerpc.InferenceService"


class BlazeClient:
    """Async gRPC client for calling BlazeRPC model endpoints.

    Usage::

        async with BlazeClient("127.0.0.1", 50051) as client:
            result = await client.predict("echo", text="hello")
            async for chunk in client.stream("tokens", prompt="hi"):
                print(chunk)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 50051) -> None:
        self._host = host
        self._port = port
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
            Input fields passed as JSON to the model function.

        Returns
        -------
        The model's return value (already unwrapped from the ``{"result": ...}`` envelope).
        """
        channel = self._ensure_channel()
        path = _build_path(model_name)
        request_bytes = json.dumps(kwargs).encode()

        stream = channel.request(path, Cardinality.UNARY_UNARY, None, None)
        async with stream as s:
            await s.send_message(request_bytes, end=True)
            response_bytes = await s.recv_message()

        data = json.loads(response_bytes)
        return data["result"]

    async def stream(self, model_name: str, **kwargs: Any) -> AsyncIterator[Any]:
        """Make a server-streaming call to a model.

        Parameters
        ----------
        model_name:
            The registered model name.
        **kwargs:
            Input fields passed as JSON to the model function.

        Yields
        ------
        Each chunk's unwrapped result value.
        """
        channel = self._ensure_channel()
        path = _build_path(model_name)
        request_bytes = json.dumps(kwargs).encode()

        stream = channel.request(path, Cardinality.UNARY_STREAM, None, None)
        async with stream as s:
            await s.send_message(request_bytes, end=True)
            async for response_bytes in s:
                data = json.loads(response_bytes)
                yield data["result"]


def _build_path(model_name: str) -> str:
    """Build the gRPC method path for a model name."""
    return f"/{SERVICE_NAME}/Predict{_sanitize_name(model_name)}"
