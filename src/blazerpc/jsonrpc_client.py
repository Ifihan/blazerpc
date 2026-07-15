"""Async JSON-RPC 2.0 client for calling BlazeRPC model endpoints.

Usage::

    async with JsonRpcClient("http://localhost:8080/jsonrpc") as client:
        result = await client.predict("echo", text="hello")

        async for chunk in client.stream("generate", prompt="hi"):
            print(chunk)
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

import numpy as np

try:
    import aiohttp
except ImportError as _exc:
    raise ImportError(
        "aiohttp is required for JsonRpcClient. "
        "Install it with:  pip install blazerpc[jsonrpc]"
    ) from _exc

from blazerpc.exceptions import BlazeRPCError, SerializationError
from blazerpc.codegen.jsonrpc_handler import jsonrpc_method
from blazerpc.runtime.json_serialization import (
    is_tensor_json,
    python_to_json,
    tensor_from_json,
    tensor_to_json,
)
from blazerpc.runtime.registry import ModelInfo, ModelRegistry
from blazerpc.types import _TensorType


class JsonRpcClient:
    """Async JSON-RPC client for BlazeRPC services.

    Unlike :class:`BlazeClient` (gRPC), this client does **not** require
    a ``registry`` parameter — JSON is self-describing.
    """

    def __init__(self, url: str, registry: ModelRegistry | None = None) -> None:
        self._url = url.rstrip("/")
        self._registry = registry
        self._session: aiohttp.ClientSession | None = None
        self._req_id = 0

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def predict(
        self, model_name: str, model_version: str = "1", **kwargs: Any
    ) -> Any:
        """Make a unary JSON-RPC prediction call.

        Numpy arrays in *kwargs* are auto-converted to tensor dicts.
        Tensor dicts in the response are auto-converted back to numpy arrays.
        """
        model = self._get_model(model_name, model_version)
        params = _prepare_params(kwargs, model)
        payload = {
            "jsonrpc": "2.0",
            "method": jsonrpc_method("predict", model_name, model_version),
            "params": params,
            "id": self._next_id(),
        }

        session = await self._ensure_session()
        async with session.post(self._url, json=payload) as resp:
            resp.raise_for_status()
            body = await resp.json()

        if "error" in body:
            err = body["error"]
            raise BlazeRPCError(f"JSON-RPC error {err['code']}: {err['message']}")

        return _restore_result(
            body.get("result"), model.output_type if model is not None else None
        )

    async def stream(
        self, model_name: str, model_version: str = "1", **kwargs: Any
    ) -> AsyncIterator[Any]:
        """Make a streaming call via the SSE endpoint.

        Yields each chunk's result value.
        """
        model = self._get_model(model_name, model_version)
        params = _prepare_params(kwargs, model)
        payload = {"params": params}
        method = jsonrpc_method("stream", model_name, model_version)
        url = f"{self._url}/stream/{method.removeprefix('stream.')}"

        session = await self._ensure_session()
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            buffer = b""
            async for chunk in resp.content.iter_any():
                buffer += chunk
                while separator := _sse_separator(buffer):
                    index, length = separator
                    event_data, buffer = buffer[:index], buffer[index + length :]
                    event_type, data_str = _parse_sse_event(event_data)

                    if event_type == "done":
                        return
                    if event_type == "error":
                        raise BlazeRPCError(f"Stream error: {data_str}")
                    if data_str:
                        yield _restore_result(
                            json.loads(data_str),
                            model.output_type if model is not None else None,
                        )

            if buffer:
                event_type, data_str = _parse_sse_event(buffer)
                if event_type == "error":
                    raise BlazeRPCError(f"Stream error: {data_str}")
                if event_type != "done" and data_str:
                    yield _restore_result(
                        json.loads(data_str),
                        model.output_type if model is not None else None,
                    )

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _get_model(self, name: str, version: str) -> ModelInfo | None:
        if self._registry is None:
            return None
        return self._registry.get(name, version)

    async def __aenter__(self) -> "JsonRpcClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_params(
    kwargs: dict[str, Any], model: ModelInfo | None = None
) -> dict[str, Any]:
    """Auto-convert numpy arrays in kwargs to tensor JSON dicts."""
    result: dict[str, Any] = {}
    for key, value in kwargs.items():
        type_hint = model.input_types.get(key) if model is not None else None
        if isinstance(type_hint, _TensorType):
            result[key] = python_to_json(value, type_hint)
        elif isinstance(value, np.ndarray):
            result[key] = tensor_to_json(value)
        else:
            result[key] = value
    if model is not None:
        for key, type_hint in model.input_types.items():
            if isinstance(type_hint, _TensorType) and key not in kwargs:
                raise SerializationError(
                    f"Missing tensor input '{key}' for model '{model.name}'"
                )
    return result


def _sse_separator(buffer: bytes) -> tuple[int, int] | None:
    """Return the earliest complete SSE event boundary in *buffer*."""
    match = re.search(
        rb"\r\n\r\n|\r\n\n|\r\n\r|\n\r\n|\n\n|\n\r|\r\r\n|\r\r",
        buffer,
    )
    if match is None:
        return None
    return match.start(), match.end() - match.start()


def _parse_sse_event(event_data: bytes) -> tuple[str, str]:
    """Extract the event type and joined data fields from one SSE event."""
    event_type = ""
    data_lines: list[str] = []
    for line in event_data.decode().splitlines():
        if line.startswith("event:"):
            event_type = line[6:].removeprefix(" ")
        elif line.startswith("data:"):
            data_lines.append(line[5:].removeprefix(" "))
    return event_type, "\n".join(data_lines)


def _restore_result(value: Any, type_hint: Any = None) -> Any:
    """Auto-convert tensor dicts in the response back to numpy arrays."""
    if isinstance(type_hint, _TensorType):
        if not isinstance(value, dict):
            raise SerializationError(
                f"Expected tensor dict for tensor output, got {type(value).__name__}"
            )
        return tensor_from_json(value, type_hint)
    if is_tensor_json(value):
        return tensor_from_json(value)
    if isinstance(value, list):
        return [
            tensor_from_json(item) if is_tensor_json(item) else item for item in value
        ]
    return value
