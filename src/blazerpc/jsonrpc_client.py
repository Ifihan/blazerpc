"""Async JSON-RPC 2.0 client for calling BlazeRPC model endpoints.

Usage::

    async with JsonRpcClient("http://localhost:8080/jsonrpc") as client:
        result = await client.predict("echo", text="hello")

        async for chunk in client.stream("generate", prompt="hi"):
            print(chunk)
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import numpy as np

try:
    import aiohttp
except ImportError as _exc:
    raise ImportError(
        "aiohttp is required for JsonRpcClient. "
        "Install it with:  pip install blazerpc[jsonrpc]"
    ) from _exc

from blazerpc.exceptions import BlazeRPCError
from blazerpc.runtime.json_serialization import (
    is_tensor_json,
    tensor_from_json,
    tensor_to_json,
)


class JsonRpcClient:
    """Async JSON-RPC client for BlazeRPC services.

    Unlike :class:`BlazeClient` (gRPC), this client does **not** require
    a ``registry`` parameter — JSON is self-describing.
    """

    def __init__(self, url: str) -> None:
        self._url = url.rstrip("/")
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

    async def predict(self, model_name: str, **kwargs: Any) -> Any:
        """Make a unary JSON-RPC prediction call.

        Numpy arrays in *kwargs* are auto-converted to tensor dicts.
        Tensor dicts in the response are auto-converted back to numpy arrays.
        """
        params = _prepare_params(kwargs)
        payload = {
            "jsonrpc": "2.0",
            "method": f"predict.{model_name}",
            "params": params,
            "id": self._next_id(),
        }

        session = await self._ensure_session()
        async with session.post(self._url, json=payload) as resp:
            body = await resp.json()

        if "error" in body:
            err = body["error"]
            raise BlazeRPCError(f"JSON-RPC error {err['code']}: {err['message']}")

        return _restore_result(body.get("result"))

    async def stream(self, model_name: str, **kwargs: Any) -> AsyncIterator[Any]:
        """Make a streaming call via the SSE endpoint.

        Yields each chunk's result value.
        """
        params = _prepare_params(kwargs)
        payload = {"params": params}
        url = f"{self._url}/stream/{model_name}"

        session = await self._ensure_session()
        async with session.post(url, json=payload) as resp:
            buffer = b""
            async for chunk in resp.content.iter_any():
                buffer += chunk
                while b"\n\n" in buffer:
                    event_data, buffer = buffer.split(b"\n\n", 1)
                    lines = event_data.decode().strip().split("\n")

                    event_type = ""
                    data_str = ""
                    for line in lines:
                        if line.startswith("event: "):
                            event_type = line[7:]
                        elif line.startswith("data: "):
                            data_str = line[6:]

                    if event_type == "done":
                        return
                    if event_type == "error":
                        raise BlazeRPCError(f"Stream error: {data_str}")
                    if data_str:
                        yield _restore_result(json.loads(data_str))

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "JsonRpcClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Auto-convert numpy arrays in kwargs to tensor JSON dicts."""
    result: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            result[key] = tensor_to_json(value)
        else:
            result[key] = value
    return result


def _restore_result(value: Any) -> Any:
    """Auto-convert tensor dicts in the response back to numpy arrays."""
    if is_tensor_json(value):
        return tensor_from_json(value)
    if isinstance(value, list):
        return [
            tensor_from_json(item) if is_tensor_json(item) else item for item in value
        ]
    return value
