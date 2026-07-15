"""Focused tests for JSON-RPC HTTP and SSE client behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("aiohttp", reason="aiohttp required for JSON-RPC tests")

from blazerpc.jsonrpc_client import JsonRpcClient


class _ChunkedContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _Response:
    def __init__(
        self, *, body: dict[str, Any] | None = None, chunks: list[bytes] | None = None
    ) -> None:
        self._body = body
        self.content = _ChunkedContent(chunks or [])
        self.status_checked = False

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        self.status_checked = True
        raise RuntimeError("HTTP failure")

    async def json(self) -> dict[str, Any]:
        assert self.status_checked
        assert self._body is not None
        return self._body


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.closed = False

    def post(self, *args: Any, **kwargs: Any) -> _Response:
        return self.response


async def test_predict_checks_http_status_before_parsing_body() -> None:
    response = _Response(body={"result": "must not be read"})
    client = JsonRpcClient("http://example.test/jsonrpc")
    client._session = _Session(response)  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="HTTP failure"):
        await client.predict("echo", text="hello")

    assert response.status_checked


async def test_stream_checks_http_status_before_reading_events() -> None:
    response = _Response(chunks=[b'data: "must not be read"\n\n'])
    client = JsonRpcClient("http://example.test/jsonrpc")
    client._session = _Session(response)  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="HTTP failure"):
        await anext(client.stream("echo"))

    assert response.status_checked


async def test_stream_parses_sse_frames_across_arbitrary_chunks() -> None:
    response = _Response(
        chunks=[
            b"da",
            b'ta: "one"\r',
            b'\n\ndata: "two',
            b'"\r\revent: do',
            b"ne\ndata: {}",
        ]
    )
    response.raise_for_status = lambda: None  # type: ignore[method-assign]
    client = JsonRpcClient("http://example.test/jsonrpc")
    client._session = _Session(response)  # type: ignore[assignment]

    assert [item async for item in client.stream("echo")] == ["one", "two"]
