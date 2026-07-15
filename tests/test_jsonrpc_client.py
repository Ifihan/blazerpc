"""Focused tests for JSON-RPC HTTP and SSE client behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("aiohttp", reason="aiohttp required for JSON-RPC tests")

from blazerpc.exceptions import BlazeRPCError
from blazerpc.jsonrpc_client import JsonRpcClient


class _ChunkedContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _Response:
    def __init__(
        self,
        *,
        body: dict[str, Any] | None = None,
        chunks: list[bytes] | None = None,
        status: int = 500,
        http_error: bool = True,
    ) -> None:
        self._body = body
        self.content = _ChunkedContent(chunks or [])
        self.status = status
        self._http_error = http_error
        self.status_checked = False

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        self.status_checked = True
        if self._http_error:
            raise RuntimeError("HTTP failure")

    async def json(self) -> dict[str, Any]:
        assert self._body is not None
        return self._body


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.closed = False
        self.posts: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.posts.append((url, kwargs))
        return self.response


async def test_predict_checks_http_status_after_parsing_non_error_body() -> None:
    response = _Response(body={"result": "must not be read"})
    client = JsonRpcClient("http://example.test/jsonrpc")
    client._session = _Session(response)  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="HTTP failure"):
        await client.predict("echo", text="hello")

    assert response.status_checked


async def test_stream_checks_http_status_before_reading_events() -> None:
    response = _Response(body={}, chunks=[b'data: "must not be read"\n\n'])
    client = JsonRpcClient("http://example.test/jsonrpc")
    client._session = _Session(response)  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="HTTP failure"):
        await anext(client.stream("echo"))

    assert response.status_checked


async def test_predict_preserves_jsonrpc_error_before_http_error() -> None:
    response = _Response(
        body={"error": {"code": -32602, "message": "bad params"}}, status=400
    )
    client = JsonRpcClient("http://example.test/jsonrpc")
    client._session = _Session(response)  # type: ignore[assignment]

    with pytest.raises(BlazeRPCError, match=r"-32602: bad params"):
        await client.predict("echo")

    assert not response.status_checked


async def test_stream_preserves_structured_error_before_http_error() -> None:
    response = _Response(
        body={"error": {"code": -32602, "message": "bad stream params"}},
        status=400,
    )
    client = JsonRpcClient("http://example.test/jsonrpc")
    client._session = _Session(response)  # type: ignore[assignment]

    with pytest.raises(BlazeRPCError, match=r"-32602: bad stream params"):
        await anext(client.stream("echo"))

    assert not response.status_checked


async def test_predict_retains_model_version_in_model_kwargs() -> None:
    response = _Response(body={"result": "ok"}, status=200, http_error=False)
    session = _Session(response)
    client = JsonRpcClient("http://example.test/jsonrpc")
    client._session = session  # type: ignore[assignment]

    assert await client.predict("echo", model_version="input") == "ok"

    assert session.posts[0][1]["json"]["method"] == "predict.echo"
    assert session.posts[0][1]["json"]["params"] == {"model_version": "input"}


async def test_predict_retains_model_name_in_model_kwargs() -> None:
    response = _Response(body={"result": "ok"}, status=200, http_error=False)
    session = _Session(response)
    client = JsonRpcClient("http://example.test/jsonrpc")
    client._session = session  # type: ignore[assignment]

    assert await client.predict("echo", model_name="input") == "ok"
    assert session.posts[0][1]["json"]["params"] == {"model_name": "input"}


async def test_predict_accepts_legacy_keyword_model_name() -> None:
    response = _Response(body={"result": "ok"}, status=200, http_error=False)
    session = _Session(response)
    client = JsonRpcClient("http://example.test/jsonrpc")
    client._session = session  # type: ignore[assignment]

    assert await client.predict(model_name="echo", text="input") == "ok"
    assert session.posts[0][1]["json"]["method"] == "predict.echo"
    assert session.posts[0][1]["json"]["params"] == {"text": "input"}


async def test_stream_retains_model_version_in_model_kwargs() -> None:
    response = _Response(
        chunks=[b"event: done\ndata: {}\n\n"], status=200, http_error=False
    )
    session = _Session(response)
    client = JsonRpcClient("http://example.test/jsonrpc")
    client._session = session  # type: ignore[assignment]

    assert [item async for item in client.stream("echo", model_version="input")] == []
    assert session.posts[0][0].endswith("/stream/echo")
    assert session.posts[0][1]["json"]["params"] == {"model_version": "input"}


async def test_explicit_versioned_client_methods_route_versions() -> None:
    unary_response = _Response(body={"result": "ok"}, status=200, http_error=False)
    unary_session = _Session(unary_response)
    client = JsonRpcClient("http://example.test/jsonrpc")
    client._session = unary_session  # type: ignore[assignment]

    assert (
        await client.predict_version(
            "echo", "2", text="hi", model_name="input-name", version="input-version"
        )
        == "ok"
    )
    assert unary_session.posts[0][1]["json"]["method"] == "predict.echo.v2"
    assert unary_session.posts[0][1]["json"]["params"] == {
        "text": "hi",
        "model_name": "input-name",
        "version": "input-version",
    }

    stream_response = _Response(
        chunks=[b'data: "ok"\n\nevent: done\ndata: {}\n\n'],
        status=200,
        http_error=False,
    )
    stream_session = _Session(stream_response)
    client._session = stream_session  # type: ignore[assignment]
    assert [
        item
        async for item in client.stream_version(
            "echo", "2", model_name="input-name", version="input-version"
        )
    ] == ["ok"]
    assert stream_session.posts[0][0].endswith("/stream/echo.v2")
    assert stream_session.posts[0][1]["json"]["params"] == {
        "model_name": "input-name",
        "version": "input-version",
    }


async def test_stream_parses_sse_frames_across_arbitrary_chunks() -> None:
    response = _Response(
        chunks=[
            b"da",
            b'ta: "one"\r',
            b'\n\ndata: "two',
            b'"\r\revent: do',
            b"ne\ndata: {}",
        ],
        status=200,
        http_error=False,
    )
    client = JsonRpcClient("http://example.test/jsonrpc")
    client._session = _Session(response)  # type: ignore[assignment]

    assert [item async for item in client.stream("echo")] == ["one", "two"]
