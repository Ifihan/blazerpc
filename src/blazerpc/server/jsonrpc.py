"""JSON-RPC HTTP server implementation.

Wraps :mod:`aiohttp` with lifecycle management, signal handling,
and graceful shutdown — parallel to :class:`GRPCServer`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from typing import TYPE_CHECKING, Sequence

try:
    from aiohttp import web
except ImportError as _exc:
    raise ImportError(
        "aiohttp is required for the JSON-RPC transport. "
        "Install it with:  pip install blazerpc[jsonrpc]"
    ) from _exc

if TYPE_CHECKING:
    from blazerpc.codegen.jsonrpc_handler import JsonRpcDispatcher
    from blazerpc.server.middleware import TransportMiddleware

log = logging.getLogger("blazerpc.server.jsonrpc")


class JsonRpcServer:
    """HTTP server for JSON-RPC 2.0 requests."""

    def __init__(
        self,
        dispatcher: JsonRpcDispatcher,
        *,
        middleware: Sequence[TransportMiddleware] | None = None,
        grace_period: float = 5.0,
    ) -> None:
        self._dispatcher = dispatcher
        self._middleware = list(middleware or [])
        self._grace_period = grace_period
        self._runner: web.AppRunner | None = None
        self._shutdown_event: asyncio.Event = asyncio.Event()

    async def start(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start serving and block until shutdown is requested."""
        app = web.Application()
        app.router.add_post("/jsonrpc", self._handle_jsonrpc)
        app.router.add_post("/jsonrpc/stream/{method:.+}", self._handle_sse)
        app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()

        log.info("JSON-RPC server listening on %s:%d", host, port)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_shutdown)

        await self._shutdown_event.wait()
        await self.stop()

    async def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._runner is None:
            return
        log.info("Shutting down JSON-RPC server (grace %.1fs)…", self._grace_period)
        await self._runner.cleanup()
        self._runner = None

    def _signal_shutdown(self) -> None:
        log.info("Received shutdown signal")
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    async def _handle_jsonrpc(self, request: web.Request) -> web.Response:
        """Handle POST /jsonrpc — JSON-RPC 2.0 endpoint."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "Parse error"},
                    "id": None,
                },
                status=200,
            )

        peer = request.remote or ""
        headers = dict(request.headers)
        start = time.monotonic()

        # JSON-RPC batch: array of requests
        if isinstance(body, list):
            results = await self._dispatcher.handle_batch(
                body, peer=peer, headers=headers
            )
            duration = time.monotonic() - start
            await self._run_middleware_response("batch", duration)
            return web.json_response(results)

        # Single request
        result = await self._dispatcher.handle(body, peer=peer, headers=headers)
        duration = time.monotonic() - start
        method = body.get("method", "") if isinstance(body, dict) else ""
        await self._run_middleware_response(method, duration)
        return web.json_response(result)

    async def _handle_sse(self, request: web.Request) -> web.StreamResponse:
        """Handle POST /jsonrpc/stream/<method> — SSE streaming endpoint."""
        method = request.match_info["method"]

        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        params = body.get("params", {}) if isinstance(body, dict) else {}
        peer = request.remote or ""
        headers = dict(request.headers)

        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        await response.prepare(request)

        try:
            async for chunk in self._dispatcher.handle_streaming(
                f"stream.{method}", params, peer=peer, headers=headers
            ):
                data = json.dumps(chunk)
                await response.write(f"data: {data}\n\n".encode())
        except Exception:
            log.exception("Streaming error for method %s", method)
            error_data = json.dumps({"error": "Internal server error"})
            await response.write(f"event: error\ndata: {error_data}\n\n".encode())

        # Signal end of stream
        await response.write(b"event: done\ndata: {}\n\n")
        await response.write_eof()
        return response

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle GET /health."""
        return web.json_response({"status": "ok"})

    # ------------------------------------------------------------------
    # Middleware helpers
    # ------------------------------------------------------------------

    async def _run_middleware_response(self, method: str, duration: float) -> None:
        from blazerpc.server.middleware import ResponseInfo

        info = ResponseInfo(status=200, duration=duration, transport="jsonrpc")
        for mw in self._middleware:
            try:
                await mw.on_response(info)
            except Exception:
                log.exception("Middleware error in on_response")
