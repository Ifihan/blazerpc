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
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Sequence

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

from blazerpc.exceptions import ModelNotFoundError, SerializationError, ValidationError

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

    async def start(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        *,
        handle_signals: bool = True,
    ) -> None:
        """Start serving and block until shutdown is requested."""
        self._shutdown_event.clear()
        app = web.Application()
        app.router.add_post("/jsonrpc", self._handle_jsonrpc)
        app.router.add_post("/jsonrpc/stream/{method:.+}", self._handle_sse)
        app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(app)
        loop = asyncio.get_running_loop()
        installed_signals: list[signal.Signals] = []
        try:
            await self._runner.setup()
            site = web.TCPSite(self._runner, host, port)
            await site.start()

            log.info("JSON-RPC server listening on %s:%d", host, port)

            if handle_signals:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    try:
                        loop.add_signal_handler(sig, self._signal_shutdown)
                    except (NotImplementedError, RuntimeError):
                        break
                    installed_signals.append(sig)

            await self._shutdown_event.wait()
        finally:
            for sig in installed_signals:
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError):
                    pass
            await self.stop()

    async def stop(self) -> None:
        """Gracefully shut down the server."""
        self._shutdown_event.set()
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
        start = time.monotonic()
        peer = request.remote or ""
        headers = dict(request.headers)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            await self._run_middleware_response("", 200, time.monotonic() - start)
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "Parse error"},
                    "id": None,
                },
                status=200,
            )

        # Each batch element has its own middleware lifecycle so method-aware
        # authorization and metrics behave the same as independent requests.
        if isinstance(body, list):
            if not body:
                result, status = await self._dispatch_jsonrpc(
                    body, peer=peer, headers=headers
                )
                return web.json_response(result, status=status)
            processed = await asyncio.gather(
                *(
                    self._dispatch_jsonrpc(item, peer=peer, headers=headers)
                    for item in body
                )
            )
            results = [result for result, _ in processed if result is not None]
            if not results:
                return web.Response(status=204)
            return web.json_response(results)

        result, status = await self._dispatch_jsonrpc(body, peer=peer, headers=headers)
        if result is None:
            return web.Response(status=204)
        return web.json_response(result, status=status)

    async def _handle_sse(self, request: web.Request) -> web.StreamResponse:
        """Handle POST /jsonrpc/stream/<method> — SSE streaming endpoint."""
        method = request.match_info["method"]
        peer = request.remote or ""
        headers = dict(request.headers)
        lifecycle_method = f"stream.{method}"
        start = time.monotonic()

        denial = await self._run_middleware_request(lifecycle_method, peer, headers)
        if denial is not None:
            status, message = denial
            await self._run_middleware_response(
                lifecycle_method, status, time.monotonic() - start
            )
            return web.json_response({"error": message}, status=status)

        status = 500
        try:
            try:
                body = await request.json()
            except json.JSONDecodeError:
                status = 400
                return web.json_response({"error": "Invalid JSON body"}, status=400)

            if not isinstance(body, dict):
                status = 400
                return web.json_response(
                    {"error": "Request body must be an object"}, status=status
                )

            params = body.get("params", {})
            try:
                self._dispatcher.validate_streaming(lifecycle_method, params)
            except ModelNotFoundError as exc:
                status = 404
                return web.json_response({"error": str(exc)}, status=status)
            except (ValidationError, SerializationError) as exc:
                status = 400
                return web.json_response({"error": str(exc)}, status=status)

            response = web.StreamResponse(
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
            await response.prepare(request)

            status = 200
            try:
                async for chunk in self._dispatcher.handle_streaming(
                    lifecycle_method, params, peer=peer, headers=headers
                ):
                    data = json.dumps(chunk)
                    await response.write(f"data: {data}\n\n".encode())
            except Exception:
                status = 500
                log.exception("Streaming error for method %s", method)
                error_data = json.dumps({"error": "Internal server error"})
                await response.write(f"event: error\ndata: {error_data}\n\n".encode())

            await response.write(b"event: done\ndata: {}\n\n")
            await response.write_eof()
            return response
        finally:
            await self._run_middleware_response(
                lifecycle_method, status, time.monotonic() - start
            )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle GET /health."""
        method = "health"
        start = time.monotonic()
        denial = await self._run_middleware_request(
            method, request.remote or "", dict(request.headers)
        )
        if denial is not None:
            status, message = denial
            result = {"error": message}
        else:
            status = 200
            result = {"status": "ok"}
        await self._run_middleware_response(method, status, time.monotonic() - start)
        return web.json_response(result, status=status)

    # ------------------------------------------------------------------
    # Middleware helpers
    # ------------------------------------------------------------------

    async def _dispatch_jsonrpc(
        self,
        body: Any,
        *,
        peer: str,
        headers: dict[str, str],
    ) -> tuple[dict[str, Any] | None, int]:
        method = body.get("method", "") if isinstance(body, dict) else ""
        request_id = body.get("id") if isinstance(body, dict) else None
        start = time.monotonic()
        denial = await self._run_middleware_request(method, peer, headers)
        if denial is not None:
            status, message = denial
            result = (
                None
                if self._is_notification(body)
                else self._jsonrpc_error(request_id, -32000, message)
            )
        else:
            try:
                result = await self._dispatcher.handle(body, peer=peer, headers=headers)
                status = 204 if result is None else 200
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Unhandled JSON-RPC error for method %s", method)
                status = 500
                result = (
                    None
                    if self._is_notification(body)
                    else self._jsonrpc_error(
                        request_id, -32603, "Internal server error"
                    )
                )

        await self._run_middleware_response(method, status, time.monotonic() - start)
        return result, status

    @staticmethod
    def _is_notification(body: Any) -> bool:
        return (
            isinstance(body, dict)
            and body.get("jsonrpc") == "2.0"
            and isinstance(body.get("method"), str)
            and "id" not in body
        )

    async def _run_middleware_request(
        self, method: str, peer: str, headers: dict[str, str]
    ) -> tuple[int, str] | None:
        from blazerpc.server.middleware import RequestInfo

        info = RequestInfo(method, peer, headers, "jsonrpc")
        for mw in self._middleware:
            try:
                await mw.on_request(info)
            except asyncio.CancelledError:
                raise
            except web.HTTPException as exc:
                try:
                    message = HTTPStatus(exc.status).phrase
                except ValueError:
                    message = "Request denied"
                return exc.status, message
            except PermissionError:
                return 403, "Forbidden"
            except Exception:
                log.exception("Middleware error in on_request")
                return 500, "Internal server error"
        return None

    async def _run_middleware_response(
        self, method: str, status: int, duration: float
    ) -> None:
        from blazerpc.server.middleware import ResponseInfo

        info = ResponseInfo(
            method=method,
            status=status,
            duration=duration,
            transport="jsonrpc",
        )
        for mw in self._middleware:
            try:
                await mw.on_response(info)
            except Exception:
                log.exception("Middleware error in on_response")

    @staticmethod
    def _jsonrpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": request_id,
        }
