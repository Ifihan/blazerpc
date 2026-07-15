"""gRPC server implementation.

Wraps :class:`grpclib.server.Server` with lifecycle management,
signal handling, and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import TYPE_CHECKING, Any, Sequence

from grpclib.encoding.base import CodecBase
from grpclib.server import Server

if TYPE_CHECKING:
    from blazerpc.server.middleware import Middleware

log = logging.getLogger("blazerpc.server")


class RawCodec(CodecBase):
    """Codec supporting raw inference and standard Protobuf services.

    BlazeRPC inference handlers declare no message type and manage their own
    serialization.  Standard services such as health and reflection declare
    generated Protobuf types and are encoded and decoded normally.
    """

    __content_subtype__ = "proto"

    def encode(self, message: Any, message_type: Any) -> bytes:
        if message_type is None:
            if not isinstance(message, bytes):
                raise TypeError("Raw messages must be bytes")
            return bytes(message)
        if not isinstance(message, message_type):
            raise TypeError(
                f"Message must be of type {message_type!r}, not {type(message)!r}"
            )
        return message.SerializeToString()

    def decode(self, data: bytes, message_type: Any) -> Any:
        if message_type is None:
            return data
        return message_type.FromString(data)


class GRPCServer:
    """Production-ready async gRPC server."""

    def __init__(
        self,
        handlers: Sequence[Any],
        *,
        middleware: Sequence[Middleware] | None = None,
        grace_period: float = 5.0,
    ) -> None:
        self._handlers = list(handlers)
        self._middleware = list(middleware or [])
        self._grace_period = grace_period
        self._server: Server | None = None
        self._shutdown_event: asyncio.Event = asyncio.Event()

    async def start(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
        *,
        handle_signals: bool = True,
    ) -> None:
        """Start serving and block until shutdown is requested."""
        self._shutdown_event.clear()
        loop = asyncio.get_running_loop()
        installed_signals: list[signal.Signals] = []
        failure: BaseException | None = None
        try:
            self._server = Server(self._handlers, codec=RawCodec())
            for mw in self._middleware:
                mw.attach(self._server)
            await self._server.start(host, port)
            log.info("Server listening on %s:%d", host, port)

            if handle_signals:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    try:
                        loop.add_signal_handler(sig, self._signal_shutdown)
                    except (NotImplementedError, RuntimeError):
                        break
                    installed_signals.append(sig)

            await self._shutdown_event.wait()
        except BaseException as exc:
            failure = exc
            raise
        finally:
            for sig in installed_signals:
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError):
                    pass
            try:
                await self.stop()
            except BaseException:
                if failure is None:
                    raise
                log.exception("Cleanup failed after gRPC server failure")

    async def stop(self) -> None:
        """Gracefully shut down the server."""
        self._shutdown_event.set()
        server = self._server
        if server is None:
            return
        self._server = None
        log.info("Shutting down (grace period %.1fs)…", self._grace_period)
        server.close()
        try:
            await asyncio.wait_for(
                server.wait_closed(),
                timeout=self._grace_period,
            )
        except asyncio.TimeoutError:
            log.warning("Grace period expired, forcing shutdown")

    def _signal_shutdown(self) -> None:
        log.info("Received shutdown signal")
        self._shutdown_event.set()
