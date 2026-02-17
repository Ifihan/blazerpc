"""gRPC server implementation.

Wraps :class:`grpclib.server.Server` with lifecycle management,
signal handling, and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any, Sequence

from grpclib.encoding.base import CodecBase
from grpclib.server import Server

log = logging.getLogger("blazerpc.server")


class RawCodec(CodecBase):
    """Pass-through codec that skips protobuf serialization.

    BlazeRPC handlers encode/decode messages themselves, so the codec
    just forwards raw bytes without calling ``FromString``/``SerializeToString``.
    """

    __content_subtype__ = "proto"

    def encode(self, message: Any, message_type: Any) -> bytes:
        if isinstance(message, bytes):
            return message
        return message

    def decode(self, data: bytes, message_type: Any) -> Any:
        return data


class GRPCServer:
    """Production-ready async gRPC server."""

    def __init__(
        self,
        handlers: Sequence[Any],
        *,
        grace_period: float = 5.0,
    ) -> None:
        self._handlers = list(handlers)
        self._grace_period = grace_period
        self._server: Server | None = None
        self._shutdown_event: asyncio.Event = asyncio.Event()

    async def start(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
    ) -> None:
        """Start serving and block until shutdown is requested."""
        self._server = Server(self._handlers, codec=RawCodec())
        await self._server.start(host, port)

        log.info("Server listening on %s:%d", host, port)

        # Install signal handlers for graceful shutdown.
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_shutdown)

        # Block until a signal triggers the shutdown event.
        await self._shutdown_event.wait()
        await self.stop()

    async def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._server is None:
            return
        log.info("Shutting down (grace period %.1fs)…", self._grace_period)
        self._server.close()
        try:
            await asyncio.wait_for(
                self._server.wait_closed(),
                timeout=self._grace_period,
            )
        except asyncio.TimeoutError:
            log.warning("Grace period expired, forcing shutdown")
        self._server = None

    def _signal_shutdown(self) -> None:
        log.info("Received shutdown signal")
        self._shutdown_event.set()
