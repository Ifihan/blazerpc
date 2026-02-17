"""BlazeApp main class - orchestrates model registration, codegen, and server lifecycle."""

from __future__ import annotations

from typing import Callable

from blazerpc.codegen.servicer import build_servicer
from blazerpc.runtime.batcher import Batcher
from blazerpc.runtime.registry import ModelRegistry
from blazerpc.server.grpc import GRPCServer
from blazerpc.server.health import build_health_service
from blazerpc.server.reflection import build_reflection_service


class BlazeApp:
    def __init__(
        self,
        name: str = "blazerpc",
        enable_batching: bool = True,
        max_batch_size: int = 32,
        batch_timeout_ms: float = 10.0,
    ):
        self.name = name
        self.registry = ModelRegistry()
        self.batcher = (
            Batcher(max_batch_size, batch_timeout_ms) if enable_batching else None
        )

    def model(
        self,
        name: str,
        version: str = "1",
        streaming: bool = False,
    ) -> Callable:
        """Decorator to register a model endpoint."""

        def decorator(func: Callable) -> Callable:
            self.registry.register(name, version, func, streaming)
            return func

        return decorator

    async def serve(self, host: str = "0.0.0.0", port: int = 50051) -> None:
        """Start the gRPC server and block until shutdown."""
        servicer = build_servicer(self.registry, batcher=self.batcher)

        health = build_health_service([servicer])
        reflection_handlers = build_reflection_service([servicer])

        handlers = [servicer, health, *reflection_handlers]
        server = GRPCServer(handlers)
        await server.start(host, port)
