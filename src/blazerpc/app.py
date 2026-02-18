"""BlazeApp main class - orchestrates model registration, codegen, and server lifecycle."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from blazerpc.codegen.servicer import build_servicer
from blazerpc.runtime.batcher import Batcher
from blazerpc.runtime.registry import ModelInfo, ModelRegistry
from blazerpc.server.grpc import GRPCServer
from blazerpc.server.health import build_health_service
from blazerpc.server.reflection import build_reflection_service


def _make_batch_inference_fn(model: ModelInfo) -> Callable[..., Any]:
    """Create an inference function that processes a batch by calling the model per-item."""
    is_async = asyncio.iscoroutinefunction(model.func)

    async def inference_fn(batch: list[dict[str, Any]]) -> list[Any]:
        results: list[Any] = []
        for kwargs in batch:
            if is_async:
                results.append(await model.func(**kwargs))
            else:
                results.append(await asyncio.to_thread(model.func, **kwargs))
        return results

    return inference_fn


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
        self.enable_batching = enable_batching
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms

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
        batchers: dict[str, Batcher] = {}
        if self.enable_batching:
            for model in self.registry.list_models():
                if model.streaming:
                    continue
                batcher = Batcher(self.max_batch_size, self.batch_timeout_ms)
                await batcher.start(_make_batch_inference_fn(model))
                batchers[model.name] = batcher

        servicer = build_servicer(self.registry, batchers=batchers)

        health = build_health_service([servicer])
        reflection_handlers = build_reflection_service([servicer])

        handlers = [servicer, health, *reflection_handlers]
        server = GRPCServer(handlers)

        try:
            await server.start(host, port)
        finally:
            for batcher in batchers.values():
                await batcher.stop()
