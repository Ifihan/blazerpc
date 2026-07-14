"""BlazeApp main class - orchestrates model registration, codegen, and server lifecycle."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Callable

from blazerpc.codegen.servicer import build_servicer
from blazerpc.context import AppState
from blazerpc.runtime.batcher import Batcher
from blazerpc.runtime.registry import ModelInfo, ModelRegistry
from blazerpc.server.grpc import GRPCServer
from blazerpc.server.health import build_health_service
from blazerpc.server.middleware import Middleware
from blazerpc.server.reflection import build_reflection_service

log = logging.getLogger("blazerpc")


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
        middleware: list[Middleware] | None = None,
        max_queue_size: int = 1024,
    ):
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if (
            isinstance(batch_timeout_ms, bool)
            or not isinstance(batch_timeout_ms, (int, float))
            or not math.isfinite(batch_timeout_ms)
            or batch_timeout_ms < 0
        ):
            raise ValueError("batch_timeout_ms must be a finite non-negative number")
        if (
            isinstance(max_queue_size, bool)
            or not isinstance(max_queue_size, int)
            or max_queue_size <= 0
        ):
            raise ValueError("max_queue_size must be a positive integer")

        self.name = name
        self.registry = ModelRegistry()
        self.state = AppState()
        self.enable_batching = enable_batching
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.max_queue_size = max_queue_size
        if middleware is None:
            self.middleware: list[Middleware] = []
        else:
            self.middleware = list(middleware)

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

    # ------------------------------------------------------------------
    # Batcher management (shared across transports)
    # ------------------------------------------------------------------

    async def _create_batchers(self) -> dict[str, Batcher]:
        """Create and start batchers for eligible models."""
        batchers: dict[str, Batcher] = {}
        if not self.enable_batching:
            return batchers
        for model_info in self.registry.list_models():
            if model_info.streaming:
                continue
            if model_info.dep_params or model_info.context_params:
                log.warning(
                    "Model '%s' uses Context/Depends — skipping batcher "
                    "(batching is not compatible with dependency injection)",
                    model_info.name,
                )
                continue
            batcher = Batcher(
                self.max_batch_size, self.batch_timeout_ms, self.max_queue_size
            )
            await batcher.start(_make_batch_inference_fn(model_info))
            batchers[model_info.name] = batcher
        return batchers

    # ------------------------------------------------------------------
    # gRPC transport
    # ------------------------------------------------------------------

    async def serve(self, host: str = "0.0.0.0", port: int = 50051) -> None:
        """Start the gRPC server and block until shutdown."""
        batchers = await self._create_batchers()

        servicer = build_servicer(
            self.registry, batchers=batchers, app_state=self.state
        )

        health = build_health_service([servicer])
        reflection_handlers = build_reflection_service([servicer])

        handlers = [servicer, health, *reflection_handlers]
        server = GRPCServer(handlers, middleware=self.middleware)

        try:
            await server.start(host, port)
        finally:
            for batcher in batchers.values():
                await batcher.stop()

    # ------------------------------------------------------------------
    # JSON-RPC transport
    # ------------------------------------------------------------------

    async def serve_jsonrpc(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start the JSON-RPC HTTP server and block until shutdown."""
        from blazerpc.codegen.jsonrpc_handler import JsonRpcDispatcher
        from blazerpc.server.jsonrpc import JsonRpcServer

        batchers = await self._create_batchers()
        dispatcher = JsonRpcDispatcher(
            self.registry, batchers=batchers, app_state=self.state
        )
        server = JsonRpcServer(dispatcher)

        try:
            await server.start(host, port)
        finally:
            for batcher in batchers.values():
                await batcher.stop()

    # ------------------------------------------------------------------
    # Both transports
    # ------------------------------------------------------------------

    async def serve_both(
        self,
        host: str = "0.0.0.0",
        grpc_port: int = 50051,
        http_port: int = 8080,
    ) -> None:
        """Start both gRPC and JSON-RPC servers with shared batchers."""
        from blazerpc.codegen.jsonrpc_handler import JsonRpcDispatcher
        from blazerpc.server.jsonrpc import JsonRpcServer

        batchers = await self._create_batchers()

        # gRPC
        servicer = build_servicer(
            self.registry, batchers=batchers, app_state=self.state
        )
        health = build_health_service([servicer])
        reflection_handlers = build_reflection_service([servicer])
        handlers = [servicer, health, *reflection_handlers]
        grpc_server = GRPCServer(handlers, middleware=self.middleware)

        # JSON-RPC
        dispatcher = JsonRpcDispatcher(
            self.registry, batchers=batchers, app_state=self.state
        )
        jsonrpc_server = JsonRpcServer(dispatcher)

        try:
            await asyncio.gather(
                grpc_server.start(host, grpc_port),
                jsonrpc_server.start(host, http_port),
            )
        finally:
            for batcher in batchers.values():
                await batcher.stop()
