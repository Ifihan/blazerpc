"""BlazeApp main class - orchestrates model registration, codegen, and server lifecycle."""

from __future__ import annotations

import asyncio
import logging
import math
import signal
from typing import Any, Callable, get_origin

import numpy as np

from blazerpc.codegen.servicer import build_servicer
from blazerpc.context import AppState
from blazerpc.runtime.batcher import Batcher
from blazerpc.runtime.registry import ModelInfo, ModelRegistry, batcher_key
from blazerpc.server.grpc import GRPCServer
from blazerpc.server.health import build_health_service
from blazerpc.server.middleware import Middleware
from blazerpc.server.reflection import build_reflection_service
from blazerpc.types import _TensorType

log = logging.getLogger("blazerpc")


def _make_batch_inference_fn(model: ModelInfo) -> Callable[..., Any]:
    """Create an inference function that combines tensor requests on axis zero."""
    is_async = asyncio.iscoroutinefunction(model.func)

    async def inference_fn(batch: list[dict[str, Any]]) -> list[Any]:
        batch_sizes: list[int] = []
        combined: dict[str, np.ndarray] = {}

        for request in batch:
            request_size: int | None = None
            for name in model.input_types:
                value = request[name]
                if not isinstance(value, np.ndarray) or value.ndim == 0:
                    raise ValueError(f"Batched input '{name}' must be a non-scalar array")
                if request_size is None:
                    request_size = value.shape[0]
                elif value.shape[0] != request_size:
                    raise ValueError(
                        "All tensor inputs in a request must have the same "
                        "leading dimension"
                    )
            if request_size is None:
                raise ValueError("A tensor batch must contain at least one input")
            batch_sizes.append(request_size)

        for name in model.input_types:
            values = [request[name] for request in batch]
            first = values[0]
            if any(
                value.ndim != first.ndim
                or value.shape[1:] != first.shape[1:]
                or value.dtype != first.dtype
                for value in values[1:]
            ):
                raise ValueError(
                    f"Batched input '{name}' must have matching rank, trailing "
                    "dimensions, and dtype"
                )
            combined[name] = np.concatenate(values, axis=0)

        if is_async:
            result = await model.func(**combined)
        else:
            result = await asyncio.to_thread(model.func, **combined)

        total_size = sum(batch_sizes)
        split_points = np.cumsum(batch_sizes[:-1]).tolist()
        if isinstance(result, np.ndarray):
            if result.ndim == 0 or result.shape[0] != total_size:
                raise ValueError(
                    "Batched tensor output leading dimension must equal the "
                    "combined input leading dimension"
                )
            return list(np.split(result, split_points, axis=0))
        if isinstance(result, list):
            if len(result) != total_size:
                raise ValueError(
                    "Batched list output length must equal the combined input "
                    "leading dimension"
                )
            boundaries = [0, *split_points, total_size]
            return [result[start:end] for start, end in zip(boundaries, boundaries[1:])]
        raise TypeError("A batched model must return a numpy array or list")

    return inference_fn


def _supports_adaptive_batching(model: ModelInfo) -> bool:
    """Return whether a model declares an unambiguous axis-zero batch contract."""
    if not model.input_types or not all(
        isinstance(input_type, _TensorType)
        and input_type.shape
        and input_type.shape[0] == "batch"
        for input_type in model.input_types.values()
    ):
        return False

    output_type = model.output_type
    return (
        isinstance(output_type, _TensorType)
        and bool(output_type.shape)
        and output_type.shape[0] == "batch"
    ) or get_origin(output_type) is list


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
        try:
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
                if not _supports_adaptive_batching(model_info):
                    continue
                batcher = Batcher(
                    self.max_batch_size, self.batch_timeout_ms, self.max_queue_size
                )
                await batcher.start(_make_batch_inference_fn(model_info))
                batchers[batcher_key(model_info.name, model_info.version)] = batcher
        except BaseException:
            for batcher in batchers.values():
                await batcher.stop()
            raise
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

        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()
        installed_signals: list[signal.Signals] = []
        server_tasks = [
            asyncio.create_task(
                grpc_server.start(host, grpc_port, handle_signals=False)
            ),
            asyncio.create_task(
                jsonrpc_server.start(host, http_port, handle_signals=False)
            ),
        ]
        shutdown_task = asyncio.create_task(shutdown_event.wait())
        all_tasks: list[asyncio.Task[Any]] = [*server_tasks, shutdown_task]

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, shutdown_event.set)
                except (NotImplementedError, RuntimeError):
                    break
                installed_signals.append(sig)

            done, _ = await asyncio.wait(
                all_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in server_tasks:
                if task in done:
                    task.result()
        finally:
            await asyncio.gather(
                grpc_server.stop(),
                jsonrpc_server.stop(),
                return_exceptions=True,
            )
            for task in all_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*all_tasks, return_exceptions=True)
            for sig in installed_signals:
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError):
                    pass
            for batcher in batchers.values():
                await batcher.stop()
