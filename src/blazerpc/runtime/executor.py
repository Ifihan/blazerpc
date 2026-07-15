"""Model execution logic.

Wraps registered model functions with sync/async bridging, input
validation, and error handling.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, cast

from blazerpc.exceptions import InferenceError
from blazerpc.runtime.registry import ModelInfo


class ModelExecutor:
    """Executes a registered model function.

    Handles both sync and async callables transparently — sync functions
    are offloaded to a thread pool via :func:`asyncio.to_thread` so they
    never block the event loop.
    """

    def __init__(self, model: ModelInfo) -> None:
        self._model = model
        self._is_async = inspect.iscoroutinefunction(model.func)

    @property
    def name(self) -> str:
        return self._model.name

    @property
    def version(self) -> str:
        return self._model.version

    async def execute(self, kwargs: dict[str, Any]) -> Any:
        """Run inference with the given keyword arguments.

        Returns the raw result from the model function.
        """
        try:
            if self._is_async:
                async_func = cast(Callable[..., Awaitable[Any]], self._model.func)
                return await async_func(**kwargs)
            return await asyncio.to_thread(self._model.func, **kwargs)
        except Exception as exc:
            raise InferenceError(
                f"Model '{self.name}' inference failed: {exc}",
                model_name=self.name,
            ) from exc

    async def execute_batch(self, kwargs_list: list[dict[str, Any]]) -> list[Any]:
        """Run inference on a batch of inputs.

        The model function receives the full batch. The caller is
        responsible for stacking / unstacking individual items.
        """
        try:
            if self._is_async:
                async_func = cast(Callable[..., Awaitable[list[Any]]], self._model.func)
                return await async_func(kwargs_list)
            sync_func = cast(Callable[..., list[Any]], self._model.func)
            return await asyncio.to_thread(sync_func, kwargs_list)
        except Exception as exc:
            raise InferenceError(
                f"Model '{self.name}' batch inference failed: {exc}",
                model_name=self.name,
            ) from exc
