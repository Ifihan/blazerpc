"""Tests for ModelExecutor."""

from __future__ import annotations

import asyncio

import pytest

from blazerpc.app import BlazeApp
from blazerpc.exceptions import InferenceError
from blazerpc.runtime.executor import ModelExecutor


def _make_executor(
    func, *, name: str = "test", streaming: bool = False
) -> ModelExecutor:
    app = BlazeApp(enable_batching=False)
    app.model(name, streaming=streaming)(func)
    model = app.registry.get(name)
    return ModelExecutor(model)


# -- sync functions --


@pytest.mark.asyncio
async def test_sync_function() -> None:
    def predict(text: str) -> str:
        return text.upper()

    executor = _make_executor(predict)
    result = await executor.execute({"text": "hello"})
    assert result == "HELLO"


@pytest.mark.asyncio
async def test_sync_function_error() -> None:
    def predict(text: str) -> str:
        raise ValueError("bad input")

    executor = _make_executor(predict)
    with pytest.raises(InferenceError, match="inference failed"):
        await executor.execute({"text": "hello"})


# -- async functions --


@pytest.mark.asyncio
async def test_async_function() -> None:
    async def predict(text: str) -> str:
        await asyncio.sleep(0)
        return text.upper()

    executor = _make_executor(predict)
    result = await executor.execute({"text": "hello"})
    assert result == "HELLO"


@pytest.mark.asyncio
async def test_async_function_error() -> None:
    async def predict(text: str) -> str:
        raise RuntimeError("boom")

    executor = _make_executor(predict)
    with pytest.raises(InferenceError):
        await executor.execute({"text": "hello"})


# -- properties --


def test_name_and_version() -> None:
    def predict(x: int) -> int:
        return x

    executor = _make_executor(predict, name="mymodel")
    assert executor.name == "mymodel"
    assert executor.version == "1"
