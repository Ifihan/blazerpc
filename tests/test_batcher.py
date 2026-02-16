"""Tests for adaptive batching."""

from __future__ import annotations

import asyncio

import pytest

from blazerpc.runtime.batcher import Batcher


@pytest.mark.asyncio
async def test_single_item_batch() -> None:
    """A single submitted request should be processed."""
    batcher = Batcher(max_size=4, timeout_ms=50)

    async def inference_fn(batch: list) -> list:
        return [item["x"] * 2 for item in batch]

    await batcher.start(inference_fn)

    result = await batcher.submit({"x": 5})
    assert result == 10

    await batcher.stop()


@pytest.mark.asyncio
async def test_multiple_items_batched() -> None:
    """Multiple concurrent requests should be collected into one batch."""
    batch_sizes: list[int] = []
    batcher = Batcher(max_size=8, timeout_ms=100)

    async def inference_fn(batch: list) -> list:
        batch_sizes.append(len(batch))
        return [item["v"] + 1 for item in batch]

    await batcher.start(inference_fn)

    # Submit 4 requests concurrently.
    tasks = [
        asyncio.create_task(batcher.submit({"v": i})) for i in range(4)
    ]
    results = await asyncio.gather(*tasks)
    assert sorted(results) == [1, 2, 3, 4]
    # They should have been batched together (or at most 2 batches).
    assert sum(batch_sizes) == 4

    await batcher.stop()


@pytest.mark.asyncio
async def test_batch_respects_max_size() -> None:
    """Batch should not exceed max_size."""
    batch_sizes: list[int] = []
    batcher = Batcher(max_size=3, timeout_ms=200)

    async def inference_fn(batch: list) -> list:
        batch_sizes.append(len(batch))
        return [1] * len(batch)

    await batcher.start(inference_fn)

    tasks = [
        asyncio.create_task(batcher.submit({"i": i})) for i in range(6)
    ]
    await asyncio.gather(*tasks)

    for bs in batch_sizes:
        assert bs <= 3

    await batcher.stop()


@pytest.mark.asyncio
async def test_batch_whole_failure() -> None:
    """If the inference function raises, all futures get the exception."""
    batcher = Batcher(max_size=4, timeout_ms=50)

    async def failing_fn(batch: list) -> list:
        raise ValueError("model exploded")

    await batcher.start(failing_fn)

    with pytest.raises(ValueError, match="model exploded"):
        await batcher.submit({"x": 1})

    await batcher.stop()


@pytest.mark.asyncio
async def test_batch_partial_failure() -> None:
    """If a result is an Exception instance, that future gets the error."""
    batcher = Batcher(max_size=4, timeout_ms=100)

    async def partial_fail_fn(batch: list) -> list:
        results = []
        for item in batch:
            if item.get("fail"):
                results.append(ValueError("bad item"))
            else:
                results.append(item["x"] * 2)
        return results

    await batcher.start(partial_fail_fn)

    ok_task = asyncio.create_task(batcher.submit({"x": 3, "fail": False}))
    fail_task = asyncio.create_task(batcher.submit({"x": 0, "fail": True}))

    ok_result = await ok_task
    assert ok_result == 6

    with pytest.raises(ValueError, match="bad item"):
        await fail_task

    await batcher.stop()


@pytest.mark.asyncio
async def test_batch_size_mismatch() -> None:
    """If inference returns wrong number of results, all futures error."""
    batcher = Batcher(max_size=4, timeout_ms=50)

    async def wrong_size_fn(batch: list) -> list:
        return [1]  # Always returns 1 result regardless of batch size

    await batcher.start(wrong_size_fn)

    tasks = [
        asyncio.create_task(batcher.submit({"x": i})) for i in range(3)
    ]
    # Give time for the batch to be processed.
    await asyncio.sleep(0.2)

    for t in tasks:
        with pytest.raises(RuntimeError, match="results"):
            await t

    await batcher.stop()


@pytest.mark.asyncio
async def test_batcher_start_stop_idempotent() -> None:
    """Starting twice and stopping twice should be safe."""
    batcher = Batcher(max_size=4, timeout_ms=50)

    async def noop_fn(batch: list) -> list:
        return [None] * len(batch)

    await batcher.start(noop_fn)
    await batcher.start(noop_fn)  # Second start is a no-op

    await batcher.stop()
    await batcher.stop()  # Second stop is a no-op
