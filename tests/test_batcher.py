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
    tasks = [asyncio.create_task(batcher.submit({"v": i})) for i in range(4)]
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

    tasks = [asyncio.create_task(batcher.submit({"i": i})) for i in range(6)]
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

    tasks = [asyncio.create_task(batcher.submit({"x": i})) for i in range(3)]
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


@pytest.mark.asyncio
async def test_queue_capacity_rejects_excess_requests() -> None:
    inference_started = asyncio.Event()
    release_inference = asyncio.Event()
    batcher = Batcher(max_size=1, timeout_ms=0, max_queue_size=1)

    async def inference_fn(batch: list) -> list:
        inference_started.set()
        await release_inference.wait()
        return [item["x"] for item in batch]

    await batcher.start(inference_fn)
    first = asyncio.create_task(batcher.submit({"x": 1}))
    await inference_started.wait()
    second = asyncio.create_task(batcher.submit({"x": 2}))
    await asyncio.sleep(0)

    assert batcher.queue.full()
    with pytest.raises(RuntimeError, match="queue is full"):
        await batcher.submit({"x": 3})

    release_inference.set()
    assert await asyncio.gather(first, second) == [1, 2]
    await batcher.stop()


@pytest.mark.asyncio
async def test_cancelled_queued_request_is_not_processed() -> None:
    inference_started = asyncio.Event()
    release_inference = asyncio.Event()
    processed: list[int] = []
    batcher = Batcher(max_size=1, timeout_ms=0, max_queue_size=1)

    async def inference_fn(batch: list) -> list:
        processed.extend(item["x"] for item in batch)
        inference_started.set()
        await release_inference.wait()
        return [item["x"] for item in batch]

    await batcher.start(inference_fn)
    first = asyncio.create_task(batcher.submit({"x": 1}))
    await inference_started.wait()
    queued = asyncio.create_task(batcher.submit({"x": 2}))
    await asyncio.sleep(0)

    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued

    release_inference.set()
    assert await first == 1
    await asyncio.sleep(0)
    assert processed == [1]
    await batcher.stop()


@pytest.mark.asyncio
async def test_cancelled_queued_request_releases_capacity_and_preserves_fifo() -> None:
    inference_started = asyncio.Event()
    release_inference = asyncio.Event()
    processed: list[int] = []
    batcher = Batcher(max_size=1, timeout_ms=0, max_queue_size=3)

    async def inference_fn(batch: list) -> list:
        value = batch[0]["x"]
        processed.append(value)
        if value == 1:
            inference_started.set()
            await release_inference.wait()
        return [value]

    await batcher.start(inference_fn)
    first = asyncio.create_task(batcher.submit({"x": 1}))
    await inference_started.wait()
    queued = [asyncio.create_task(batcher.submit({"x": value})) for value in (2, 3, 4)]
    await asyncio.sleep(0)
    assert batcher.queue.full()

    queued[1].cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued[1]
    assert not batcher.queue.full()

    replacement = asyncio.create_task(batcher.submit({"x": 5}))
    await asyncio.sleep(0)
    assert batcher.queue.full()
    release_inference.set()

    assert await asyncio.gather(first, queued[0], queued[2], replacement) == [
        1,
        2,
        4,
        5,
    ]
    assert processed == [1, 2, 4, 5]
    await batcher.stop()


@pytest.mark.asyncio
async def test_stop_cancels_in_flight_and_queued_requests() -> None:
    inference_started = asyncio.Event()
    inference_cancelled = asyncio.Event()
    batcher = Batcher(max_size=1, timeout_ms=0, max_queue_size=1)

    async def inference_fn(batch: list) -> list:
        inference_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            inference_cancelled.set()
            raise
        return batch

    await batcher.start(inference_fn)
    in_flight = asyncio.create_task(batcher.submit({"x": 1}))
    await inference_started.wait()
    queued = asyncio.create_task(batcher.submit({"x": 2}))
    await asyncio.sleep(0)

    await batcher.stop()

    results = await asyncio.gather(in_flight, queued, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert inference_cancelled.is_set()
    assert batcher.queue.empty()


@pytest.mark.asyncio
async def test_cancelled_stop_caller_waits_for_cleanup_then_propagates() -> None:
    inference_started = asyncio.Event()
    inference_cancelled = asyncio.Event()
    finish_cancellation = asyncio.Event()
    batcher = Batcher(max_size=1, timeout_ms=0)

    async def inference_fn(batch: list) -> list:
        inference_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            inference_cancelled.set()
            await finish_cancellation.wait()
            raise

    await batcher.start(inference_fn)
    submission = asyncio.create_task(batcher.submit({"x": 1}))
    await inference_started.wait()
    stopping = asyncio.create_task(batcher.stop())
    await inference_cancelled.wait()

    stopping.cancel()
    await asyncio.sleep(0)
    assert not stopping.done()
    finish_cancellation.set()

    with pytest.raises(asyncio.CancelledError):
        await stopping
    with pytest.raises(asyncio.CancelledError):
        await submission
    assert batcher._task is None
    assert not batcher._pending
    assert batcher.queue.empty()


@pytest.mark.asyncio
async def test_concurrent_stop_callers_complete_together() -> None:
    inference_started = asyncio.Event()
    inference_cancelled = asyncio.Event()
    finish_cancellation = asyncio.Event()
    batcher = Batcher(max_size=1, timeout_ms=0)

    async def inference_fn(batch: list) -> list:
        inference_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            inference_cancelled.set()
            await finish_cancellation.wait()
            raise

    await batcher.start(inference_fn)
    submission = asyncio.create_task(batcher.submit({"x": 1}))
    await inference_started.wait()
    stops = [asyncio.create_task(batcher.stop()) for _ in range(2)]
    await inference_cancelled.wait()
    await asyncio.sleep(0)
    assert not any(stop.done() for stop in stops)

    finish_cancellation.set()
    await asyncio.gather(*stops)
    with pytest.raises(asyncio.CancelledError):
        await submission
    assert batcher._task is None
    assert batcher.queue.empty()


@pytest.mark.asyncio
async def test_batcher_restart_waits_for_stop_cleanup() -> None:
    inference_started = asyncio.Event()
    inference_cancelled = asyncio.Event()
    finish_cancellation = asyncio.Event()
    batcher = Batcher(max_size=1, timeout_ms=0)

    async def first_fn(batch: list) -> list:
        inference_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            inference_cancelled.set()
            await finish_cancellation.wait()
            raise

    async def second_fn(batch: list) -> list:
        return [item["x"] * 2 for item in batch]

    await batcher.start(first_fn)
    submission = asyncio.create_task(batcher.submit({"x": 2}))
    await inference_started.wait()
    stopping = asyncio.create_task(batcher.stop())
    await inference_cancelled.wait()
    restarting = asyncio.create_task(batcher.start(second_fn))
    await asyncio.sleep(0)
    assert not restarting.done()

    finish_cancellation.set()
    await asyncio.gather(stopping, restarting)
    with pytest.raises(asyncio.CancelledError):
        await submission
    assert await batcher.submit({"x": 2}) == 4
    await batcher.stop()


@pytest.mark.asyncio
async def test_start_immediately_after_scheduling_stop_restarts_batcher() -> None:
    batcher = Batcher(max_size=1, timeout_ms=0)

    async def first_fn(batch: list) -> list:
        return [item["x"] for item in batch]

    async def second_fn(batch: list) -> list:
        return [item["x"] * 2 for item in batch]

    await batcher.start(first_fn)
    stopping = asyncio.create_task(batcher.stop())
    await batcher.start(second_fn)
    await stopping

    assert await batcher.submit({"x": 3}) == 6
    await batcher.stop()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_size": 0}, "max_size"),
        ({"timeout_ms": -1}, "timeout_ms"),
        ({"timeout_ms": float("inf")}, "timeout_ms"),
        ({"max_queue_size": 0}, "max_queue_size"),
    ],
)
def test_batcher_rejects_invalid_numeric_configuration(
    kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Batcher(**kwargs)
