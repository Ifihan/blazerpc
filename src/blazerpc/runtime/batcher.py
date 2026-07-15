"""Adaptive batching logic for efficient GPU utilization.

Collects individual inference requests into batches based on a
configurable maximum batch size and timeout.  Supports partial failure
handling — if the batch function raises, each pending future receives
the exception; if a per-item error list is returned, individual futures
are resolved or rejected accordingly.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

log = logging.getLogger("blazerpc.batcher")


@dataclass
class BatchItem:
    """A single request waiting to be batched."""

    request: Any
    future: asyncio.Future[Any] = field(repr=False)


class Batcher:
    """Adaptive request batcher.

    Parameters
    ----------
    max_size:
        Maximum number of items in one batch.
    timeout_ms:
        Maximum time (in milliseconds) to wait for a full batch
        before dispatching a partial one.
    max_queue_size:
        Maximum number of requests waiting for admission to a batch.
    """

    def __init__(
        self,
        max_size: int = 32,
        timeout_ms: float = 10.0,
        max_queue_size: int = 1024,
    ) -> None:
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, (int, float))
            or not math.isfinite(timeout_ms)
            or timeout_ms < 0
        ):
            raise ValueError("timeout_ms must be a finite non-negative number")
        if (
            isinstance(max_queue_size, bool)
            or not isinstance(max_queue_size, int)
            or max_queue_size <= 0
        ):
            raise ValueError("max_queue_size must be a positive integer")

        self.max_size = max_size
        self.timeout = timeout_ms / 1000
        self.queue: asyncio.Queue[BatchItem] = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._accepting = True
        self._task: asyncio.Task[None] | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._pending: dict[asyncio.Future[Any], BatchItem] = {}

    async def submit(self, request: Any) -> Any:
        """Submit a request and wait for the batched result."""
        if not self._accepting:
            raise RuntimeError("batcher is stopped")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        item = BatchItem(request, future)
        self._pending[future] = item
        try:
            try:
                self.queue.put_nowait(item)
            except asyncio.QueueFull as exc:
                item.request = None
                raise RuntimeError("batcher queue is full") from exc
            return await future
        except asyncio.CancelledError:
            future.cancel()
            self._remove_queued(item)
            item.request = None
            raise
        finally:
            self._pending.pop(future, None)

    async def start(self, inference_fn: Callable[..., Any]) -> None:
        """Start the background batching loop.

        Parameters
        ----------
        inference_fn:
            An async callable that receives a list of request dicts
            and returns a list of results (same length).
        """
        if self._running:
            return
        if self._stop_task is not None and not self._stop_task.done():
            await asyncio.shield(self._stop_task)
            if self._running:
                return
        self._stop_task = None
        self._accepting = True
        self._running = True
        self._task = asyncio.create_task(self._process_loop(inference_fn))

    def stop(self) -> Coroutine[Any, Any, None]:
        """Stop the batching loop and cancel all unresolved submissions."""
        self._accepting = False
        self._running = False
        stop_task = self._stop_task
        if stop_task is None:
            stop_task = asyncio.create_task(self._stop())
            self._stop_task = stop_task
        return self._wait_for_stop(stop_task)

    async def _wait_for_stop(self, stop_task: asyncio.Task[None]) -> None:
        try:
            await asyncio.shield(stop_task)
        except asyncio.CancelledError:
            # Keep cleanup alive, but do not turn cancellation into success.
            await asyncio.shield(stop_task)
            raise

    async def _stop(self) -> None:
        self._accepting = False
        self._running = False
        for future, item in list(self._pending.items()):
            item.request = None
            future.cancel()
        self._pending.clear()
        task = self._task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            if self._task is task:
                self._task = None
        while not self.queue.empty():
            item = self.queue.get_nowait()
            item.request = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _remove_queued(self, item: BatchItem) -> None:
        """Remove an item from the queue without changing live-item order."""
        retained: list[BatchItem] = []
        for _ in range(self.queue.qsize()):
            queued = self.queue.get_nowait()
            if queued is not item:
                retained.append(queued)
        for queued in retained:
            self.queue.put_nowait(queued)

    async def _collect_batch(self) -> list[BatchItem]:
        """Collect items up to *max_size* or until *timeout* expires."""
        batch: list[BatchItem] = []
        try:
            first = await self.queue.get()
            batch.append(first)
            deadline = asyncio.get_event_loop().time() + self.timeout
            while len(batch) < self.max_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                batch.append(item)
        except asyncio.TimeoutError:
            pass
        return batch

    async def _process_loop(self, inference_fn: Callable[..., Any]) -> None:
        """Main batching loop — runs as a background task."""
        while self._running:
            batch = await self._collect_batch()
            if not batch:
                continue

            batch = [item for item in batch if not item.future.done()]
            if not batch:
                continue

            log.debug("Processing batch of %d items", len(batch))

            try:
                results = await inference_fn([item.request for item in batch])
                if len(results) != len(batch):
                    exc = RuntimeError(
                        f"Batch function returned {len(results)} results "
                        f"for {len(batch)} inputs"
                    )
                    for item in batch:
                        if not item.future.done():
                            item.future.set_exception(exc)
                    continue

                for item, result in zip(batch, results):
                    if not item.future.done():
                        if isinstance(result, Exception):
                            item.future.set_exception(result)
                        else:
                            item.future.set_result(result)

            except Exception as exc:
                # Whole-batch failure: propagate to every pending future.
                log.error("Batch inference failed: %s", exc)
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)

    async def process_loop(self, inference_fn: Callable[..., Any]) -> None:
        """Run the batching loop (blocking).

        Kept for backwards compatibility — prefer :meth:`start` /
        :meth:`stop` for non-blocking lifecycle management.
        """
        self._accepting = True
        self._running = True
        await self._process_loop(inference_fn)
