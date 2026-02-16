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
from dataclasses import dataclass, field
from typing import Any, Callable

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
    """

    def __init__(self, max_size: int = 32, timeout_ms: float = 10.0) -> None:
        self.max_size = max_size
        self.timeout = timeout_ms / 1000
        self.queue: asyncio.Queue[BatchItem] = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def submit(self, request: Any) -> Any:
        """Submit a request and wait for the batched result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        await self.queue.put(BatchItem(request, future))
        return await future

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
        self._running = True
        self._task = asyncio.create_task(self._process_loop(inference_fn))

    async def stop(self) -> None:
        """Stop the batching loop and drain remaining items."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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
        self._running = True
        await self._process_loop(inference_fn)
