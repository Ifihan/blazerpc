"""Adaptive batching logic for efficient GPU utilization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class BatchItem:
    request: Any
    future: asyncio.Future[Any]


class Batcher:
    def __init__(self, max_size: int = 32, timeout_ms: float = 10.0) -> None:
        self.max_size = max_size
        self.timeout = timeout_ms / 1000
        self.queue: asyncio.Queue[BatchItem] = asyncio.Queue()

    async def submit(self, request: Any) -> Any:
        """Submit a request and wait for batched result."""
        future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        await self.queue.put(BatchItem(request, future))
        return await future

    async def _collect_batch(self) -> list[BatchItem]:
        """Collect items up to max_size or timeout."""
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

    async def process_loop(self, inference_fn: Callable[..., Any]) -> None:
        """Main batching loop."""
        while True:
            batch = await self._collect_batch()
            if batch:
                results = await inference_fn([item.request for item in batch])
                for item, result in zip(batch, results):
                    item.future.set_result(result)
