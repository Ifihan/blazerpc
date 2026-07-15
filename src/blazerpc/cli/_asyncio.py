"""Event-loop runner shared by CLI entry points."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")


def run(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine with uvloop when it is available."""
    try:
        import uvloop
    except ImportError:
        return asyncio.run(coro)
    return uvloop.run(coro)
