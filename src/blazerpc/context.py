"""Context injection and dependency resolution for BlazeRPC handlers.

Provides a FastAPI-like ``Depends()`` mechanism and per-request ``Context``
object so model handlers can access gRPC metadata, peer info, and shared
application state without boilerplate.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from grpclib.server import Stream


class AppState(types.SimpleNamespace):
    """Arbitrary app-level state container.

    Attach shared resources (loaded models, database pools, config) to
    an ``AppState`` instance and access them from dependency functions::

        app.state.model = load_my_model()
        app.state.db_pool = create_pool()
    """


class Context:
    """Per-request context object injected into handler parameters.

    Attributes
    ----------
    metadata
        gRPC invocation metadata (headers) sent by the client.
    peer
        Connection peer info (address, certificate).
    method
        Full gRPC method path, e.g.
        ``"/blazerpc.InferenceService/PredictIris"``.
    app_state
        Reference to :attr:`BlazeApp.state`.
    """

    __slots__ = ("metadata", "peer", "method", "app_state")

    def __init__(
        self,
        stream: Stream[Any, Any],
        method: str,
        app_state: AppState,
    ) -> None:
        self.metadata = stream.metadata
        self.peer = stream.peer
        self.method = method
        self.app_state = app_state


class Depends:
    """Mark a handler parameter as an injected dependency.

    The dependency function receives the per-request :class:`Context`
    and returns the value to inject.  Both sync and async functions are
    supported::

        def get_db(ctx: Context) -> Database:
            return ctx.app_state.db_pool

        @app.model("predict")
        async def predict(
            text: str,
            db: Database = Depends(get_db),
        ) -> str:
            ...
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    def __repr__(self) -> str:
        return f"Depends({self.fn.__qualname__})"
