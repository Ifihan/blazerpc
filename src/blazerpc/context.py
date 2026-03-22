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
        Invocation metadata (gRPC headers or HTTP headers) sent by the client.
    peer
        Connection peer info (address, certificate).
    method
        Full method path, e.g.
        ``"/blazerpc.InferenceService/PredictIris"`` (gRPC) or
        ``"predict.echo"`` (JSON-RPC).
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

    @classmethod
    def from_raw(
        cls,
        metadata: Any,
        peer: Any,
        method: str,
        app_state: AppState,
    ) -> "Context":
        """Create a Context without a grpclib Stream.

        Used by protocol-agnostic code paths (e.g. JSON-RPC) that already
        have the metadata and peer info extracted from the transport.
        """
        ctx = object.__new__(cls)
        ctx.metadata = metadata
        ctx.peer = peer
        ctx.method = method
        ctx.app_state = app_state
        return ctx


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
