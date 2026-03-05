"""Tests for the context injection system (Context, Depends, AppState)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from blazerpc import BlazeApp, Context, Depends
from blazerpc.codegen.servicer import _resolve_deps
from blazerpc.context import AppState
from blazerpc.types import extract_type_info


# ---------------------------------------------------------------------------
# AppState
# ---------------------------------------------------------------------------


def test_app_state_attribute_access() -> None:
    state = AppState()
    state.model = "my_model"
    state.db = "my_db"
    assert state.model == "my_model"
    assert state.db == "my_db"


def test_app_state_is_empty_by_default() -> None:
    state = AppState()
    assert vars(state) == {}


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def _make_mock_stream(metadata: Any = None, peer: Any = None) -> MagicMock:
    stream = MagicMock()
    stream.metadata = metadata
    stream.peer = peer
    return stream


def test_context_construction() -> None:
    stream = _make_mock_stream(metadata={"auth": "token"}, peer="127.0.0.1")
    state = AppState()
    state.x = 42

    ctx = Context(stream=stream, method="/svc/Method", app_state=state)

    assert ctx.metadata == {"auth": "token"}
    assert ctx.peer == "127.0.0.1"
    assert ctx.method == "/svc/Method"
    assert ctx.app_state.x == 42


def test_context_has_slots() -> None:
    assert hasattr(Context, "__slots__")


# ---------------------------------------------------------------------------
# Depends
# ---------------------------------------------------------------------------


def test_depends_stores_callable() -> None:
    def my_fn(ctx: Context) -> str:
        return "hello"

    dep = Depends(my_fn)
    assert dep.fn is my_fn


def test_depends_repr() -> None:
    def get_db(ctx: Context) -> str:
        return "db"

    dep = Depends(get_db)
    assert "get_db" in repr(dep)


# ---------------------------------------------------------------------------
# extract_type_info separates inputs / deps / context
# ---------------------------------------------------------------------------


def test_extract_type_info_with_depends() -> None:
    def get_db(ctx: Context) -> str:
        return "db"

    def handler(text: str, db: str = Depends(get_db)) -> str:
        return text

    info = extract_type_info(handler)
    assert "text" in info["inputs"]
    assert "db" not in info["inputs"]
    assert "db" in info["deps"]
    assert isinstance(info["deps"]["db"], Depends)


def test_extract_type_info_with_context() -> None:
    def handler(text: str, ctx: Context) -> str:
        return text

    info = extract_type_info(handler)
    assert "text" in info["inputs"]
    assert "ctx" not in info["inputs"]
    assert "ctx" in info["context_params"]


def test_extract_type_info_mixed() -> None:
    def get_model(ctx: Context) -> str:
        return "model"

    def handler(
        text: str,
        count: int,
        ctx: Context,
        model: str = Depends(get_model),
    ) -> str:
        return text

    info = extract_type_info(handler)
    assert set(info["inputs"].keys()) == {"text", "count"}
    assert info["context_params"] == ["ctx"]
    assert "model" in info["deps"]
    assert info["output"] is str


def test_extract_type_info_no_deps() -> None:
    """Existing functions without deps/context still work."""

    def handler(text: str) -> str:
        return text

    info = extract_type_info(handler)
    assert info["inputs"] == {"text": str}
    assert info["deps"] == {}
    assert info["context_params"] == []


# ---------------------------------------------------------------------------
# Model registration with deps
# ---------------------------------------------------------------------------


def test_model_with_context_registers() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("test")
    def handler(text: str, ctx: Context) -> str:
        return text

    model = app.registry.get("test")
    assert "text" in model.input_types
    assert "ctx" not in model.input_types
    assert "ctx" in model.context_params


def test_model_with_depends_registers() -> None:
    app = BlazeApp(enable_batching=False)

    def get_value(ctx: Context) -> int:
        return 42

    @app.model("test")
    def handler(text: str, val: int = Depends(get_value)) -> str:
        return text

    model = app.registry.get("test")
    assert "text" in model.input_types
    assert "val" not in model.input_types
    assert "val" in model.dep_params


def test_model_with_only_context_registers() -> None:
    """A model with only a Context param and no request fields is valid."""
    app = BlazeApp(enable_batching=False)

    @app.model("health")
    def handler(ctx: Context) -> str:
        return ctx.method

    model = app.registry.get("health")
    assert model.input_types == {}
    assert model.context_params == ["ctx"]


# ---------------------------------------------------------------------------
# _resolve_deps
# ---------------------------------------------------------------------------


async def test_resolve_deps_context_injection() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("test")
    def handler(text: str, ctx: Context) -> str:
        return text

    model = app.registry.get("test")
    stream = _make_mock_stream(metadata={"key": "val"})

    resolved = await _resolve_deps(model, stream, "/svc/Test", app.state)
    assert "ctx" in resolved
    assert isinstance(resolved["ctx"], Context)
    assert resolved["ctx"].method == "/svc/Test"


async def test_resolve_deps_sync_dependency() -> None:
    app = BlazeApp(enable_batching=False)
    app.state.value = 99

    def get_value(ctx: Context) -> int:
        return ctx.app_state.value

    @app.model("test")
    def handler(text: str, val: int = Depends(get_value)) -> str:
        return text

    model = app.registry.get("test")
    stream = _make_mock_stream()

    resolved = await _resolve_deps(model, stream, "/svc/Test", app.state)
    assert resolved["val"] == 99


async def test_resolve_deps_async_dependency() -> None:
    app = BlazeApp(enable_batching=False)

    async def async_dep(ctx: Context) -> str:
        return "async_result"

    @app.model("test")
    def handler(text: str, dep: str = Depends(async_dep)) -> str:
        return text

    model = app.registry.get("test")
    stream = _make_mock_stream()

    resolved = await _resolve_deps(model, stream, "/svc/Test", app.state)
    assert resolved["dep"] == "async_result"


# ---------------------------------------------------------------------------
# BlazeApp.state
# ---------------------------------------------------------------------------


def test_app_has_state() -> None:
    app = BlazeApp(enable_batching=False)
    assert isinstance(app.state, AppState)


def test_app_state_accepts_attributes() -> None:
    app = BlazeApp(enable_batching=False)
    app.state.model = "loaded_model"
    assert app.state.model == "loaded_model"
