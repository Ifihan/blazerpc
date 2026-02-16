"""Tests for health check service."""

from __future__ import annotations

import pytest

from blazerpc.app import BlazeApp
from blazerpc.codegen.servicer import build_servicer
from blazerpc.server.health import build_health_service


def test_build_health_service_no_servicers() -> None:
    """Health service can be created without servicers."""
    health = build_health_service()
    # Should have a __mapping__ method (grpclib protocol).
    assert hasattr(health, "__mapping__")


def test_build_health_service_with_servicers() -> None:
    """Health service accepts servicer instances for per-service checks."""
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    servicer = build_servicer(app.registry)
    health = build_health_service([servicer])
    assert hasattr(health, "__mapping__")


def test_health_service_has_rpc_methods() -> None:
    """The health service should expose Check and Watch RPCs."""
    health = build_health_service()
    mapping = health.__mapping__()
    paths = list(mapping.keys())
    # grpclib health service uses the standard grpc.health.v1 path.
    assert any("Health" in p for p in paths)
