"""Health check service (gRPC health checking protocol).

Wraps grpclib's built-in ``Health`` service so it can be registered
alongside the inference servicer.
"""

from __future__ import annotations

from typing import Any

from grpclib.health.service import Health


def build_health_service(
    servicers: list[Any] | None = None,
) -> Health:
    """Create a gRPC health service.

    When *servicers* is ``None`` the health service reports ``SERVING``
    unconditionally (no checks attached).  Pass servicer instances mapped
    to check lists for fine-grained health tracking.
    """
    if servicers:
        # Map each servicer → empty check list (always SERVING unless
        # custom checks are added later).
        checks = {s: [] for s in servicers}
        return Health(checks)
    return Health()
