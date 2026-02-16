"""gRPC server reflection support.

Wraps grpclib's built-in ``ServerReflection`` so that tools like
``grpcurl`` and ``grpcui`` can discover available services.
"""

from __future__ import annotations

import logging
from typing import Any

from grpclib.reflection.service import ServerReflection

log = logging.getLogger("blazerpc.reflection")


def build_reflection_service(
    service_names: list[str] | None = None,
) -> list[Any]:
    """Create gRPC reflection handlers.

    Parameters
    ----------
    service_names:
        Fully-qualified service names to expose (e.g.
        ``["blazerpc.InferenceService"]``).  When *None* an empty
        reflection service is returned.

    Returns
    -------
    list
        A list of grpclib-compatible handlers that can be passed to
        :class:`grpclib.server.Server`.
    """
    if service_names is None:
        service_names = []

    return ServerReflection.extend(service_names)
