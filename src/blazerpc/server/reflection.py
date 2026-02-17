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
    handlers: list[Any] | None = None,
) -> list[Any]:
    """Create gRPC reflection handlers.

    Parameters
    ----------
    handlers:
        gRPC service handler objects (e.g. the servicer returned by
        :func:`~blazerpc.codegen.servicer.build_servicer`).  When
        *None* an empty reflection service is returned.

    Returns
    -------
    list
        A list of grpclib-compatible handlers that can be passed to
        :class:`grpclib.server.Server`.
    """
    if handlers is None:
        handlers = []

    return ServerReflection.extend(handlers)
