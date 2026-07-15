"""gRPC server reflection support.

Wraps grpclib's built-in ``ServerReflection`` so that tools like
``grpcurl`` and ``grpcui`` can discover available services.
"""

from __future__ import annotations

import logging
from typing import Any

from google.protobuf.descriptor_pool import DescriptorPool
from grpclib.health.v1 import health_pb2
from grpclib.reflection.v1 import reflection_pb2
from grpclib.reflection.v1alpha import reflection_pb2 as reflection_v1alpha_pb2
from grpclib.reflection.service import ServerReflection

from blazerpc.codegen.servicer import InferenceServicer

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

    pool = DescriptorPool()
    for descriptor in (
        health_pb2.DESCRIPTOR,
        reflection_pb2.DESCRIPTOR,
        reflection_v1alpha_pb2.DESCRIPTOR,
    ):
        pool.AddSerializedFile(descriptor.serialized_pb)

    for handler in handlers:
        if isinstance(handler, InferenceServicer):
            pool.Add(handler.file_descriptor)

    return ServerReflection.extend(handlers, pool=pool)
