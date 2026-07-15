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
        All gRPC service handler objects that will be installed (e.g. inference
        and health services). When *None* only reflection services are returned.

    Returns
    -------
    list
        The supplied handlers plus the v1 and v1alpha reflection handlers,
        ready to pass directly to :class:`grpclib.server.Server`.
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
