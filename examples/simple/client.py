"""
Client for the simple sentiment service.

This script connects to a running BlazeRPC server and calls the
``PredictSentiment`` and ``PredictEcho`` RPCs.

Prerequisites:
    1. Start the server:   uv run blaze serve examples.simple.app:app
    2. Run this client:    uv run python examples/simple/client.py
"""

import asyncio

from grpclib.client import Channel
from grpclib.reflection.v1.reflection_grpc import ServerReflectionStub
from grpclib.reflection.v1.reflection_pb2 import ServerReflectionRequest


async def main() -> None:
    # Connect to the local server on the default port.
    channel = Channel(host="127.0.0.1", port=50051)

    try:
        print("Connected to BlazeRPC server at 127.0.0.1:50051")
        print()

        # ---- List available services via reflection ----
        try:
            stub = ServerReflectionStub(channel)
            print("Services discovered via reflection:")
            async with stub.ServerReflectionInfo.open() as stream:
                await stream.send_message(
                    ServerReflectionRequest(list_services="")
                )
                response = await stream.recv_message()
                for svc in response.list_services_response.service:
                    print(f"  - {svc.name}")
                print()
        except Exception:
            print("(Reflection not available -- skipping service listing)")
            print()

        print("To call RPCs you need generated stubs from the .proto file.")
        print("Export with:  uv run blaze proto examples.simple.app:app")
        print()

    finally:
        channel.close()


if __name__ == "__main__":
    asyncio.run(main())
