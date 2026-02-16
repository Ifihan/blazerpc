"""
Client for the streaming token generation service.

Connects to a running BlazeRPC streaming server and prints tokens as
they arrive.

Prerequisites:
    1. Start the server:   uv run blaze serve examples.streaming.app:app
    2. Run this client:    uv run python examples/streaming/client.py
"""

import asyncio
import sys


async def main() -> None:
    print("Streaming client")
    print("=" * 40)
    print()
    print("This client requires generated gRPC stubs.")
    print("Export the .proto file first:")
    print("  uv run blaze proto examples.streaming.app:app --output-dir ./proto_out")
    print()
    print("Then compile the stubs with grpclib's protoc plugin and")
    print("use the generated stub to call the PredictGenerate RPC.")
    print()
    print("The streaming RPC yields tokens one at a time. A real")
    print("client would read from the response stream in a loop:")
    print()
    print("  async for token in stub.PredictGenerate(request):")
    print('      sys.stdout.write(token)')
    print()


if __name__ == "__main__":
    asyncio.run(main())
