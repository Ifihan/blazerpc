"""
Client for the streaming token generation service.

Connects to a running BlazeRPC streaming server and prints tokens as
they arrive using BlazeClient.

Prerequisites:
    1. Start the server:   uv run blaze serve examples.streaming.app:app
    2. Run this client:    uv run python examples/streaming/client.py
"""

import asyncio
import sys

from blazerpc import BlazeClient

from examples.streaming.app import app


async def main() -> None:
    async with BlazeClient("127.0.0.1", 50051, registry=app.registry) as client:
        print("Streaming tokens: ", end="", flush=True)
        async for token in client.stream("generate", prompt="BlazeRPC"):
            sys.stdout.write(token)
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    asyncio.run(main())
