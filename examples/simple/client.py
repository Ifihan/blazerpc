"""
Client for the simple sentiment service.

This script connects to a running BlazeRPC server and calls the
``PredictSentiment`` and ``PredictEcho`` RPCs using BlazeClient.

Prerequisites:
    1. Start the server:   uv run blaze serve examples.simple.app:app
    2. Run this client:    uv run python examples/simple/client.py
"""

import asyncio

from blazerpc import BlazeClient

from examples.simple.app import app


async def main() -> None:
    async with BlazeClient("127.0.0.1", 50051, registry=app.registry) as client:
        # Unary call: echo
        reply = await client.predict("echo", text="BlazeRPC")
        print(f"echo → {reply}")

        # Unary call: sentiment (list[str] input, list[float] output)
        scores = await client.predict(
            "sentiment", text=["great product", "terrible experience"]
        )
        print(f"sentiment → {scores}")


if __name__ == "__main__":
    asyncio.run(main())
