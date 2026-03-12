"""
Client for the dependency injection example.

Calls the DI-powered handlers, demonstrating how Context and Depends
work transparently from the client's perspective.

Prerequisites:
    1. uv add scikit-learn
    2. Start the server:   uv run blaze serve examples.dependency_injection.app:app
    3. Run this client:    uv run python examples/dependency_injection/client.py
"""

import asyncio

import numpy as np

from blazerpc import BlazeClient

from examples.dependency_injection.app import app


async def main() -> None:
    async with BlazeClient("127.0.0.1", 50051, registry=app.registry) as client:
        # 1. Classify using the injected sklearn model
        sample = np.array([5.1, 3.5, 1.4, 0.2], dtype=np.float32)
        probs = await client.predict("classify", features=sample)
        print(f"classify → probabilities: {probs}")

        # 2. Get a human-readable label
        label = await client.predict("label", features=sample)
        print(f"label    → {label}")

        # 3. Call whoami (uses Context and auth dependency)
        info = await client.predict("whoami")
        print(f"whoami   → {info}")


if __name__ == "__main__":
    asyncio.run(main())
