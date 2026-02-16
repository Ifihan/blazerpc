"""
Streaming token generation service.

This example shows how to build a server-streaming RPC, similar to how
a large language model returns tokens one at a time. The client
receives each token as it is produced rather than waiting for the
entire response.

Run the server:
    uv run blaze serve examples.streaming.app:app
"""

import asyncio

from blazerpc import BlazeApp

app = BlazeApp(name="streaming-demo", enable_batching=False)


@app.model("generate", streaming=True)
async def generate_tokens(prompt: str) -> str:
    """Simulate an LLM generating tokens one at a time.

    Each ``yield`` sends a single token to the client over the open
    gRPC stream. The client sees tokens arrive incrementally, which
    is essential for low-latency chat and text-generation interfaces.

    In production, replace the loop below with your model's
    auto-regressive decoding loop.
    """
    tokens = f"Hello! You asked about: {prompt}".split()

    for token in tokens:
        # Simulate the latency of generating one token.
        await asyncio.sleep(0.1)
        yield token + " "
