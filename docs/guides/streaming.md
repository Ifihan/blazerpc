# Streaming

Server-side streaming lets you send responses incrementally over an open gRPC stream. This is essential for workloads where the client should see partial results as they are produced -- for example, an LLM generating tokens one at a time.

## Basic usage

Set `streaming=True` on the model decorator and write an async generator function:

```python
import asyncio
from blazerpc import BlazeApp

app = BlazeApp()

@app.model("generate", streaming=True)
async def generate_tokens(prompt: str) -> str:
    tokens = run_my_llm(prompt)
    for token in tokens:
        await asyncio.sleep(0)  # yield control to the event loop
        yield token
```

Each `yield` sends one message to the client over the open gRPC stream. The client receives tokens as they are produced, without waiting for the full response.

## How it works

When BlazeRPC sees `streaming=True` on a model, it:

1. Generates a `returns (stream Response)` RPC in the `.proto` file instead of a plain `returns (Response)`.
2. Creates a streaming handler that iterates over the generator and calls `stream.send_message()` for each yielded value.

Synchronous generators also work, but async generators are preferred because they do not block the event loop between yields.

## Generated proto

A streaming model produces a proto definition like this:

```protobuf
service InferenceService {
  rpc PredictGenerate(GenerateRequest) returns (stream GenerateResponse);
}
```

Clients consume the stream by reading messages in a loop until the server closes the stream.

## Client disconnection

If the client disconnects mid-stream, grpclib raises `asyncio.CancelledError` inside the generator. BlazeRPC catches and re-raises this so the generator is properly cleaned up. If you need to run cleanup logic on disconnection, use a `try/finally` block:

```python
@app.model("generate", streaming=True)
async def generate_tokens(prompt: str) -> str:
    try:
        async for token in model.generate(prompt):
            yield token
    finally:
        # cleanup resources if needed
        pass
```

## When to use streaming

Streaming is the right choice when:

- **Response latency matters more than throughput.** The client sees the first token in milliseconds instead of waiting for the full response.
- **Responses are unbounded or very large.** Streaming avoids buffering the entire response in memory.
- **The client drives cancellation.** A user can stop generation early without wasting server compute.

For workloads where the full response is small and available immediately, a standard unary RPC is simpler and has less overhead.

!!! note
    Streaming models are not compatible with adaptive batching. When `streaming=True`, requests are processed individually even if batching is enabled on the app.
