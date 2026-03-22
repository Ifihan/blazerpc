# BlazeRPC

A lightweight, framework-agnostic RPC library for serving machine learning models in Python.

BlazeRPC gives you a FastAPI-like developer experience: decorate a function, start the server, and you have a production-ready inference endpoint over gRPC or JSON-RPC. No handwritten `.proto` files, no boilerplate servicers, no glue code.

## What it does

You write a plain Python function with type annotations. BlazeRPC turns it into a fully operational gRPC service.

```python
from blazerpc import BlazeApp

app = BlazeApp()

@app.model("sentiment")
def predict_sentiment(text: list[str]) -> list[float]:
    return model.predict(text)
```

```bash
blaze serve app:app
```

That single command:

1. Inspects your function's type annotations.
2. Generates a `.proto` schema with matching request/response messages.
3. Builds a gRPC servicer that routes requests to your function.
4. Starts an async gRPC server with health checks and reflection.

Want JSON-RPC instead? Swap the transport:

```bash
blaze serve app:app --transport jsonrpc
```

Or run both simultaneously:

```bash
blaze serve app:app --transport both
```

## Key features

- **Decorator-based API** -- Register models with `@app.model("name")`, just like route handlers in a web framework.
- **Automatic proto generation** -- BlazeRPC reads your type annotations and produces a valid `.proto` file. No hand-written schemas.
- **Adaptive batching** -- Individual requests are grouped into batches for GPU-efficient inference. Configurable batch size and timeout.
- **Server-side streaming** -- Return tokens one at a time with `streaming=True`, ideal for LLM inference.
- **Dual transport** -- Serve over gRPC (binary Protobuf) or JSON-RPC 2.0 (JSON over HTTP), or both at once. Same handlers, zero code changes.
- **Health checks and reflection** -- Built-in gRPC health checking protocol and server reflection, compatible with `grpcurl`, `grpcui`, and Kubernetes probes.
- **Framework integrations** -- Optional helpers for PyTorch, TensorFlow, and ONNX Runtime that handle tensor conversion automatically.
- **Prometheus metrics** -- Request counts and latencies are exported out of the box.

## Quick links

- [Getting started](getting-started.md) -- Installation and your first BlazeRPC service.
- [JSON-RPC transport](guides/jsonrpc.md) -- Serve models over JSON-RPC 2.0 with HTTP.
- [Architecture](architecture.md) -- How the internals fit together.
- [Configuration](configuration.md) -- All `BlazeApp` parameters, CLI flags, and tuning knobs.
- [API reference](api-reference.md) -- Every public class, function, and exception.
- [Contributing](contributing.md) -- How to set up a development environment and submit changes.
