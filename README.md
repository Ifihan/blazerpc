# BlazeRPC

A lightweight, framework-agnostic gRPC library for serving machine learning models in Python. BlazeRPC gives you a FastAPI-like developer experience -- decorate a function, start the server, and you have a production-ready gRPC inference endpoint.

## Why BlazeRPC?

Serving ML models over gRPC typically involves writing `.proto` files by hand, compiling them into Python stubs, and wiring up boilerplate servicers. BlazeRPC removes all of that. You write a plain Python function, add a decorator, and the library generates the protobuf schema, servicer, and server for you.

**Key features:**

- **Decorator-based API** -- Register models with `@app.model("name")`, just like route handlers in a web framework.
- **Automatic proto generation** -- BlazeRPC inspects your function's type annotations and produces a valid `.proto` file. No hand-written schemas.
- **Adaptive batching** -- Individual requests are automatically grouped into batches for GPU-efficient inference. Configurable batch size and timeout.
- **Server-side streaming** -- Return tokens one at a time with `streaming=True`, ideal for LLM inference and real-time pipelines.
- **Health checks and reflection** -- Built-in gRPC health checking protocol and server reflection, compatible with `grpcurl`, `grpcui`, and Kubernetes probes.
- **Framework integrations** -- Optional helpers for PyTorch, TensorFlow, and ONNX Runtime that handle tensor conversion automatically.
- **Prometheus metrics** -- Request counts, latencies, and batch sizes are exported out of the box.

## Installation

```bash
pip install blazerpc
```

With framework-specific extras:

```bash
pip install blazerpc[pytorch]      # PyTorch tensor conversion helpers
pip install blazerpc[tensorflow]   # TensorFlow tensor conversion helpers
pip install blazerpc[onnx]         # ONNX Runtime model wrapper
pip install blazerpc[all]          # All optional integrations
```

## Quick start

### 1. Define your models

Create a file called `app.py`:

```python
from blazerpc import BlazeApp

app = BlazeApp()

@app.model("sentiment")
def predict_sentiment(text: list[str]) -> list[float]:
    # Replace with your real model inference
    return [0.95] * len(text)
```

BlazeRPC reads the type annotations on your function to generate the gRPC request and response messages. Supported types include `str`, `int`, `float`, `bool`, `list[float]`, `list[str]`, and tensor types via `TensorInput` / `TensorOutput`.

### 2. Start the server

```bash
blaze serve app:app
```

```
⚡ BlazeRPC server starting...
  ✓ Loaded model: sentiment v1
  ✓ Server listening on 0.0.0.0:50051
```

The server registers three services automatically:

| Service                                    | Purpose                |
| ------------------------------------------ | ---------------------- |
| `blazerpc.InferenceService`                | Your model RPCs        |
| `grpc.health.v1.Health`                    | Standard health checks |
| `grpc.reflection.v1alpha.ServerReflection` | Service discovery      |

### 3. Export the `.proto` file

```bash
blaze proto app:app --output-dir ./proto_out
```

This writes a `blaze_service.proto` file that you can compile with `protoc` or share with clients in any language. The generated proto looks like this:

```protobuf
syntax = "proto3";
package blazerpc;

message TensorProto {
  repeated int64 shape = 1;
  string dtype = 2;
  bytes data = 3;
}

message SentimentRequest {
  repeated string text = 1;
}

message SentimentResponse {
  repeated float result = 1;
}

service InferenceService {
  rpc PredictSentiment(SentimentRequest) returns (SentimentResponse);
}
```

## Streaming

To build a server-streaming endpoint (for example, returning tokens from an LLM), set `streaming=True`:

```python
@app.model("generate", streaming=True)
async def generate_tokens(prompt: str) -> str:
    tokens = run_my_llm(prompt)
    for token in tokens:
        yield token
```

Each `yield` sends a message to the client over the open gRPC stream. The client receives tokens as they are produced, without waiting for the full response.

## Adaptive batching

When `enable_batching=True` (the default), BlazeRPC collects individual requests and groups them into batches before calling your model function. This is essential for GPU workloads where batch inference is significantly faster than processing requests one at a time.

```python
app = BlazeApp(
    enable_batching=True,
    max_batch_size=32,       # Maximum requests per batch
    batch_timeout_ms=10.0,   # Maximum wait time before dispatching a partial batch
)
```

The batching layer handles:

- **Collecting requests** from concurrent clients into a single batch.
- **Dispatching partial batches** when the timeout expires, ensuring low latency even under light load.
- **Partial failure isolation** -- if one item in a batch fails, only that client receives an error. Other clients in the batch still get their results.

## Tensor types

For models that operate on NumPy arrays, use `TensorInput` and `TensorOutput` to declare the expected shape and dtype:

```python
import numpy as np
from blazerpc import BlazeApp, TensorInput, TensorOutput

app = BlazeApp()

@app.model("classify")
def classify(
    image: TensorInput[np.float32, "batch", 224, 224, 3],
) -> TensorOutput[np.float32, "batch", 1000]:
    # image is serialized as a TensorProto on the wire
    return model.predict(image)
```

The generated proto uses a `TensorProto` message with `shape`, `dtype`, and raw `bytes` fields for zero-copy serialization.

## Framework integrations

### PyTorch

```python
from blazerpc.contrib.pytorch import torch_model

@app.model("classifier")
@torch_model(device="cuda")
def classify(image):
    # `image` is automatically converted from np.ndarray to a torch.Tensor
    # on the specified device. The return value is converted back to np.ndarray.
    return model(image)
```

### TensorFlow

```python
from blazerpc.contrib.tensorflow import tf_model

@app.model("classifier")
@tf_model
def classify(image):
    return model(image)
```

### ONNX Runtime

```python
from blazerpc.contrib.onnx import ONNXModel

onnx_model = ONNXModel("model.onnx", providers=["CUDAExecutionProvider"])

@app.model("classifier")
def classify(image: np.ndarray) -> np.ndarray:
    return onnx_model.predict(image)[0]
```

## Middleware

BlazeRPC provides a middleware system built on grpclib's event hooks. Attach middleware to the underlying server to add logging, metrics, or custom request processing.

```python
from blazerpc.server.middleware import LoggingMiddleware, MetricsMiddleware

# These are attached inside app.serve() or manually on the grpclib Server:
# LoggingMiddleware().attach(grpclib_server)
# MetricsMiddleware().attach(grpclib_server)
```

**Built-in middleware:**

| Middleware            | Description                                                                                    |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| `LoggingMiddleware`   | Logs every RPC call with method name, peer address, and response status.                       |
| `MetricsMiddleware`   | Exports Prometheus metrics: `blazerpc_requests_total` and `blazerpc_request_duration_seconds`. |
| `ExceptionMiddleware` | Base class for custom exception-to-gRPC-status mapping.                                        |

To build your own middleware, subclass `Middleware` and implement `on_request` and `on_response`:

```python
from blazerpc.server.middleware import Middleware

class AuthMiddleware(Middleware):
    async def on_request(self, event):
        token = dict(event.metadata).get("authorization")
        if not token:
            raise GRPCError(Status.UNAUTHENTICATED, "Missing token")

    async def on_response(self, event):
        pass
```

## CLI reference

```bash
blaze serve <app_path> [OPTIONS]

  Start the BlazeRPC gRPC server.

  Arguments:
    app_path    App import path in module:attribute format (e.g. app:app)

  Options:
    --host TEXT       Host to bind to              [default: 0.0.0.0]
    --port INTEGER    Port to listen on            [default: 50051]
    --workers INTEGER Number of worker processes   [default: 1]
    --reload          Enable auto-reload           [default: False]
```

```bash
blaze proto <app_path> [OPTIONS]

  Export the generated .proto file.

  Arguments:
    app_path    App import path in module:attribute format (e.g. app:app)

  Options:
    --output-dir TEXT  Output directory for .proto files  [default: .]
```

## Project structure

```bash
src/blazerpc/
  __init__.py          # Public API: BlazeApp, TensorInput, TensorOutput, exceptions
  app.py               # BlazeApp class -- model registration and server lifecycle
  types.py             # TensorInput, TensorOutput, type introspection
  exceptions.py        # Exception hierarchy (BlazeRPCError and subclasses)
  decorators.py        # Reserved for future decorator extensions
  cli/
    main.py            # Typer CLI (blaze serve, blaze proto)
    serve.py           # App loading from import strings
    proto.py           # Proto file export
  codegen/
    proto.py           # .proto file generation from type annotations
    servicer.py        # Dynamic grpclib servicer generation
  runtime/
    registry.py        # Model registry (stores registered models and metadata)
    executor.py        # Model execution with sync/async bridging
    batcher.py         # Adaptive request batching
    serialization.py   # Tensor and scalar serialization
  server/
    grpc.py            # GRPCServer wrapper with signal handling and graceful shutdown
    health.py          # gRPC health checking protocol
    reflection.py      # gRPC server reflection
    middleware.py       # Logging, metrics, and extensible middleware base
  contrib/
    pytorch.py         # PyTorch <-> NumPy conversion and @torch_model decorator
    tensorflow.py      # TensorFlow <-> NumPy conversion and @tf_model decorator
    onnx.py            # ONNX Runtime session wrapper
```

## Development

```bash
# Clone the repository
git clone https://github.com/Ifihan/blazerpc.git
cd blazerpc

# Install dependencies (requires uv)
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/

# Type check
uv run mypy src/blazerpc/
```

## Contributing

We welcome contributions of all kinds -- bug fixes, new features, documentation improvements, and example applications. See the [Contributing Guide](CONTRIBUTING.md) for instructions on setting up a development environment, running tests, and submitting a pull request.

## License

MIT -- see [LICENSE](LICENSE) for details.
