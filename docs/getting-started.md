# Getting started

This guide walks you through installing BlazeRPC, defining a model, starting the server, and exporting a `.proto` file.

## Installation

```bash
pip install blazerpc
```

If you use a specific ML framework, install the corresponding extra:

```bash
pip install blazerpc[pytorch]      # PyTorch tensor conversion helpers
pip install blazerpc[tensorflow]   # TensorFlow tensor conversion helpers
pip install blazerpc[onnx]         # ONNX Runtime model wrapper
pip install blazerpc[all]          # All optional integrations
```

## Define a model

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

## Start the server

```bash
blaze serve app:app
```

The import string follows the `module:attribute` convention. BlazeRPC imports the module, looks up the attribute, and starts the gRPC server.

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

## Export the `.proto` file

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

## Multiple models

Register as many models as you need on the same app. Each model becomes its own RPC method:

```python
@app.model("sentiment")
def predict_sentiment(text: list[str]) -> list[float]:
    return [0.92] * len(text)

@app.model("ner")
def predict_ner(text: str) -> list[str]:
    return ["BlazeRPC", "gRPC", "Python"]

@app.model("summarize")
def summarize(text: str, max_length: int) -> str:
    return text[:max_length]
```

All three models are served under the same `InferenceService` and discovered through a single reflection endpoint.

## Next steps

- [Streaming](guides/streaming.md) -- Return tokens incrementally for LLM workloads.
- [Adaptive batching](guides/batching.md) -- Group requests into batches for GPU efficiency.
- [Framework integrations](guides/integrations.md) -- Use PyTorch, TensorFlow, or ONNX Runtime with automatic tensor conversion.
- [Configuration](configuration.md) -- Tune batch size, timeouts, and server options.
