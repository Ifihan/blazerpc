# Changelog

All notable changes to BlazeRPC are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-03-03

### Breaking Changes

- **Wire format changed from JSON to binary Protobuf.** The server now sends and
  receives real Protobuf-encoded messages using `betterproto` message classes built
  at runtime from model type annotations. Standard gRPC clients (Postman, `grpcurl`,
  generated stubs) now work without patches.
- **`BlazeClient` requires a `registry` parameter.** Pass `registry=app.registry`
  when constructing `BlazeClient` so it can build the correct Protobuf message
  classes for each model. The previous dict-based JSON API is removed.
- **Streaming model functions must declare a return type annotation** (`-> ChunkType`)
  so BlazeRPC can build the correct Protobuf response message class.

### Added

- `src/blazerpc/codegen/proto_types.py` — dynamic `betterproto.Message` class
  builder. Generates `(RequestClass, ResponseClass)` pairs at startup from
  `ModelInfo` without requiring a `protoc` code-generation step.

### Changed

- `RawCodec` is retained as the pass-through codec mechanism, but now conveys
  binary Protobuf bytes (encoded by betterproto) rather than JSON.
- `BlazeClient._ensure_channel()` no longer imports `RawCodec` from a separate
  path; both client and server use the same `RawCodec` from `server.grpc`.

## [1.1.0] - 2026-02-22

### Added

- `OTelMetricsMiddleware` for pushing RPC metrics via the OpenTelemetry Metrics
  API. Exports `blazerpc.rpc.count` (Counter) and `blazerpc.rpc.duration`
  (Histogram). Accepts an optional custom `Meter` instance for configuring
  exporters.
- Middleware configuration on `BlazeApp` and `GRPCServer` via a new `middleware`
  parameter. Middleware instances are automatically attached to the gRPC server
  on startup.
- `otel` optional dependency group (`opentelemetry-sdk`, `opentelemetry-exporter-otlp`)
  for push-based telemetry. Install with `pip install blazerpc[otel]`.

## [1.0.0] - 2026-02-16

First stable release of BlazeRPC.

### Added

#### Core

- `BlazeApp` class with `@app.model()` decorator for registering inference
  endpoints from plain Python functions.
- Automatic `.proto` file generation from function type annotations. Supported
  types: `str`, `int`, `float`, `bool`, `bytes`, `list[T]`, `TensorInput`,
  and `TensorOutput`.
- `TensorInput` and `TensorOutput` generic types for declaring tensor shapes
  and dtypes (e.g. `TensorInput[np.float32, "batch", 224, 224, 3]`).
- Exception hierarchy: `BlazeRPCError`, `ValidationError`,
  `ModelNotFoundError`, `SerializationError`, `InferenceError`, and
  `ConfigurationError`.

#### Server

- Async gRPC server built on grpclib with signal handling (SIGINT, SIGTERM)
  and configurable graceful shutdown.
- gRPC health checking protocol (`grpc.health.v1.Health`), registered
  automatically on every server.
- gRPC server reflection for service discovery with `grpcurl` and `grpcui`.
- Adaptive request batching with configurable `max_batch_size` and
  `batch_timeout_ms`. Supports partial failure isolation -- one bad item
  in a batch does not affect other clients.
- Server-side streaming for async generator model functions
  (`streaming=True`).

#### Code generation

- `ProtoGenerator` produces valid proto3 from a `ModelRegistry`, including
  `TensorProto`, per-model request/response messages, and an
  `InferenceService` with unary and server-streaming RPCs.
- Dynamic `InferenceServicer` implementing grpclib's `__mapping__()` protocol.
  Each registered model becomes an RPC handler with automatic request
  decoding, model execution, and response encoding.

#### CLI

- `blaze serve <app_path>` -- Start the gRPC server with a startup banner
  listing all loaded models.
- `blaze proto <app_path>` -- Export the generated `.proto` file to disk.
- uvloop is installed automatically on supported platforms for better
  async performance.

#### Middleware

- `Middleware` abstract base class built on grpclib's event system
  (`RecvRequest` / `SendTrailingMetadata`).
- `LoggingMiddleware` -- Logs every RPC call with method name, peer address,
  and response status.
- `MetricsMiddleware` -- Exports Prometheus metrics:
  `blazerpc_requests_total{method, status}` and
  `blazerpc_request_duration_seconds{method}`.
- `ExceptionMiddleware` -- Extensible base for custom exception-to-gRPC-status
  mapping.

#### Framework integrations

- **PyTorch**: `torch_to_numpy()`, `numpy_to_torch()`, and `@torch_model`
  decorator for automatic tensor conversion with device placement.
- **TensorFlow**: `tf_to_numpy()`, `numpy_to_tf()`, and `@tf_model`
  decorator for automatic tensor conversion.
- **ONNX Runtime**: `ONNXModel` wrapper class with `predict()` and
  `predict_dict()` methods for session management.

#### Serialization

- `TensorProto` dataclass for zero-copy tensor serialization via
  `np.ndarray.tobytes()` / `np.frombuffer()`.
- `python_to_proto()` and `proto_to_python()` for scalar and collection
  type conversion.

#### Testing

- 91 tests covering all modules: types, serialization, codegen, executor,
  server, CLI, batcher, health, middleware, and framework integrations.
- Integration tests verifying full register-serve-call flows with grpclib's
  in-process server.
