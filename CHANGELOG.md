# Changelog

All notable changes to BlazeRPC are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.1] - 2026-06-26

### Security

- **aiohttp** bumped to `>=3.11.0` (from `>=3.9.0`) in `jsonrpc` and `dev` extras —
  fixes ~20 CVEs including cross-origin credential leakage (DigestAuth, cookies,
  Proxy-Authorization header), websocket memory limit bypass, HTTP/1 pipelining DoS,
  CRLF injection in multipart headers, C parser null-byte injection, and TLS hostname
  override when reusing connections.
- **pytest** bumped to `>=8.1.0` (from `>=7.4.0`) — fixes vulnerable `tmpdir`
  handling (CVE-2024-6345).
- **pymdown-extensions** bumped to `>=10.14.3` (from `>=10.0`) — fixes path traversal
  bypass regression in `pymdownx.snippets` `restrict_base_path`.
- **urllib3** floor raised to `>=2.3.0` in `pytorch`, `tensorflow`, and `dev` extras —
  fixes sensitive header forwarding across origins in proxied redirects and
  decompression-bomb safeguard bypass in streaming API.
- **pillow** floor raised to `>=10.4.0` in `tensorflow` extra — fixes OOB write with
  invalid PSD tile extents, FITS GZIP decompression bomb, heap buffer overflow with
  nested list coordinates, integer overflow in font processing, and PDF trailer
  infinite loop DoS.
- **keras** floor raised to `>=3.3.0` in `tensorflow` extra — fixes untrusted
  deserialization vulnerability.
- **idna** floor raised to `>=3.7` in `jsonrpc` and `dev` extras — fixes bypass of the
  CVE-2024-3651 fix in `idna.encode()`.
- GitHub Actions: added `permissions: contents: read` to all jobs in `ci.yml` and
  `publish.yml` to enforce least-privilege token scope.

---

## [2.2.0] - 2026-03-25

### Added

- **JSON-RPC 2.0 transport** as an alternative to gRPC. The same `@app.model()`
  handlers serve over both transports without changes.
  - `JsonRpcDispatcher` — JSON-RPC 2.0 request dispatcher with batch support,
    error codes (parse error, method not found, invalid params, internal error),
    and full dependency injection (Context, Depends).
  - `JsonRpcServer` — aiohttp-based HTTP server with three routes:
    `POST /jsonrpc` (unary), `POST /jsonrpc/stream/{method}` (SSE streaming),
    and `GET /health`.
  - `JsonRpcClient` — async client with `predict()` and `stream()` methods.
    No registry parameter needed — JSON is self-describing. Auto-converts
    numpy arrays to/from base64-encoded tensor dicts.
  - `tensor_to_json()` / `tensor_from_json()` — base64 tensor serialization
    for the JSON transport.
- `app.serve_jsonrpc(host, port)` — start only the JSON-RPC HTTP server.
- `app.serve_both(host, grpc_port, http_port)` — start both gRPC and JSON-RPC
  servers simultaneously with shared batchers.
- `--transport` CLI option for `blaze serve` (values: `grpc`, `jsonrpc`, `both`).
- `--http-port` CLI option (default `8080`) for the JSON-RPC HTTP server.
- `Context.from_raw()` classmethod for protocol-agnostic context creation.
- Transport-agnostic middleware base classes: `TransportMiddleware`,
  `TransportLoggingMiddleware`, `TransportMetricsMiddleware`, `RequestInfo`,
  and `ResponseInfo`.
- `jsonrpc` optional dependency group (`aiohttp>=3.9.0`). Install with
  `pip install blazerpc[jsonrpc]`.
- JSON-RPC guide, updated architecture docs, and API reference entries for
  all new classes.

### Changed

- Extracted shared model invocation logic into `blazerpc.codegen.invoke`
  (`invoke_model`, `invoke_streaming_model`, `resolve_deps`) so both gRPC
  and JSON-RPC transports reuse the same execution path.
- `blazerpc.codegen.servicer` now delegates to `invoke` module instead of
  inlining invocation and dependency resolution.
- Batcher creation extracted to `BlazeApp._create_batchers()` for reuse
  across transports.

## [2.1.0] - 2026-03-12

### Added

- **Dependency injection system** with three building blocks:
  - `app.state` — attach shared resources (models, DB pools, config) at startup.
  - `Context` — per-request object providing gRPC metadata, peer info, method
    path, and access to `app.state`.
  - `Depends(fn)` — mark a handler parameter as an injected dependency resolved
    at request time. Supports both sync and async dependency functions.
- `Context` and `Depends` parameters are automatically excluded from generated
  Protobuf messages — clients never see them on the wire.
- Models using `Context` or `Depends` are automatically excluded from the
  adaptive batcher at startup with a log warning, since each request requires
  its own per-request context.
- Dependency injection guide with progressive tutorial, runtime flow diagram,
  auth pattern example, and decision guide for choosing between `app.state`,
  `Context`, and `Depends`.

### Changed

- `build_servicer()` accepts an optional `app_state` parameter for passing
  application state to the DI resolution layer.
- Batching documentation updated with an "Automatic exclusions" section
  covering streaming and DI model exclusions.
- API reference updated with cross-links to the dependency injection guide.

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
  and dtypes (e.g. `TensorInput[np.float32, tuple[Literal["batch"],
  Literal[224], Literal[224], Literal[3]]]`).
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
