# Plan: Add JSON-RPC 2.0 Transport to BlazeRPC

## Context

BlazeRPC currently only supports gRPC as its transport. The user wants to add JSON-RPC 2.0 over HTTP as a second transport, since JSON-RPC is widely used and lowers the barrier to entry (no `.proto` files or gRPC tooling needed). The goal is to let the same `@app.model()` handlers serve over both gRPC and JSON-RPC without changes.

---

## Design Decisions

1. **JSON-RPC method naming**: `predict.<model_name>` for unary, `stream.<model_name>` for streaming
2. **Tensor encoding in JSON**: Base64-encoded bytes with shape/dtype metadata: `{"shape": [4], "dtype": "float", "data": "<base64>"}`
3. **Streaming over HTTP**: Server-Sent Events (SSE) — each yielded chunk is an SSE `data:` line
4. **HTTP framework**: `aiohttp` (async, lightweight, no heavy framework dependency)
5. **New optional dependency group**: `jsonrpc = ["aiohttp>=3.9.0"]`
6. **No breaking changes**: Existing gRPC path stays identical
7. **Shared batchers**: When running both transports, gRPC and JSON-RPC share the same batcher instances
8. **Context adaptation**: Add `Context.from_http()` classmethod so `Depends` functions work identically for both transports

---

## Implementation Phases

### Phase 1: Extract Shared Invocation Logic

**Goal**: Pull protocol-agnostic model invocation out of the gRPC servicer so both transports can reuse it.

#### New file: `src/blazerpc/codegen/invoke.py`

Extract from [servicer.py](src/blazerpc/codegen/servicer.py) into reusable functions:

```python
async def invoke_model(model, kwargs, batcher=None) -> Any
    # Handles: batcher submit vs direct call, sync/async bridging, InferenceError wrapping
    # Lines 130-142 of servicer.py

async def invoke_streaming_model(model, kwargs) -> AsyncIterator[Any]
    # Handles: async gen vs sync gen iteration, CancelledError, InferenceError
    # Lines 169-181 of servicer.py

async def resolve_deps(model, metadata, peer, method, app_state) -> dict[str, Any]
    # Protocol-agnostic version of _resolve_deps (lines 186-206 of servicer.py)
    # Builds Context internally, accepts raw metadata/peer instead of grpclib Stream
```

#### Modify: `src/blazerpc/context.py`

Add a `from_http` classmethod to `Context` (keep existing `__init__` unchanged):

```python
@classmethod
def from_http(cls, headers: dict[str, str], peer: str, method: str, app_state: AppState) -> Context:
    ctx = object.__new__(cls)
    ctx.metadata = headers
    ctx.peer = peer
    ctx.method = method
    ctx.app_state = app_state
    return ctx
```

#### Modify: `src/blazerpc/codegen/servicer.py`

- Import and delegate to `invoke.invoke_model` / `invoke.invoke_streaming_model` in `_make_unary_handler` and `_make_streaming_handler`
- Import and use `invoke.resolve_deps` instead of local `_resolve_deps`
- Keep all protobuf encode/decode logic here (gRPC-specific)

---

### Phase 2: JSON Tensor Serialization

#### New file: `src/blazerpc/runtime/json_serialization.py`

```python
def tensor_to_json(arr: np.ndarray) -> dict
    # Returns {"shape": list, "dtype": str, "data": base64_string}
    # Reuses DTYPE_MAP from types.py

def tensor_from_json(obj: dict) -> np.ndarray
    # Inverse: base64 decode + np.frombuffer + reshape
    # Reuses _PROTO_TO_NUMPY from serialization.py
```

---

### Phase 3: JSON-RPC Dispatcher

#### New file: `src/blazerpc/codegen/jsonrpc_handler.py`

The JSON-RPC equivalent of `servicer.py`. Handles JSON-RPC 2.0 envelope parsing and model dispatch.

```python
class JsonRpcDispatcher:
    def __init__(self, registry, batchers=None, app_state=None)

    async def handle(self, request_body: dict, peer: str, headers: dict) -> dict
        # Parse JSON-RPC envelope (id, method, params)
        # Route "predict.<name>" → unary handler
        # Route "stream.<name>" → error (use SSE endpoint)
        # Return JSON-RPC response envelope

    async def handle_streaming(self, method: str, params: dict, peer, headers) -> AsyncIterator[dict]
        # For SSE: yield JSON chunks from streaming model

# Internal helpers:
def _decode_json_request(params: dict, model: ModelInfo) -> dict[str, Any]
    # Convert JSON params to model kwargs, tensor fields via tensor_from_json

def _encode_json_response(result: Any, model: ModelInfo) -> Any
    # Convert model output to JSON-safe, numpy arrays via tensor_to_json
```

JSON-RPC error codes: `-32600` (invalid request), `-32601` (method not found), `-32602` (invalid params), `-32603` (internal error).

---

### Phase 4: HTTP Server

#### New file: `src/blazerpc/server/jsonrpc.py`

Parallel to [grpc.py](src/blazerpc/server/grpc.py). Uses `aiohttp`.

```python
class JsonRpcServer:
    def __init__(self, dispatcher, middleware=None, grace_period=5.0)

    async def start(self, host, port) -> None
        # Create aiohttp.web.Application
        # Routes:
        #   POST /jsonrpc       → JSON-RPC endpoint (supports batch requests)
        #   GET  /jsonrpc/stream/<method>  → SSE streaming endpoint
        #   GET  /health         → {"status": "ok"}
        # Signal handlers for graceful shutdown

    async def stop(self) -> None
```

Guard `import aiohttp` with try/except and raise `ConfigurationError` with install instructions if missing.

---

### Phase 5: Transport Middleware

#### Modify: `src/blazerpc/server/middleware.py`

Add at the bottom (existing classes unchanged):

```python
@dataclass
class RequestInfo:
    method: str
    peer: str
    headers: dict[str, str]
    transport: str  # "grpc" | "jsonrpc"

@dataclass
class ResponseInfo:
    status: int
    duration: float
    transport: str

class TransportMiddleware(ABC):
    async def on_request(self, info: RequestInfo) -> None: ...
    async def on_response(self, info: ResponseInfo) -> None: ...

class TransportLoggingMiddleware(TransportMiddleware): ...
class TransportMetricsMiddleware(TransportMiddleware): ...
```

---

### Phase 6: JSON-RPC Client

#### New file: `src/blazerpc/jsonrpc_client.py`

Mirrors `BlazeClient` API but uses HTTP + JSON-RPC.

```python
class JsonRpcClient:
    def __init__(self, url: str)  # e.g. "http://localhost:8080/jsonrpc"
    # No registry needed (JSON is self-describing)

    async def predict(self, model_name: str, **kwargs) -> Any
        # Send JSON-RPC request via aiohttp
        # Auto-convert np.ndarray kwargs via tensor_to_json
        # Auto-convert tensor dicts in response via tensor_from_json

    async def stream(self, model_name: str, **kwargs) -> AsyncIterator[Any]
        # Connect to SSE endpoint, yield chunks

    async def close(self) -> None
    async def __aenter__ / __aexit__  # Context manager
```

---

### Phase 7: App & CLI Integration

#### Modify: `src/blazerpc/app.py`

- Extract batcher creation into `_create_batchers()` private method (reused by all serve methods)
- Add `serve_jsonrpc(host, port)` method
- Add `serve_both(host, grpc_port, http_port)` method (starts both with shared batchers via `asyncio.gather`)
- Existing `serve()` unchanged

#### Modify: `src/blazerpc/cli/main.py`

Add to `serve` command:
- `--transport`: `grpc` | `jsonrpc` | `both` (default: `grpc`)
- `--http-port`: Port for JSON-RPC server (default: `8080`)

Update startup banner to show transport info.

#### Modify: `pyproject.toml`

- Add `jsonrpc = ["aiohttp>=3.9.0"]` to `[project.optional-dependencies]`
- Add `"jsonrpc"` to `all` extra

#### Modify: `src/blazerpc/__init__.py`

- Add `JsonRpcClient` to exports (lazy import to avoid requiring aiohttp)

---

### Phase 8: Tests

| File | Purpose |
|------|---------|
| `tests/test_invoke.py` | Unit tests for extracted `invoke_model`, `invoke_streaming_model`, `resolve_deps` |
| `tests/test_json_serialization.py` | Round-trip tests for `tensor_to_json` / `tensor_from_json` (various dtypes, edge cases) |
| `tests/test_jsonrpc_handler.py` | `JsonRpcDispatcher.handle()`: valid requests, method not found, invalid params, tensor I/O, batch requests |
| `tests/test_jsonrpc_server.py` | Server start/stop lifecycle, health endpoint |
| `tests/test_jsonrpc_client.py` | Client against in-process server |
| `tests/integration/test_jsonrpc_e2e.py` | Full E2E: echo, numeric, tensor, streaming (SSE), context/depends, batching |
| `tests/test_cli.py` (modify) | Test `--transport` and `--http-port` options |

---

### Phase 9: Example & Docs

- Add `examples/jsonrpc/` with `app.py` and `client.py`
- Update docs if needed (can be a follow-up)

---

## File Summary

| File | Action |
|------|--------|
| `src/blazerpc/codegen/invoke.py` | **NEW** — shared model invocation |
| `src/blazerpc/runtime/json_serialization.py` | **NEW** — JSON tensor encode/decode |
| `src/blazerpc/codegen/jsonrpc_handler.py` | **NEW** — JSON-RPC 2.0 dispatcher |
| `src/blazerpc/server/jsonrpc.py` | **NEW** — aiohttp HTTP server |
| `src/blazerpc/jsonrpc_client.py` | **NEW** — async JSON-RPC client |
| `src/blazerpc/context.py` | **MODIFY** — add `from_http` classmethod |
| `src/blazerpc/codegen/servicer.py` | **MODIFY** — delegate to `invoke.py` |
| `src/blazerpc/server/middleware.py` | **MODIFY** — add `TransportMiddleware` |
| `src/blazerpc/app.py` | **MODIFY** — add `serve_jsonrpc`, `serve_both`, extract `_create_batchers` |
| `src/blazerpc/cli/main.py` | **MODIFY** — add `--transport`, `--http-port` |
| `src/blazerpc/__init__.py` | **MODIFY** — export `JsonRpcClient` |
| `pyproject.toml` | **MODIFY** — add `jsonrpc` optional dep |
| 6 new test files + 1 modified | Tests |
| `examples/jsonrpc/` | **NEW** — example app + client |

---

## Verification

```bash
# Run existing tests (must all still pass — no regressions)
uv run pytest tests/ -v

# Run new tests
uv run pytest tests/test_invoke.py tests/test_json_serialization.py tests/test_jsonrpc_handler.py tests/test_jsonrpc_server.py tests/test_jsonrpc_client.py tests/integration/test_jsonrpc_e2e.py -v

# Lint & format
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Type check
uv run mypy src/blazerpc/

# Manual smoke test
uv sync --extra jsonrpc --extra dev
uv run blaze serve examples.jsonrpc.app:app --transport jsonrpc --http-port 8080
# In another terminal:
curl -X POST http://localhost:8080/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "predict.echo", "params": {"text": "hello"}, "id": 1}'
```
