# JSON-RPC transport

BlazeRPC supports JSON-RPC 2.0 as an alternative transport to gRPC. The same `@app.model()` handlers serve over both protocols without changes — you choose the transport at startup.

JSON-RPC is useful when:

- Your clients are web browsers or lightweight HTTP tools (curl, Postman).
- You don't want to deal with Protobuf schemas and gRPC tooling.
- You need a human-readable wire format for debugging.
- Your infrastructure is HTTP-native (API gateways, load balancers, proxies).

!!! info "Same handlers, different wire format"
    JSON-RPC uses JSON over HTTP. gRPC uses binary Protobuf over HTTP/2. Both transports call the exact same model functions, including dependency injection (`Context`, `Depends`), streaming, and batching.

## Installation

JSON-RPC requires `aiohttp`. Install it via the `jsonrpc` extra:

```bash
uv add "blazerpc[jsonrpc]"
```

## Starting the server

### JSON-RPC only

```python
import asyncio
from blazerpc import BlazeApp

app = BlazeApp()

@app.model("echo")
def echo(text: str) -> str:
    return f"Echo: {text}"

asyncio.run(app.serve_jsonrpc(host="0.0.0.0", port=8080))
```

Or from the CLI:

```bash
blaze serve app:app --transport jsonrpc --http-port 8080
```

### Both transports simultaneously

```python
asyncio.run(app.serve_both(host="0.0.0.0", grpc_port=50051, http_port=8080))
```

Or from the CLI:

```bash
blaze serve app:app --transport both --port 50051 --http-port 8080
```

When running both transports, batchers are shared — a single batcher per model serves requests from both gRPC and JSON-RPC clients.

## Endpoints

The JSON-RPC server exposes three HTTP routes:

| Route | Method | Purpose |
| ----- | ------ | ------- |
| `/jsonrpc` | POST | JSON-RPC 2.0 endpoint (unary and batch) |
| `/jsonrpc/stream/{model}` | POST | Server-Sent Events (SSE) streaming |
| `/health` | GET | Health check (`{"status": "ok"}`) |

## Making requests

### Unary prediction

Send a standard JSON-RPC 2.0 request:

```bash
curl -X POST http://localhost:8080/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "predict.echo",
    "params": {"text": "hello"},
    "id": 1
  }'
```

Response:

```json
{
  "jsonrpc": "2.0",
  "result": "Echo: hello",
  "id": 1
}
```

The method name follows the pattern `predict.<model_name>`.

### Batch requests

Send an array of JSON-RPC requests in a single HTTP call:

```bash
curl -X POST http://localhost:8080/jsonrpc \
  -H "Content-Type: application/json" \
  -d '[
    {"jsonrpc": "2.0", "method": "predict.echo", "params": {"text": "a"}, "id": 1},
    {"jsonrpc": "2.0", "method": "predict.echo", "params": {"text": "b"}, "id": 2}
  ]'
```

Response:

```json
[
  {"jsonrpc": "2.0", "result": "Echo: a", "id": 1},
  {"jsonrpc": "2.0", "result": "Echo: b", "id": 2}
]
```

### Streaming via SSE

Streaming models use Server-Sent Events. Send a POST with params to the stream endpoint:

```bash
curl -X POST http://localhost:8080/jsonrpc/stream/gen \
  -H "Content-Type: application/json" \
  -d '{"params": {"prompt": "hello world"}}'
```

The server responds with an SSE event stream:

```
data: "hello"

data: "world"

event: done
data: {}
```

### Tensor data

NumPy arrays are serialized as JSON objects with base64-encoded data:

```json
{
  "shape": [3],
  "dtype": "float32",
  "data": "AACAPwAAAEAAAEBA"
}
```

The `JsonRpcClient` handles this conversion automatically. If you're building a custom client, encode tensors in this format.

## Python client

`JsonRpcClient` provides the same API as `BlazeClient` but over HTTP. Unlike the gRPC client, it does **not** require a `registry` parameter — JSON is self-describing.

```python
import asyncio
from blazerpc import JsonRpcClient

async def main():
    async with JsonRpcClient("http://localhost:8080/jsonrpc") as client:
        # Unary call
        result = await client.predict("echo", text="hello")
        print(result)  # "Echo: hello"

        # Streaming call
        async for chunk in client.stream("gen", prompt="a b c"):
            print(chunk)  # "a", "b", "c"

asyncio.run(main())
```

NumPy arrays are auto-converted in both directions:

```python
import numpy as np

arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
result = await client.predict("double", arr=arr)
# result is an np.ndarray: [2.0, 4.0, 6.0]
```

## Dependency injection

Context and Depends work identically over JSON-RPC. HTTP headers are passed as `Context.metadata`, and the client's IP address is available as `Context.peer`:

```python
from blazerpc import BlazeApp, Context, Depends

app = BlazeApp()
app.state.secret = "s3cret"

def require_auth(ctx: Context) -> str:
    token = ctx.metadata.get("Authorization", "")
    if token != f"Bearer {ctx.app_state.secret}":
        raise ValueError("Unauthorized")
    return token

@app.model("protected")
def protected(text: str, auth: str = Depends(require_auth)) -> str:
    return f"Authenticated: {text}"
```

!!! note "Context differences between transports"
    Over gRPC, `ctx.metadata` is the gRPC invocation metadata (a `MultiDict`). Over JSON-RPC, it is a plain `dict` of HTTP headers. `ctx.peer` is the client IP in both cases. `ctx.method` is the JSON-RPC method string (e.g. `"predict.echo"`) rather than a gRPC method path.

## Error handling

JSON-RPC errors follow the [JSON-RPC 2.0 spec](https://www.jsonrpc.org/specification#error_object):

| Code | Meaning |
| ---- | ------- |
| `-32700` | Parse error — invalid JSON |
| `-32600` | Invalid request — missing required fields |
| `-32601` | Method not found |
| `-32602` | Invalid params |
| `-32603` | Internal error — model raised an exception |

Example error response:

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32601,
    "message": "Method not found: predict.nonexistent"
  },
  "id": 1
}
```

## Middleware

The JSON-RPC server supports transport-agnostic middleware via `TransportMiddleware`. See the [middleware guide](middleware.md#transport-agnostic-middleware) for details.

```python
from blazerpc.server.middleware import TransportLoggingMiddleware
from blazerpc.server.jsonrpc import JsonRpcServer

server = JsonRpcServer(dispatcher, middleware=[TransportLoggingMiddleware()])
```

## Next steps

- [Getting started](../getting-started.md) — Installation and your first service.
- [Streaming](streaming.md) — Return tokens incrementally for LLM workloads.
- [Dependency injection](dependency-injection.md) — Inject shared resources and per-request context.
- [Middleware](middleware.md) — Logging, metrics, and custom middleware.
- [API reference](../api-reference.md#blazerpcjsonrpcclient) — Full API docs for `JsonRpcClient` and `JsonRpcServer`.
