"""JSON-RPC example — serve models over HTTP with JSON-RPC 2.0.

Start the server::

    blaze serve examples.jsonrpc.app:app --transport jsonrpc --http-port 8080

Test with curl::

    curl -X POST http://localhost:8080/jsonrpc \\
      -H "Content-Type: application/json" \\
      -d '{"jsonrpc":"2.0","method":"predict.echo","params":{"text":"hello"},"id":1}'
"""

from blazerpc import BlazeApp, Context, Depends

app = BlazeApp(name="jsonrpc-example", enable_batching=False)


# -- Simple echo model -------------------------------------------------------


@app.model("echo")
def echo(text: str) -> str:
    """Return the input text prefixed with 'Echo:'."""
    return f"Echo: {text}"


# -- Math model ---------------------------------------------------------------


@app.model("add")
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


# -- Dependency injection example ---------------------------------------------

app.state.secret = "blazerpc-rocks"


def get_secret(ctx: Context) -> str:
    return ctx.app_state.secret


@app.model("greet")
def greet(name: str, secret: str = Depends(get_secret)) -> str:
    """Greet the user using an injected secret."""
    return f"Hello {name}! (secret={secret})"


# -- Streaming model ----------------------------------------------------------


@app.model("words", streaming=True)
async def words(sentence: str) -> str:
    """Yield each word in the sentence one at a time."""
    for word in sentence.split():
        yield word
