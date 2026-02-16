"""
Simple sentiment analysis service.

This example demonstrates the most basic BlazeRPC usage: registering a
single model function and serving it over gRPC.

Run the server:
    uv run blaze serve examples.simple.app:app

Export the .proto file:
    uv run blaze proto examples.simple.app:app --output-dir ./proto_out
"""

from blazerpc import BlazeApp

app = BlazeApp(name="simple-demo", enable_batching=False)


@app.model("sentiment")
def predict_sentiment(text: list[str]) -> list[float]:
    """Return a sentiment score between 0 (negative) and 1 (positive).

    This is a stub implementation. In production you would load a real
    model (e.g. a fine-tuned transformer) and run inference here.
    """
    # Stub: return 0.85 for every input
    return [0.85] * len(text)


@app.model("echo")
def echo(text: str) -> str:
    """Echo the input back. Useful for connectivity checks."""
    return f"You said: {text}"
