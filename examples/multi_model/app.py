"""
Multi-model service.

This example registers several models on a single BlazeRPC server.
Each model becomes its own RPC method under the
``blazerpc.InferenceService`` service, so clients can call whichever
model they need without running separate servers.

Generated RPCs:
    - PredictSentiment(SentimentRequest) -> SentimentResponse
    - PredictNer(NerRequest)             -> NerResponse
    - PredictSummarize(SummarizeRequest) -> SummarizeResponse

Run the server:
    uv run blaze serve examples.multi_model.app:app

Export all models' .proto:
    uv run blaze proto examples.multi_model.app:app --output-dir ./proto_out
"""

from blazerpc import BlazeApp

app = BlazeApp(name="multi-model-demo", enable_batching=False)


# ---- Model 1: Sentiment analysis ----


@app.model("sentiment")
def predict_sentiment(text: list[str]) -> list[float]:
    """Score each text from 0 (negative) to 1 (positive)."""
    return [0.92] * len(text)


# ---- Model 2: Named entity recognition ----


@app.model("ner")
def predict_ner(text: str) -> list[str]:
    """Extract named entities from the input text.

    Returns a list of entity strings. A real implementation would
    return structured spans with labels and offsets.
    """
    # Stub: pretend every input contains these entities.
    return ["BlazeRPC", "gRPC", "Python"]


# ---- Model 3: Summarization ----


@app.model("summarize")
def summarize(text: str, max_length: int) -> str:
    """Summarize the input text to at most ``max_length`` characters.

    A real implementation would use a sequence-to-sequence model.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."
