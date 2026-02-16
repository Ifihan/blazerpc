"""CLI entry point for the blaze command."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="blaze",
    help="BlazeRPC - Lightning-fast gRPC for ML inference",
    add_completion=False,
)


@app.command()
def serve(
    app_path: str = typer.Argument(..., help="App import path (e.g. app:app)"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(50051, help="Port to listen on"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
    """Start the BlazeRPC gRPC server."""
    raise NotImplementedError


@app.command()
def proto(
    output_dir: str = typer.Option(".", help="Output directory for .proto files"),
) -> None:
    """Export generated .proto files."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
