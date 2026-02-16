"""CLI entry point for the blaze command."""

from __future__ import annotations

import asyncio
import logging

import typer

from blazerpc.cli.proto import export_proto
from blazerpc.cli.serve import load_app

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    blaze_app = load_app(app_path)

    # Install uvloop when available for better performance.
    try:
        import uvloop  # type: ignore[import-untyped]

        uvloop.install()
    except ImportError:
        pass

    # Print startup banner.
    models = blaze_app.registry.list_models()
    typer.echo("")
    typer.echo("⚡ BlazeRPC server starting...")
    for m in models:
        tag = " (stream)" if m.streaming else ""
        typer.echo(f"  ✓ Loaded model: {m.name} v{m.version}{tag}")
    typer.echo(f"  ✓ Server listening on {host}:{port}")
    typer.echo("")

    asyncio.run(blaze_app.serve(host, port))


@app.command()
def proto(
    app_path: str = typer.Argument(..., help="App import path (e.g. app:app)"),
    output_dir: str = typer.Option(".", help="Output directory for .proto files"),
) -> None:
    """Export generated .proto files."""
    blaze_app = load_app(app_path)
    path = export_proto(blaze_app, output_dir)
    typer.echo(f"✓ Proto written to {path}")


if __name__ == "__main__":
    app()
