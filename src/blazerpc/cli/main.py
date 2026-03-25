"""CLI entry point for the blaze command."""

from __future__ import annotations

import asyncio
import logging

import typer

from blazerpc.cli.proto import export_proto
from blazerpc.cli.serve import load_app
from blazerpc.cli.reload import run_with_reload

app = typer.Typer(
    name="blaze",
    help="BlazeRPC - Lightning-fast gRPC for ML inference",
    add_completion=False,
)


@app.command()
def serve(
    app_path: str = typer.Argument(..., help="App import path (e.g. app:app)"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(50051, help="Port to listen on (gRPC)"),
    http_port: int = typer.Option(8080, help="Port for JSON-RPC HTTP server"),
    transport: str = typer.Option("grpc", help="Transport: grpc, jsonrpc, or both"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
    """Start the BlazeRPC server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if reload:
        typer.echo("")
        typer.echo("⚡ BlazeRPC server starting (reload mode)...")
        typer.echo("  ✓ Watching for changes in current directory")
        typer.echo(f"  ✓ Server will listen on {host}:{port}")
        typer.echo("")
        run_with_reload(app_path, host, port)
        return

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

    if transport == "grpc":
        typer.echo(f"  ✓ gRPC listening on {host}:{port}")
        typer.echo("")
        asyncio.run(blaze_app.serve(host, port))
    elif transport == "jsonrpc":
        typer.echo(f"  ✓ JSON-RPC listening on {host}:{http_port}")
        typer.echo("")
        asyncio.run(blaze_app.serve_jsonrpc(host, http_port))
    elif transport == "both":
        typer.echo(f"  ✓ gRPC listening on {host}:{port}")
        typer.echo(f"  ✓ JSON-RPC listening on {host}:{http_port}")
        typer.echo("")
        asyncio.run(blaze_app.serve_both(host, grpc_port=port, http_port=http_port))
    else:
        typer.echo(f"Unknown transport: {transport}. Use grpc, jsonrpc, or both.")
        raise typer.Exit(1)


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
