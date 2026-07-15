"""CLI entry point for the blaze command."""

from __future__ import annotations

import asyncio
import logging

import typer

from blazerpc.cli.reload import run_with_reload
from blazerpc.cli.proto import export_proto
from blazerpc.cli.serve import load_app

app = typer.Typer(
    name="blaze",
    help="BlazeRPC - Lightning-fast gRPC for ML inference",
    add_completion=False,
)

_TRANSPORTS = ("grpc", "jsonrpc", "both")


def _validate_serve_options(
    transport: str, workers: int, port: int, http_port: int
) -> None:
    if transport not in _TRANSPORTS:
        raise typer.BadParameter(
            "must be grpc, jsonrpc, or both", param_hint="--transport"
        )
    if workers != 1:
        raise typer.BadParameter("only 1 worker is supported", param_hint="--workers")
    if not 1 <= port <= 65535:
        raise typer.BadParameter("must be between 1 and 65535", param_hint="--port")
    if not 1 <= http_port <= 65535:
        raise typer.BadParameter(
            "must be between 1 and 65535", param_hint="--http-port"
        )
    if transport == "both" and port == http_port:
        raise typer.BadParameter(
            "must differ from --port when using both transports",
            param_hint="--http-port",
        )


@app.command()
def serve(
    app_path: str = typer.Argument(..., help="App import path (e.g. app:app)"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(50051, help="Port to listen on (gRPC)"),
    http_port: int = typer.Option(8080, help="Port for JSON-RPC HTTP server"),
    transport: str = typer.Option("grpc", help="Transport: grpc, jsonrpc, or both"),
    workers: int = typer.Option(1, help="Worker count (only 1 is supported)"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
    """Start the BlazeRPC server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _validate_serve_options(transport, workers, port, http_port)

    if reload:
        typer.echo("")
        typer.echo("⚡ BlazeRPC server starting (reload mode)...")
        typer.echo("  ✓ Watching for changes in current directory")
        if transport in ("grpc", "both"):
            typer.echo(f"  ✓ gRPC will listen on {host}:{port}")
        if transport in ("jsonrpc", "both"):
            typer.echo(f"  ✓ JSON-RPC will listen on {host}:{http_port}")
        typer.echo("")
        run_with_reload(app_path, host, port, http_port, transport)
        return

    blaze_app = load_app(app_path)

    # Install uvloop when available for better performance.
    try:
        import uvloop

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
