"""Hot reload support for development.

Uses ``watchfiles`` to watch for Python file changes and restart
the server process automatically.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

log = logging.getLogger("blazerpc.reload")


def _run_server(app_path: str, host: str, port: int) -> None:
    """Entry point for the child process — load and serve."""
    from blazerpc.cli.serve import load_app

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import uvloop  # type: ignore[import-untyped]

        uvloop.install()
    except ImportError:
        pass

    blaze_app = load_app(app_path)
    asyncio.run(blaze_app.serve(host, port))


def run_with_reload(app_path: str, host: str, port: int) -> None:
    """Run the server in a subprocess, restarting on file changes.

    Requires the ``watchfiles`` package (install with
    ``pip install blazerpc[reload]`` or ``pip install watchfiles``).
    """
    try:
        from watchfiles import run_process
    except ImportError:
        print(
            "Error: watchfiles is required for --reload.\n"
            "Install it with: pip install blazerpc[reload]",
            file=sys.stderr,
        )
        raise SystemExit(1)

    log.info("Watching for file changes (cwd: %s)", ".")
    run_process(
        ".",
        target=_run_server,
        args=(app_path, host, port),
        watch_filter=_python_filter,
    )


def _python_filter(change: Any, path: str) -> bool:
    """Only trigger reload for Python file changes."""
    return path.endswith(".py")
