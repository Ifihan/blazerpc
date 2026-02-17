"""blaze serve command implementation."""

from __future__ import annotations

import importlib
from typing import Any

from blazerpc.app import BlazeApp
from blazerpc.exceptions import ConfigurationError


def load_app(import_string: str) -> Any:
    """Load a :class:`~blazerpc.app.BlazeApp` from an import string.

    The *import_string* must be in ``module:attribute`` form, e.g.
    ``"myapp.main:app"``.
    """
    if ":" not in import_string:
        raise ConfigurationError(
            f"Invalid import string '{import_string}'. "
            "Expected format: 'module:attribute' (e.g. 'app:app')"
        )

    module_path, _, attr_name = import_string.partition(":")

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ConfigurationError(
            f"Could not import module '{module_path}': {exc}"
        ) from exc

    try:
        app = getattr(module, attr_name)
    except AttributeError as exc:
        raise ConfigurationError(
            f"Module '{module_path}' has no attribute '{attr_name}'"
        ) from exc

    if not isinstance(app, BlazeApp):
        raise ConfigurationError(
            f"'{import_string}' is not a BlazeApp instance (got {type(app).__name__})"
        )

    return app
