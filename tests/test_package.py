"""Import and distribution metadata tests for the top-level package."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import subprocess
import sys

import blazerpc

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def test_runtime_version_matches_distribution_metadata() -> None:
    assert blazerpc.__version__ == importlib.metadata.version("blazerpc")


def test_runtime_version_is_available_from_source_checkout() -> None:
    script = """
import importlib.metadata

metadata_version = importlib.metadata.version
def missing_metadata(name):
    if name == "blazerpc":
        raise importlib.metadata.PackageNotFoundError(name)
    return metadata_version(name)

importlib.metadata.version = missing_metadata
import blazerpc
print(blazerpc.__version__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == blazerpc.__version__


def test_base_install_exports_without_aiohttp() -> None:
    script = """
import importlib.abc
import sys

class BlockAiohttp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "aiohttp" or fullname.startswith("aiohttp."):
            raise ModuleNotFoundError("No module named 'aiohttp'", name="aiohttp")
        return None

sys.meta_path.insert(0, BlockAiohttp())
import blazerpc
assert "JsonRpcClient" not in blazerpc.__all__
namespace = {}
exec("from blazerpc import *", namespace)
try:
    blazerpc.JsonRpcClient
except ImportError as exc:
    assert "pip install blazerpc[jsonrpc]" in str(exc)
else:
    raise AssertionError("JsonRpcClient access should require aiohttp")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_jsonrpc_client_is_exported_when_dependency_is_available() -> None:
    from blazerpc.jsonrpc_client import JsonRpcClient

    assert "JsonRpcClient" in blazerpc.__all__
    assert blazerpc.JsonRpcClient is JsonRpcClient


def test_project_metadata_matches_runtime_and_complete_extra() -> None:
    with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    assert project["version"] == blazerpc.__version__
    assert project["optional-dependencies"]["all"] == [
        "blazerpc[pytorch,tensorflow,onnx,otel,jsonrpc,reload]"
    ]
