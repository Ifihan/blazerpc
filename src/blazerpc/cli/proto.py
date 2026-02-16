"""blaze proto command implementation."""

from __future__ import annotations

import os
from pathlib import Path

from blazerpc.app import BlazeApp
from blazerpc.codegen.proto import ProtoGenerator


def export_proto(app: BlazeApp, output_dir: str) -> str:
    """Generate the .proto file and write it to *output_dir*.

    Returns the path to the written file.
    """
    proto_content = ProtoGenerator().generate(app.registry)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / "blaze_service.proto"
    file_path.write_text(proto_content)
    return str(file_path)
