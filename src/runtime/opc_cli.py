"""Packaged console-script shim for the source-tree OPC dispatcher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_dispatcher_main():
    dispatcher_path = Path(__file__).resolve().parents[2] / "scripts" / "core" / "cli.py"
    spec = importlib.util.spec_from_file_location("opc_source_dispatcher", dispatcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load OPC dispatcher from {dispatcher_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


main = _load_dispatcher_main()


__all__ = ["main"]
