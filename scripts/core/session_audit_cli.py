#!/usr/bin/env python3
"""Entry point for ``opc session audit``. Contains no analysis logic."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from scripts.core.session_audit.cli import main  # noqa: E402, I001

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
