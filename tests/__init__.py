"""Pytest namespace package placeholder with a stable import root."""

import sys
from pathlib import Path

# Ensure the repository root is on sys.path for integration tests that rely on direct imports.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
