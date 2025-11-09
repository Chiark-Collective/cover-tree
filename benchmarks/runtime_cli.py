"""
Compatibility shim ensuring ``benchmarks.runtime_cli`` keeps working.
"""

from cli.runtime import *  # noqa: F401,F403

from cli import runtime as _runtime

__all__ = getattr(_runtime, "__all__", [])
