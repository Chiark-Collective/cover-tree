"""
Compatibility shim for :mod:`covertreex.runtime.diagnostics`.
"""

from covertreex.runtime.diagnostics import *  # noqa: F401,F403

from covertreex.runtime import diagnostics as _diagnostics

__all__ = getattr(_diagnostics, "__all__", [])
