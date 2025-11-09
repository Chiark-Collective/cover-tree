"""
Backwards-compatible shim for the runtime configuration helpers.

The canonical implementations now live under :mod:`covertreex.runtime`.
"""

from covertreex.runtime.config import *  # noqa: F401,F403

from covertreex.runtime import config as _config

__all__ = getattr(_config, "__all__", [])
