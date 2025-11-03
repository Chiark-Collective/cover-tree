"""Parallel compressed cover tree library."""

from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("covertreex")
except Exception:  # pragma: no cover - best effort during local development
    __version__ = "0.0.1"

__all__ = ["__version__"]
