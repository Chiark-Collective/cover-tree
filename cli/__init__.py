"""Command-line entrypoints for covertreex utilities."""

from .pcct.runtime_config import runtime_from_args  # re-export for compatibility

__all__ = ["runtime_from_args"]