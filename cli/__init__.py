"""Command-line entrypoints for covertreex utilities."""

from .runtime import runtime_from_args  # re-export for compatibility

__all__ = ["runtime_from_args"]
