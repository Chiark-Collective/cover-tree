"""
Compatibility shim for :mod:`covertreex.runtime.logging`.
"""

from covertreex.runtime.logging import *  # noqa: F401,F403

from covertreex.runtime import logging as _logging

__all__ = getattr(_logging, "__all__", [])
