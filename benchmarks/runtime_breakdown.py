"""
Compatibility shim for the refactored runtime breakdown CLI living under ``cli``.
"""

from cli import runtime_breakdown as _runtime_breakdown
from cli.runtime_breakdown import *  # noqa: F401,F403

__all__ = getattr(_runtime_breakdown, "__all__", [])


def main() -> None:
    _runtime_breakdown.main()


if __name__ == "__main__":
    main()
