"""
Compatibility shim that forwards to the refactored CLI module in ``cli.queries``.

The actual implementation now lives under ``cli/`` so we can share the same
dataset helpers and runtime plumbing across entrypoints without mutating
environment variables mid-run.
"""

from cli import queries as _queries
from cli.queries import *  # noqa: F401,F403

__all__ = getattr(_queries, "__all__", [])


def main() -> None:
    _queries.main()


if __name__ == "__main__":
    main()
