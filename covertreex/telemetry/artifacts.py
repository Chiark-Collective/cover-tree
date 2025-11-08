from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Optional, Union

PathLike = Union[str, os.PathLike[str]]

_ARTIFACT_ENV = "COVERTREEX_ARTIFACT_ROOT"
_DEFAULT_ROOT = Path("artifacts")

__all__ = [
    "artifact_root",
    "resolve_artifact_path",
    "timestamped_artifact",
    "generate_run_id",
]


def artifact_root(create: bool = True) -> Path:
    """Return the base path for artifacts, creating it if requested."""

    root_env = os.environ.get(_ARTIFACT_ENV)
    root = Path(root_env).expanduser() if root_env else _DEFAULT_ROOT
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_artifact_path(
    path: PathLike,
    *,
    category: Optional[str] = None,
    create_parents: bool = True,
) -> Path:
    """
    Normalise a user-provided path so relative locations live under the artifact root.

    Absolute paths are respected; relative paths are joined beneath `artifact_root()/category`.
    """

    target = Path(path).expanduser()
    if not target.is_absolute():
        base = artifact_root()
        if category:
            base = base / category
        target = base / target
    if create_parents:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def timestamped_artifact(
    *,
    category: Optional[str],
    prefix: str,
    suffix: str,
    create_parents: bool = True,
) -> Path:
    """Construct an artifact path with a timestamped filename."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}_{stamp}{suffix}"
    return resolve_artifact_path(filename, category=category, create_parents=create_parents)


def generate_run_id(*, prefix: str = "pcct") -> str:
    """Produce a monotonic, human-readable run identifier."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    entropy = token_hex(3)
    return f"{prefix}-{stamp}-{entropy}"
