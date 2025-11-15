from __future__ import annotations

from importlib import metadata
from typing import Any, Callable, Iterable, Sequence

from covertreex.logging import get_logger

LOGGER = get_logger("plugins.loader")


def _select_entry_points(group: str) -> Sequence[Any]:
    try:
        entry_points = metadata.entry_points()
    except Exception:  # pragma: no cover - best effort when metadata lookup fails
        LOGGER.debug("Failed to read entry points for group %s.", group)
        return ()
    if hasattr(entry_points, "select"):
        return entry_points.select(group=group)  # type: ignore[attr-defined]
    return entry_points.get(group, ())  # type: ignore[call-arg]


def load_entrypoint_plugins(group: str, handler: Callable[[Any], None]) -> list[str]:
    """Load plugins registered via setuptools entry points."""

    loaded: list[str] = []
    for entry in _select_entry_points(group):
        try:
            payload = entry.load()
            handler(payload)
            loaded.append(entry.name)
        except Exception:  # pragma: no cover - defensive logging path
            LOGGER.exception("Failed to load plugin entry point '%s' (%s).", entry.name, group)
    return loaded
