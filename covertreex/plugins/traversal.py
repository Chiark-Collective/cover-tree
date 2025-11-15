from __future__ import annotations

from typing import Any, Iterable, Sequence

from covertreex.algo.traverse.strategies.registry import (
    deregister_traversal_strategy,
    describe_traversal_strategies,
    register_traversal_strategy,
)
from covertreex.logging import get_logger

from ._loader import load_entrypoint_plugins

LOGGER = get_logger("plugins.traversal")
GROUP = "covertreex.traversal"


def _iter_specs(payload: Any) -> Iterable[tuple[str, Any, Any, str | None]]:
    if payload is None:
        return ()
    if hasattr(payload, "__call__") and not hasattr(payload, "name"):
        return _iter_specs(payload())
    if isinstance(payload, dict):
        return [
            (
                payload["name"],
                payload["predicate"],
                payload["factory"],
                payload.get("origin"),
            )
        ]
    if hasattr(payload, "name") and hasattr(payload, "predicate") and hasattr(payload, "factory"):
        origin = getattr(payload, "origin", None)
        return [(payload.name, payload.predicate, payload.factory, origin)]
    if isinstance(payload, Sequence):
        items: list[tuple[str, Any, Any, str | None]] = []
        for entry in payload:
            items.extend(_iter_specs(entry))
        return items
    raise TypeError(f"Unsupported traversal plugin payload: {payload!r}")


def _register_payload(payload: Any) -> None:
    for name, predicate, factory, origin in _iter_specs(payload):
        register_traversal_strategy(
            name,
            predicate=predicate,
            factory=factory,
            origin=origin,
        )


def load_entrypoints() -> list[str]:
    return load_entrypoint_plugins(GROUP, _register_payload)


# Load external entry points eagerly.
_LOADED = load_entrypoints()
if _LOADED:
    LOGGER.debug("Loaded traversal entry points: %s", ", ".join(_LOADED))


def list_plugins() -> list[dict[str, str]]:
    return describe_traversal_strategies()


__all__ = ["list_plugins", "load_entrypoints", "register_traversal_strategy", "deregister_traversal_strategy"]
