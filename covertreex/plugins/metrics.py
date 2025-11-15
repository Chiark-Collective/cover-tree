from __future__ import annotations

from typing import Any, Iterable, Sequence

from covertreex.core.metrics import Metric, _REGISTRY, describe_registered_metrics
from covertreex.logging import get_logger

from ._loader import load_entrypoint_plugins

LOGGER = get_logger("plugins.metrics")
GROUP = "covertreex.metrics"


def _iter_metrics(payload: Any) -> Iterable[Metric]:
    if payload is None:
        return ()
    if isinstance(payload, Metric):
        return [payload]
    if hasattr(payload, "__call__") and not isinstance(payload, Metric):
        return _iter_metrics(payload())
    if isinstance(payload, Sequence):
        items: list[Metric] = []
        for entry in payload:
            items.extend(_iter_metrics(entry))
        return items
    raise TypeError(f"Unsupported metric plugin payload: {payload!r}")


def _register_payload(payload: Any) -> None:
    for metric in _iter_metrics(payload):
        _REGISTRY.register(metric, overwrite=True)


def load_entrypoints() -> list[str]:
    return load_entrypoint_plugins(GROUP, _register_payload)


_LOADED = load_entrypoints()
if _LOADED:
    LOGGER.debug("Loaded metric entry points: %s", ", ".join(_LOADED))


def list_metrics() -> list[dict[str, str]]:
    return list(describe_registered_metrics())


__all__ = ["list_metrics", "load_entrypoints"]
