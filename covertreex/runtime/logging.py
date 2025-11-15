"""Project-wide logging utilities that honour `RuntimeConfig`."""

from __future__ import annotations

import logging
from typing import Optional

from . import config as cx_config


def _resolve_runtime_config(
    *,
    runtime: cx_config.RuntimeConfig | None = None,
    context: cx_config.RuntimeContext | None = None,
) -> cx_config.RuntimeConfig | None:
    if runtime is not None:
        return runtime
    if context is not None:
        return context.config
    active = cx_config.current_runtime_context()
    if active is not None:
        return active.config
    return None


def get_logger(
    name: Optional[str] = None,
    *,
    runtime: cx_config.RuntimeConfig | None = None,
    context: cx_config.RuntimeContext | None = None,
) -> logging.Logger:
    """Return a logger configured according to the runtime configuration."""

    logger_name = "covertreex" if name is None else f"covertreex.{name}"
    logger = logging.getLogger(logger_name)
    resolved = _resolve_runtime_config(runtime=runtime, context=context)
    if resolved is not None:
        logger.setLevel(resolved.log_level)
    return logger


__all__ = ["get_logger"]
