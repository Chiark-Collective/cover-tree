"""Project-wide logging utilities that honour `RuntimeConfig`."""

from __future__ import annotations

import logging
from typing import Optional

from . import config as cx_config


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger configured according to the runtime configuration."""

    logger_name = "covertreex" if name is None else f"covertreex.{name}"
    runtime = cx_config.runtime_config()
    logger = logging.getLogger(logger_name)
    logger.setLevel(runtime.log_level)
    return logger
