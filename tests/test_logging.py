import logging

import pytest

from covertreex import config as cx_config
from covertreex.logging import get_logger


def test_logger_respects_runtime_level(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COVERTREEX_LOG_LEVEL", "DEBUG")
    cx_config.reset_runtime_config_cache()

    logger = get_logger("tests.logging")

    assert logger.level == logging.DEBUG
    assert logger.name == "covertreex.tests.logging"

    cx_config.reset_runtime_config_cache()
