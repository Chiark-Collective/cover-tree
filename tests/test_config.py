import os

import pytest

from covertreex import config as cx_config


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "COVERTREEX_BACKEND",
        "COVERTREEX_PRECISION",
        "COVERTREEX_DEVICE",
        "COVERTREEX_ENABLE_NUMBA",
        "COVERTREEX_LOG_LEVEL",
        "COVERTREEX_MIS_SEED",
        "JAX_ENABLE_X64",
        "JAX_PLATFORM_NAME",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_runtime_config_defaults(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()

    assert runtime.backend == "jax"
    assert runtime.precision == "float64"
    assert runtime.jax_enable_x64 is True
    assert runtime.devices  # at least one device recorded
    assert runtime.primary_platform in {"cpu", "gpu", "tpu"}


def test_precision_override(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_PRECISION", "float32")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()

    assert runtime.precision == "float32"
    assert runtime.jax_enable_x64 is False


def test_device_fallback_to_cpu(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    # Request an unlikely device to trigger fallback logic.
    monkeypatch.setenv("COVERTREEX_DEVICE", "gpu:99")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()

    assert runtime.primary_platform == "cpu"


def test_invalid_backend(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_BACKEND", "invalid-backend")
    cx_config.reset_runtime_config_cache()

    with pytest.raises(ValueError):
        cx_config.runtime_config()


def test_mis_seed_parsing(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_MIS_SEED", "123")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.mis_seed == 123
