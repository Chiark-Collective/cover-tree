from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import jax

_SUPPORTED_BACKENDS = {"jax"}
_SUPPORTED_PRECISION = {"float32", "float64"}


def _bool_from_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_devices(raw: str | None) -> Tuple[str, ...]:
    if not raw:
        return ()
    devices = tuple(
        spec.strip().lower()
        for spec in raw.split(",")
        if spec.strip()
    )
    return devices


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value '{raw}'") from exc


def _normalise_precision(value: str | None) -> str:
    if value is None:
        return "float64"
    value = value.strip().lower()
    if value not in _SUPPORTED_PRECISION:
        raise ValueError(f"Unsupported precision '{value}'. Expected one of {_SUPPORTED_PRECISION}.")
    return value


def _infer_precision_from_env() -> str:
    precision = os.getenv("COVERTREEX_PRECISION")
    if precision:
        return _normalise_precision(precision)
    jax_enable_x64 = os.getenv("JAX_ENABLE_X64")
    if jax_enable_x64 is not None:
        return "float64" if _bool_from_env(jax_enable_x64, default=False) else "float32"
    return "float64"


def _infer_backend_from_env() -> str:
    backend = os.getenv("COVERTREEX_BACKEND", "jax").strip().lower()
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of {_SUPPORTED_BACKENDS}.")
    return backend


def _device_label(device: jax.Device) -> str:
    index = getattr(device, "id", getattr(device, "device_id", 0))
    return f"{device.platform}:{index}"


def _resolve_jax_devices(requested: Tuple[str, ...]) -> Tuple[str, ...]:
    available = jax.devices()
    if not available:
        return ()
    if not requested:
        return tuple(_device_label(device) for device in available)

    selected: list[str] = []
    for spec in requested:
        if ":" in spec:
            platform, idx = spec.split(":", 1)
            matches = [
                device
                for device in available
                if device.platform == platform and str(getattr(device, "id", None)) == idx
            ]
        else:
            matches = [device for device in available if device.platform == spec]
        if matches:
            selected.extend(_device_label(device) for device in matches)
    if selected:
        return tuple(selected)

    cpu_devices = [device for device in available if device.platform == "cpu"]
    if cpu_devices:
        logging.getLogger("covertreex").debug(
            "Requested devices %s unavailable; falling back to CPU.", requested
        )
        return tuple(_device_label(device) for device in cpu_devices)
    return tuple(_device_label(device) for device in available)


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    precision: str
    devices: Tuple[str, ...]
    enable_numba: bool
    log_level: str
    mis_seed: int | None

    @property
    def jax_enable_x64(self) -> bool:
        return self.precision == "float64"

    @property
    def primary_platform(self) -> str | None:
        if not self.devices:
            return None
        return self.devices[0].split(":", 1)[0]


def _apply_jax_runtime_flags(config: RuntimeConfig) -> None:
    if config.backend != "jax":
        return
    jax.config.update("jax_enable_x64", config.jax_enable_x64)

    if config.primary_platform and "JAX_PLATFORM_NAME" not in os.environ:
        jax.config.update("jax_platform_name", config.primary_platform)


def _configure_logging(level: str) -> None:
    logger = logging.getLogger("covertreex")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)


@lru_cache(maxsize=None)
def runtime_config() -> RuntimeConfig:
    backend = _infer_backend_from_env()
    precision = _normalise_precision(_infer_precision_from_env())
    requested_devices = _parse_devices(os.getenv("COVERTREEX_DEVICE"))
    devices = _resolve_jax_devices(requested_devices) if backend == "jax" else ()
    enable_numba = _bool_from_env(os.getenv("COVERTREEX_ENABLE_NUMBA"), default=False)
    log_level = os.getenv("COVERTREEX_LOG_LEVEL", "INFO").upper()
    mis_seed = _parse_optional_int(os.getenv("COVERTREEX_MIS_SEED"))

    config = RuntimeConfig(
        backend=backend,
        precision=precision,
        devices=devices,
        enable_numba=enable_numba,
        log_level=log_level,
        mis_seed=mis_seed,
    )
    _apply_jax_runtime_flags(config)
    _configure_logging(config.log_level)
    return config


def reset_runtime_config_cache() -> None:
    runtime_config.cache_clear()
