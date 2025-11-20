from __future__ import annotations

import os
import platform
import tempfile
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import typer

from covertreex import config as cx_config
from covertreex.api import Runtime as ApiRuntime
from covertreex.telemetry import artifact_root
from profiles.loader import ProfileError
from profiles.overrides import OverrideError

from .support.runtime_utils import thread_env_snapshot as _thread_env_snapshot

try:  # pragma: no cover - optional dependency
    import numba  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    numba = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import jax  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    jax = None  # type: ignore

from covertreex.algo._mis_numba import NUMBA_AVAILABLE as NUMBA_MIS_AVAILABLE
from covertreex.algo._grid_numba import NUMBA_GRID_AVAILABLE
from covertreex.algo._scope_numba import NUMBA_SCOPE_AVAILABLE
from covertreex.algo._residual_scope_numba import NUMBA_RESIDUAL_SCOPE_AVAILABLE
from covertreex.algo._traverse_numba import NUMBA_TRAVERSAL_AVAILABLE
from covertreex.core._persistence_numba import NUMBA_PERSISTENCE_AVAILABLE
from covertreex.metrics._residual_numba import NUMBA_RESIDUAL_AVAILABLE
from covertreex.queries._knn_numba import NUMBA_QUERY_AVAILABLE


app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    invoke_without_command=True,
    help="Environment guardrails for PCCT runs.",
)


@dataclass(frozen=True)
class KernelCapability:
    label: str
    detail: str
    available: bool
    required: Callable[[cx_config.RuntimeConfig], bool]


def _is_residual_metric(cfg: cx_config.RuntimeConfig) -> bool:
    metric = (cfg.metric or "").lower()
    return "residual" in metric


_NUMBA_CAPABILITIES: Sequence[KernelCapability] = (
    KernelCapability(
        label="MIS kernels",
        detail="Randomised MIS selection speedups.",
        available=NUMBA_MIS_AVAILABLE,
        required=lambda cfg: cfg.enable_numba,
    ),
    KernelCapability(
        label="Traversal kernels",
        detail="Batch traversal / scope planning.",
        available=NUMBA_TRAVERSAL_AVAILABLE,
        required=lambda cfg: cfg.enable_numba,
    ),
    KernelCapability(
        label="Scope kernels",
        detail="Conflict scope builders.",
        available=NUMBA_SCOPE_AVAILABLE,
        required=lambda cfg: cfg.enable_numba,
    ),
    KernelCapability(
        label="Grid ordering kernels",
        detail="Hilbert/grid batch ordering.",
        available=NUMBA_GRID_AVAILABLE,
        required=lambda cfg: cfg.enable_numba,
    ),
    KernelCapability(
        label="k-NN kernels",
        detail="Numba-accelerated query stack.",
        available=NUMBA_QUERY_AVAILABLE,
        required=lambda cfg: cfg.enable_numba,
    ),
    KernelCapability(
        label="Persistence kernels",
        detail="Copy-on-write journal operations.",
        available=NUMBA_PERSISTENCE_AVAILABLE,
        required=lambda cfg: cfg.enable_numba,
    ),
    KernelCapability(
        label="Residual scope kernels",
        detail="Residual traversal scope streaming.",
        available=NUMBA_RESIDUAL_SCOPE_AVAILABLE,
        required=lambda cfg: cfg.enable_numba or _is_residual_metric(cfg),
    ),
    KernelCapability(
        label="Residual backend kernels",
        detail="Synthetic residual backend (SGEMM/whitening).",
        available=NUMBA_RESIDUAL_AVAILABLE,
        required=lambda cfg: cfg.enable_numba or _is_residual_metric(cfg),
    ),
)


@dataclass
class DoctorMessage:
    status: str
    label: str
    detail: str


class DoctorReport:
    def __init__(self) -> None:
        self.messages: List[DoctorMessage] = []

    def add(self, status: str, label: str, detail: str) -> None:
        self.messages.append(DoctorMessage(status=status, label=label, detail=detail))

    def render(self) -> None:
        colors = {
            "ok": "green",
            "warn": "yellow",
            "error": "red",
            "info": "cyan",
        }
        for message in self.messages:
            prefix = f"[{message.status}]"
            color = colors.get(message.status, None)
            typer.secho(f"{prefix:<8} {message.label}: {message.detail}", fg=color)

    def exit_code(self, *, fail_on_warning: bool) -> int:
        has_error = any(msg.status == "error" for msg in self.messages)
        has_warning = any(msg.status == "warn" for msg in self.messages)
        if has_error:
            return 2
        if fail_on_warning and has_warning:
            return 1
        return 0


def _load_runtime(profile: str, overrides: Optional[List[str]]) -> ApiRuntime:
    try:
        runtime = ApiRuntime.from_profile(profile, overrides=overrides)
    except (ProfileError, OverrideError, ValueError) as exc:
        raise typer.BadParameter(f"Invalid profile or overrides: {exc}") from exc
    return runtime


def _emit_runtime_overview(cfg: cx_config.RuntimeConfig, *, profile: str, overrides: Optional[List[str]]) -> None:
    override_label = ", ".join(overrides) if overrides else "none"
    metric_label = cfg.metric or "auto"
    typer.echo(
        f"pcct doctor | profile={profile} backend={cfg.backend} metric={metric_label} "
        f"precision={cfg.precision}"
    )
    typer.echo(
        f"devices={cfg.devices or ('cpu',)} enable_numba={cfg.enable_numba} "
        f"enable_sparse_traversal={cfg.enable_sparse_traversal} overrides={override_label}"
    )


def _check_numba(cfg: cx_config.RuntimeConfig, report: DoctorReport) -> None:
    numba_available = numba is not None
    if numba_available:
        report.add(
            "ok" if cfg.enable_numba else "info",
            "numba import",
            f"numba {getattr(numba, '__version__', 'unknown')} detected.",
        )
    else:
        status = "warn" if cfg.enable_numba else "info"
        detail = "Numba is not installed; kernel fallbacks will be used."
        if cfg.enable_numba:
            detail += " Install `numba` or set `--set enable_numba=false`."
        report.add(status, "numba import", detail)

    for capability in _NUMBA_CAPABILITIES:
        required = capability.required(cfg)
        if capability.available:
            status = "ok" if required else "info"
            suffix = "enabled" if required else "available"
            report.add(status, capability.label, f"{capability.detail} ({suffix}).")
            continue

        if required:
            report.add(
                "warn",
                capability.label,
                f"{capability.detail} unavailable; falling back to Python paths.",
            )
        else:
            report.add(
                "info",
                capability.label,
                f"{capability.detail} unavailable (not requested by current profile).",
            )


def _check_jax(cfg: cx_config.RuntimeConfig, report: DoctorReport) -> None:
    backend = cfg.backend
    jax_version = getattr(jax, "__version__", None)
    if backend != "jax":
        if jax_version:
            report.add("info", "jax backend", f"jax {jax_version} installed (backend not requested).")
        else:
            report.add("info", "jax backend", "Backend 'jax' not requested.")
        return

    if jax is None:
        report.add("error", "jax backend", "backend='jax' requested but JAX is not installed.")
        return

    try:
        devices = jax.devices()
        device_summary = ", ".join(device.platform for device in devices) or "none"
    except Exception:  # pragma: no cover - depends on environment
        device_summary = "unknown"
    try:
        x64_enabled = bool(jax.config.read("jax_enable_x64"))
    except Exception:  # pragma: no cover - depends on environment
        x64_enabled = False

    report.add("ok", "jax backend", f"jax {jax_version} detected (devices: {device_summary}).")
    if cfg.precision == "float64" and not x64_enabled:
        report.add(
            "warn",
            "jax precision",
            "float64 requested but JAX x64 support is disabled. Set JAX_ENABLE_X64=1.",
        )


def _check_artifact_root(report: DoctorReport) -> None:
    env_override = os.environ.get("COVERTREEX_ARTIFACT_ROOT")
    try:
        root = artifact_root(create=True)
    except OSError as exc:  # pragma: no cover - depends on environment
        report.add("error", "artifact root", f"Failed to initialise artifact root: {exc}")
        return

    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=".pcct-doctor-", delete=True) as handle:
            handle.write(b"0")
            handle.flush()
    except OSError as exc:
        report.add(
            "error",
            "artifact root",
            f"{root} is not writable ({exc}). Configure COVERTREEX_ARTIFACT_ROOT.",
        )
        return

    suffix = f"from $COVERTREEX_ARTIFACT_ROOT={env_override}" if env_override else "default path"
    report.add("ok", "artifact root", f"{root} writable ({suffix}).")


def _check_threads(report: DoctorReport) -> None:
    snapshot = _thread_env_snapshot()
    report.add(
        "info",
        "thread environment",
        f"BLAS={snapshot.get('blas_threads', 'auto')} NUMBA={snapshot.get('numba_threads', 'auto')}",
    )


def _emit_platform_metadata() -> None:
    typer.echo(
        f"platform={platform.platform()} python={platform.python_version()} "
        f"cpus={os.cpu_count() or 'unknown'}"
    )


@app.callback()
def doctor(
    profile: str = typer.Option(
        "default",
        "--profile",
        help="Profile to inspect (defaults to 'default').",
    ),
    set_override: Optional[List[str]] = typer.Option(
        None,
        "--set",
        metavar="PATH=VALUE",
        help="Apply dot-path overrides when loading the profile.",
    ),
    fail_on_warning: bool = typer.Option(
        False,
        "--fail-on-warning",
        help="Return exit code 1 when warnings are emitted.",
    ),
) -> None:
    """Run preflight environment diagnostics."""

    runtime = _load_runtime(profile, set_override)
    cfg = runtime.to_config()

    _emit_runtime_overview(cfg, profile=profile, overrides=set_override)
    _emit_platform_metadata()

    report = DoctorReport()
    _check_numba(cfg, report)
    _check_jax(cfg, report)
    _check_artifact_root(report)
    _check_threads(report)

    report.render()
    raise typer.Exit(code=report.exit_code(fail_on_warning=fail_on_warning))


__all__ = ["app"]
