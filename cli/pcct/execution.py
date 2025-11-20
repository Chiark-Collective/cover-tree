from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional

from covertreex import config as cx_config, reset_residual_metric
from covertreex.api import Runtime as ApiRuntime
from covertreex.telemetry import generate_run_id, timestamped_artifact

from .runtime_config import runtime_from_args
from .support.gate_utils import append_gate_profile_log as _append_gate_profile_log
from .support.runtime_utils import (
    ensure_thread_env_defaults as _ensure_thread_env_defaults,
    thread_env_snapshot as _thread_env_snapshot,
    resolve_artifact_arg as _resolve_artifact_arg,
    validate_residual_runtime as _validate_residual_runtime,
)
from .support.telemetry_utils import CLITelemetryHandles, initialise_cli_telemetry


@dataclass
class BenchmarkRun:
    """Holds shared CLI runtime/telemetry state for benchmark-style commands."""

    args: Any
    run_id: str
    runtime: ApiRuntime
    runtime_snapshot: Dict[str, Any]
    thread_snapshot: Dict[str, Any]
    context: cx_config.RuntimeContext
    log_metadata: Dict[str, Any]
    telemetry_handles: CLITelemetryHandles | None
    residual_gate_profile_path: Optional[str]
    residual_gate_profile_log: Optional[str]

    @property
    def log_writer(self):
        return self.telemetry_handles.log_writer if self.telemetry_handles else None

    @property
    def log_path(self) -> Optional[str]:
        return self.telemetry_handles.log_path if self.telemetry_handles else None

    @property
    def scope_cap_recorder(self):
        return self.telemetry_handles.scope_cap_recorder if self.telemetry_handles else None

    @property
    def telemetry_view(self):
        return self.telemetry_handles.traversal_view if self.telemetry_handles else None


def _prepare_residual_gate_artifacts(args: Any, run_id: str) -> tuple[Optional[str], Optional[str]]:
    if getattr(args, "metric", None) != "residual":
        setattr(args, "residual_gate_profile_path", None)
        setattr(args, "residual_gate_profile_log", None)
        return None, None

    profile_log_path = _resolve_artifact_arg(getattr(args, "residual_gate_profile_log", None), category="profiles")
    profile_json_path: Optional[str]
    if getattr(args, "residual_gate_profile_path", None):
        profile_json_path = _resolve_artifact_arg(
            getattr(args, "residual_gate_profile_path"),
            category="profiles",
        )
    elif profile_log_path:
        profile_json_path = str(
            timestamped_artifact(
                category="profiles",
                prefix=f"residual_gate_profile_{run_id}",
                suffix=".json",
            )
        )
    else:
        profile_json_path = None
    setattr(args, "residual_gate_profile_path", profile_json_path)
    setattr(args, "residual_gate_profile_log", profile_log_path)
    return profile_json_path, profile_log_path


@contextmanager
def benchmark_run(
    args: Any,
    *,
    benchmark: str,
    metadata: Mapping[str, Any],
) -> Iterator[BenchmarkRun]:
    """Context manager that wires runtime/context/telemetry for CLI commands."""

    run_id = getattr(args, "run_id", None) or generate_run_id()
    gate_profile_path, gate_profile_log = _prepare_residual_gate_artifacts(args, run_id)
    _ensure_thread_env_defaults()
    runtime = runtime_from_args(args)
    runtime_snapshot: Dict[str, Any] = dict(runtime.to_model().model_dump())
    if getattr(args, "metric", None) == "residual":
        _validate_residual_runtime(runtime_snapshot)
    context = runtime.activate()
    thread_snapshot = _thread_env_snapshot()
    runtime_snapshot["runtime_blas_threads"] = thread_snapshot["blas_threads"]
    runtime_snapshot["runtime_numba_threads"] = thread_snapshot["numba_threads"]

    log_metadata: Dict[str, Any] = {"benchmark": benchmark}
    if metadata:
        log_metadata.update(metadata)

    telemetry_handles: CLITelemetryHandles | None = None
    run: BenchmarkRun | None = None
    try:
        telemetry_handles = initialise_cli_telemetry(
            args=args,
            run_id=run_id,
            runtime_snapshot=runtime_snapshot,
            log_metadata=log_metadata,
            context=context,
        )
        run = BenchmarkRun(
            args=args,
            run_id=run_id,
            runtime=runtime,
            runtime_snapshot=runtime_snapshot,
            thread_snapshot=thread_snapshot,
            context=context,
            log_metadata=log_metadata,
            telemetry_handles=telemetry_handles,
            residual_gate_profile_path=gate_profile_path,
            residual_gate_profile_log=gate_profile_log,
        )
        yield run
    finally:
        reset_residual_metric()
        if run is not None and getattr(args, "metric", None) == "residual":
            _append_gate_profile_log(
                profile_json_path=run.residual_gate_profile_path,
                profile_log_path=run.residual_gate_profile_log,
                run_id=run.run_id,
                log_metadata=run.log_metadata,
                runtime_snapshot=run.runtime_snapshot,
                batch_log_path=run.log_path,
            )
        cx_config.reset_runtime_context()
        if telemetry_handles is not None:
            telemetry_handles.close()


__all__ = ["BenchmarkRun", "benchmark_run"]
