from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

BENCHMARK_BATCH_SCHEMA_VERSION = 2
BENCHMARK_BATCH_SCHEMA_ID = f"covertreex.benchmark_batch.v{BENCHMARK_BATCH_SCHEMA_VERSION}"
BENCHMARK_BATCH_JSON_SCHEMA: Dict[str, Any] = {
    "id": BENCHMARK_BATCH_SCHEMA_ID,
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "description": "Per-batch PCCT telemetry emitted during benchmark runs.",
    "type": "object",
    "required": [
        "schema_id",
        "schema_version",
        "run_id",
        "run_hash",
        "timestamp",
        "batch_event_index",
        "batch_index",
        "batch_size",
        "runtime",
        "metadata",
    ],
    "properties": {
        "schema_id": {"const": BENCHMARK_BATCH_SCHEMA_ID},
        "schema_version": {"type": "integer", "minimum": 1},
        "run_id": {"type": "string"},
        "run_hash": {"type": "string"},
        "runtime_digest": {"type": "string"},
        "timestamp": {"type": "number"},
        "batch_event_index": {"type": "integer", "minimum": 0},
        "batch_index": {"type": "integer", "minimum": 0},
        "batch_size": {"type": "integer", "minimum": 0},
        "runtime": {"type": "object"},
        "metadata": {"type": "object"},
        "seed_pack": {"type": "object"},
    },
    "additionalProperties": True,
}


@dataclass(frozen=True)
class BenchmarkBatchRecord:
    """Typed representation of benchmark batch telemetry rows."""

    schema_id: str
    schema_version: int
    run_id: str
    run_hash: str
    runtime_digest: str | None
    timestamp: float
    batch_event_index: int
    batch_index: int
    batch_size: int
    runtime: Mapping[str, Any]
    metadata: Mapping[str, Any]
    seed_pack: Mapping[str, Any]
    measurements: Mapping[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BenchmarkBatchRecord":
        missing = [
            field
            for field in (
                "schema_id",
                "schema_version",
                "run_id",
                "run_hash",
                "timestamp",
                "batch_event_index",
                "batch_index",
                "batch_size",
            )
            if field not in payload
        ]
        if missing:
            raise ValueError(f"Telemetry payload missing required fields: {missing}")
        if payload["schema_id"] != BENCHMARK_BATCH_SCHEMA_ID:
            raise ValueError(
                "Unexpected schema id: {}".format(payload.get("schema_id"))
            )
        runtime = payload.get("runtime") or {}
        metadata = payload.get("metadata") or {}
        seed_pack = payload.get("seed_pack") or {}
        runtime_digest = payload.get("runtime_digest")
        reserved = {
            "schema_id",
            "schema_version",
            "run_id",
            "run_hash",
            "runtime_digest",
            "timestamp",
            "batch_event_index",
            "batch_index",
            "batch_size",
            "runtime",
            "metadata",
            "seed_pack",
        }
        measurements = {k: v for k, v in payload.items() if k not in reserved}
        return cls(
            schema_id=str(payload["schema_id"]),
            schema_version=int(payload["schema_version"]),
            run_id=str(payload["run_id"]),
            run_hash=str(payload["run_hash"]),
            runtime_digest=str(runtime_digest) if runtime_digest is not None else None,
            timestamp=float(payload["timestamp"]),
            batch_event_index=int(payload["batch_event_index"]),
            batch_index=int(payload["batch_index"]),
            batch_size=int(payload["batch_size"]),
            runtime=dict(runtime),
            metadata=dict(metadata),
            seed_pack=dict(seed_pack),
            measurements=measurements,
        )

RESIDUAL_SCOPE_CAP_SCHEMA_VERSION = 1
RESIDUAL_SCOPE_CAP_SCHEMA_ID = "covertreex.residual_scope_cap_summary.v1"
RESIDUAL_SCOPE_CAP_SCHEMA: Dict[str, object] = {
    "id": RESIDUAL_SCOPE_CAP_SCHEMA_ID,
    "version": RESIDUAL_SCOPE_CAP_SCHEMA_VERSION,
    "description": "Summary tables derived from traversal residual scope radii.",
    "required": ("schema", "schema_id", "levels", "metadata"),
}

RESIDUAL_GATE_PROFILE_SCHEMA_VERSION = 2
RESIDUAL_GATE_PROFILE_SCHEMA_ID = "covertreex.residual_gate_profile.v2"
RESIDUAL_GATE_PROFILE_SCHEMA: Dict[str, object] = {
    "id": RESIDUAL_GATE_PROFILE_SCHEMA_ID,
    "version": RESIDUAL_GATE_PROFILE_SCHEMA_VERSION,
    "description": "Aggregated Gate-1 profile payload derived from residual benchmark telemetry.",
    "required": (
        "schema_id",
        "run_id",
        "radius_bin_edges",
        "max_whitened",
        "max_ratio",
        "counts",
        "samples_total",
    ),
}

RUNTIME_BREAKDOWN_SCHEMA_ID = "covertreex.runtime_breakdown.v1"
RUNTIME_BREAKDOWN_SCHEMA: Dict[str, object] = {
    "id": RUNTIME_BREAKDOWN_SCHEMA_ID,
    "description": "Warm-up vs steady-state runtime metrics for PCCT and baselines.",
    "fields": (
        "schema_id",
        "run",
        "label",
        "build_warmup_seconds",
        "build_steady_seconds",
        "build_total_seconds",
        "query_warmup_seconds",
        "query_steady_seconds",
        "build_cpu_seconds",
        "build_cpu_utilisation",
        "build_rss_delta_bytes",
        "build_max_rss_bytes",
        "query_cpu_seconds",
        "query_cpu_utilisation",
        "query_rss_delta_bytes",
        "query_max_rss_bytes",
    ),
}

RUNTIME_BREAKDOWN_CHUNK_FIELDS: Tuple[str, ...] = (
    "traversal_chunk_segments_warmup",
    "traversal_chunk_segments_steady",
    "traversal_chunk_emitted_warmup",
    "traversal_chunk_emitted_steady",
    "traversal_chunk_max_members_warmup",
    "traversal_chunk_max_members_steady",
    "conflict_chunk_segments_warmup",
    "conflict_chunk_segments_steady",
    "conflict_chunk_emitted_warmup",
    "conflict_chunk_emitted_steady",
    "conflict_chunk_max_members_warmup",
    "conflict_chunk_max_members_steady",
)

RUNTIME_BREAKDOWN_FIELDNAMES: Tuple[str, ...] = (
    "schema_id",
    "run_id",
    "run",
    "label",
    "build_warmup_seconds",
    "build_steady_seconds",
    "build_total_seconds",
    "query_warmup_seconds",
    "query_steady_seconds",
    "build_cpu_seconds",
    "build_cpu_utilisation",
    "build_rss_delta_bytes",
    "build_max_rss_bytes",
    "query_cpu_seconds",
    "query_cpu_utilisation",
    "query_rss_delta_bytes",
    "query_max_rss_bytes",
) + RUNTIME_BREAKDOWN_CHUNK_FIELDS


def runtime_breakdown_fieldnames(*, include_run: bool = True) -> Tuple[str, ...]:
    """Return stable CSV headers for runtime breakdown telemetry."""

    fields = list(RUNTIME_BREAKDOWN_FIELDNAMES)
    if not include_run and "run" in fields:
        fields.remove("run")
    return tuple(fields)


BATCH_OPS_RESULT_SCHEMA_ID = "covertreex.batch_ops_summary.v1"
BATCH_OPS_RESULT_SCHEMA: Dict[str, object] = {
    "id": BATCH_OPS_RESULT_SCHEMA_ID,
    "description": "Throughput benchmark summary emitted by benchmarks.batch_ops.",
    "required": (
        "schema_id",
        "run_id",
        "timestamp",
        "mode",
        "batches",
        "batch_size",
        "points_processed",
        "elapsed_seconds",
    ),
}

__all__ = [
    "BENCHMARK_BATCH_JSON_SCHEMA",
    "BENCHMARK_BATCH_SCHEMA_ID",
    "BENCHMARK_BATCH_SCHEMA_VERSION",
    "BenchmarkBatchRecord",
    "RESIDUAL_SCOPE_CAP_SCHEMA",
    "RESIDUAL_SCOPE_CAP_SCHEMA_ID",
    "RESIDUAL_SCOPE_CAP_SCHEMA_VERSION",
    "RESIDUAL_GATE_PROFILE_SCHEMA",
    "RESIDUAL_GATE_PROFILE_SCHEMA_ID",
    "RESIDUAL_GATE_PROFILE_SCHEMA_VERSION",
    "RUNTIME_BREAKDOWN_SCHEMA",
    "RUNTIME_BREAKDOWN_SCHEMA_ID",
    "RUNTIME_BREAKDOWN_CHUNK_FIELDS",
    "RUNTIME_BREAKDOWN_FIELDNAMES",
    "runtime_breakdown_fieldnames",
    "BATCH_OPS_RESULT_SCHEMA",
    "BATCH_OPS_RESULT_SCHEMA_ID",
]
