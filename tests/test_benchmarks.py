import csv
import json
import numpy as np
import sys
import pytest
from types import SimpleNamespace

jax = pytest.importorskip("jax")

from benchmarks.batch_ops import BenchmarkResult, benchmark_delete, benchmark_insert, _write_result_artifact
from benchmarks.queries import _build_tree, benchmark_knn_latency, run_baseline_comparisons
from benchmarks.runtime_cli import runtime_from_args as _runtime_from_args
from benchmarks import runtime_breakdown
from covertreex import config as cx_config
from covertreex.baseline import has_gpboost_cover_tree
from covertreex.telemetry import (
    BENCHMARK_BATCH_SCHEMA_ID,
    BENCHMARK_BATCH_SCHEMA_VERSION,
    BenchmarkLogWriter,
    RUNTIME_BREAKDOWN_SCHEMA_ID,
    runtime_breakdown_fieldnames,
)
from tests.utils.datasets import gaussian_points


@pytest.fixture(autouse=True)
def _ensure_euclidean_metric(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COVERTREEX_METRIC", "euclidean")
    cx_config.reset_runtime_config_cache()
    yield
    cx_config.reset_runtime_config_cache()


def test_benchmark_insert_delete_smoke():
    tree, insert_result = benchmark_insert(
        dimension=3,
        batch_size=4,
        batches=2,
        seed=0,
    )
    assert tree.num_points == insert_result.points_processed
    assert insert_result.mode == "insert"
    assert insert_result.batches == 2

    tree, delete_result = benchmark_delete(
        tree,
        batch_size=2,
        batches=1,
        seed=1,
    )
    assert delete_result.mode == "delete"
    assert delete_result.batches == 1
    assert delete_result.points_processed == 2


def test_benchmark_knn_latency_smoke():
    _, result = benchmark_knn_latency(
        dimension=3,
        tree_points=32,
        query_count=4,
        k=2,
        batch_size=8,
        seed=0,
    )
    assert result.queries == 4
    assert result.k == 2
    assert result.latency_ms >= 0.0
    assert result.build_seconds is not None


def test_run_baseline_comparisons_sequential():
    rng = np.random.default_rng(0)
    points = gaussian_points(rng, 16, 3)
    queries = gaussian_points(rng, 4, 3)
    results = run_baseline_comparisons(points, queries, k=2, mode="sequential")
    assert len(results) == 1
    baseline = results[0]
    assert baseline.name == "sequential"
    assert baseline.latency_ms >= 0.0


def test_run_baseline_comparisons_gpboost():
    if not has_gpboost_cover_tree():
        pytest.skip("GPBoost baseline requires numba")
    rng = np.random.default_rng(1)
    points = gaussian_points(rng, 16, 3)
    queries = gaussian_points(rng, 4, 3)
    results = run_baseline_comparisons(points, queries, k=2, mode="gpboost")
    assert len(results) == 1
    baseline = results[0]
    assert baseline.name == "gpboost"
    assert baseline.latency_ms >= 0.0


def test_runtime_breakdown_csv_output(tmp_path, monkeypatch):
    csv_path = tmp_path / "metrics.csv"
    png_path = tmp_path / "plot.png"
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    argv = [
        "runtime_breakdown",
        "--dimension",
        "2",
        "--tree-points",
        "16",
        "--batch-size",
        "4",
        "--queries",
        "8",
        "--k",
        "2",
        "--seed",
        "5",
        "--output",
        str(png_path),
        "--csv-output",
        str(csv_path),
        "--skip-external",
        "--skip-gpboost",
        "--skip-jax",
        "--backend",
        "numpy",
        "--precision",
        "float64",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    runtime_breakdown.main()
    cx_config.reset_runtime_context()
    assert csv_path.exists()
    contents = csv_path.read_text().strip().splitlines()
    expected = ",".join(runtime_breakdown_fieldnames())
    assert contents[0] == expected
    reader = csv.DictReader(contents)
    rows = list(reader)
    assert rows
    assert any(row["label"].startswith("PCCT") for row in rows)
    assert all(row["schema_id"] == RUNTIME_BREAKDOWN_SCHEMA_ID for row in rows)
    assert all(row["run_id"] for row in rows)


def test_benchmark_log_writer_emits_json(tmp_path):
    log_path = tmp_path / "batches.jsonl"
    writer = BenchmarkLogWriter(
        str(log_path),
        run_id="test-run",
        runtime={"backend": "numpy"},
        metadata={"benchmark": "unit-test"},
    )
    try:
        _build_tree(
            dimension=2,
            tree_points=8,
            batch_size=4,
            seed=0,
            log_writer=writer,
        )
    finally:
        writer.close()

    contents = log_path.read_text().strip().splitlines()
    assert contents
    first_entry = json.loads(contents[0])
    assert first_entry["batch_index"] == 0
    assert first_entry["batch_event_index"] == 0
    assert first_entry["schema_id"] == BENCHMARK_BATCH_SCHEMA_ID
    assert first_entry["schema_version"] == BENCHMARK_BATCH_SCHEMA_VERSION
    assert first_entry["run_id"] == "test-run"
    assert "run_hash" in first_entry and first_entry["run_hash"]
    assert first_entry.get("runtime_digest")
    assert "seed_pack" in first_entry
    assert first_entry["runtime"]["backend"] == "numpy"
    assert first_entry["runtime_backend"] == "numpy"
    assert "traversal_ms" in first_entry
    assert "rss_bytes" in first_entry or "rss_delta_bytes" in first_entry


def test_batch_ops_result_artifact(tmp_path):
    path = tmp_path / "summary.json"
    result = BenchmarkResult(
        mode="insert",
        elapsed_seconds=1.23,
        batches=4,
        batch_size=8,
        points_processed=32,
        throughput_points_per_sec=26.0,
    )
    args = SimpleNamespace(dimension=3, seed=0, bootstrap_batches=2)
    runtime_snapshot = {"backend": "jax"}
    _write_result_artifact(
        path,
        run_id="batch-run",
        runtime_snapshot=runtime_snapshot,
        args=args,
        result=result,
    )
    payload = json.loads(path.read_text())
    assert payload["schema_id"].endswith("batch_ops_summary.v1")
    assert payload["run_id"] == "batch-run"
    assert payload["runtime"]["backend"] == "jax"
    assert payload["mode"] == "insert"
    assert payload["batches"] == 4


def _cli_args(**overrides):
    defaults = dict(
        metric="euclidean",
        residual_gate=None,
        residual_gate_lookup_path="lookup.json",
        residual_gate_margin=0.02,
        residual_gate_cap=0.0,
        residual_scope_caps=None,
        residual_scope_cap_default=None,
        batch_order=None,
        batch_order_seed=None,
        prefix_schedule=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_runtime_from_args_residual_lookup():
    args = _cli_args(
        metric="residual",
        residual_gate="lookup",
        residual_gate_lookup_path="caps.json",
        residual_gate_margin=0.1,
        residual_gate_cap=3.5,
        residual_scope_caps="caps.json",
        residual_scope_cap_default=0.75,
        batch_order="hilbert",
        batch_order_seed=42,
        prefix_schedule="doubling",
    )
    runtime = _runtime_from_args(args)
    assert runtime.metric == "residual_correlation"
    assert runtime.backend == "numpy"
    assert runtime.batch_order == "hilbert"
    assert runtime.batch_order_seed == 42
    assert runtime.prefix_schedule == "doubling"
    assert runtime.enable_sparse_traversal is True
    assert runtime.residual is not None
    residual = runtime.residual
    assert residual.gate1_enabled is True
    assert residual.lookup_path == "caps.json"
    assert residual.lookup_margin == 0.1
    assert residual.gate1_radius_cap == 3.5
    assert residual.scope_cap_path == "caps.json"
    assert residual.scope_cap_default == 0.75


def test_runtime_from_args_euclidean_defaults():
    runtime = _runtime_from_args(_cli_args())
    assert runtime.metric == "euclidean"
    assert runtime.backend is None
    assert runtime.residual is None
