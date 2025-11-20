from __future__ import annotations

from contextlib import contextmanager

import json

from typer.testing import CliRunner

from cli.pcct import app as pcct_app
from cli.pcct.support.benchmark_utils import QueryBenchmarkResult


class _DummyRun:
    runtime_snapshot = {}
    thread_snapshot = {}
    telemetry_view = None
    log_writer = None
    scope_cap_recorder = None
    context = None
    log_path = None


def _fake_benchmark_run():
    @contextmanager
    def _manager(options, **kwargs):
        yield _DummyRun()

    return _manager


def test_pcct_profile_list_displays_profiles() -> None:
    runner = CliRunner()
    result = runner.invoke(pcct_app, ["profile", "list"])
    assert result.exit_code == 0
    assert "default" in result.stdout


def test_pcct_profile_describe_outputs_json() -> None:
    runner = CliRunner()
    result = runner.invoke(pcct_app, ["profile", "describe", "default", "--format", "json"])
    assert result.exit_code == 0
    assert '"metadata"' in result.stdout
    assert '"runtime"' in result.stdout


def test_pcct_doctor_reports_environment(tmp_path) -> None:
    runner = CliRunner()
    env = {"COVERTREEX_ARTIFACT_ROOT": str(tmp_path)}
    result = runner.invoke(pcct_app, ["doctor", "--profile", "default"], env=env)
    assert result.exit_code == 0
    assert "artifact root" in result.stdout


def test_pcct_query_command_invokes_runner(monkeypatch) -> None:
    runner = CliRunner()
    recorded = {}

    def fake_execute(options, run) -> QueryBenchmarkResult:
        recorded["metric"] = options.metric
        recorded["dimension"] = options.dimension
        return QueryBenchmarkResult(
            elapsed_seconds=0.1,
            queries=options.queries,
            k=options.k,
            latency_ms=1.0,
            queries_per_second=1000.0,
            build_seconds=0.05,
        )

    monkeypatch.setattr("cli.pcct.query_cli.execute_query_benchmark", fake_execute)
    monkeypatch.setattr("cli.pcct.query_cli.benchmark_run", _fake_benchmark_run())

    result = runner.invoke(
        pcct_app,
        ["query", "--dimension", "4", "--tree-points", "8", "--queries", "4", "--k", "1", "--metric", "euclidean"],
    )
    assert result.exit_code == 0
    assert recorded["metric"] == "euclidean"
    assert recorded["dimension"] == 4


def test_pcct_query_manual_runtime_knobs(monkeypatch) -> None:
    runner = CliRunner()
    recorded = {}

    def fake_execute(options, run) -> QueryBenchmarkResult:
        recorded["backend"] = options.backend
        recorded["devices"] = options.devices
        recorded["diagnostics"] = options.diagnostics
        recorded["scope_chunk_max_segments"] = options.scope_chunk_max_segments
        recorded["scope_conflict_buffer_reuse"] = options.scope_conflict_buffer_reuse
        recorded["prefix_schedule"] = options.prefix_schedule
        return QueryBenchmarkResult(
            elapsed_seconds=0.1,
            queries=options.queries,
            k=options.k,
            latency_ms=1.0,
            queries_per_second=1000.0,
            build_seconds=0.05,
        )

    monkeypatch.setattr("cli.pcct.query_cli.execute_query_benchmark", fake_execute)
    monkeypatch.setattr("cli.pcct.query_cli.benchmark_run", _fake_benchmark_run())

    result = runner.invoke(
        pcct_app,
        [
            "query",
            "--metric",
            "euclidean",
            "--backend",
            "jax",
            "--device",
            "cpu0",
            "--device",
            "cpu1",
            "--enable-diagnostics",
            "--scope-chunk-max-segments",
            "4",
            "--no-scope-conflict-buffer-reuse",
            "--prefix-schedule",
            "adaptive",
        ],
    )
    assert result.exit_code == 0
    assert recorded["backend"] == "jax"
    assert recorded["devices"] == ("cpu0", "cpu1")
    assert recorded["diagnostics"] is True
    assert recorded["scope_chunk_max_segments"] == 4
    assert recorded["scope_conflict_buffer_reuse"] is False
    assert recorded["prefix_schedule"] == "adaptive"


def test_pcct_query_requires_profile_for_overrides(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("cli.pcct.query_cli.benchmark_run", _fake_benchmark_run())
    monkeypatch.setattr(
        "cli.pcct.query_cli.execute_query_benchmark",
        lambda *args, **kwargs: QueryBenchmarkResult(
            elapsed_seconds=0.1, queries=0, k=0, latency_ms=0.0, queries_per_second=0.0, build_seconds=None
        ),
    )

    result = runner.invoke(pcct_app, ["query", "--metric", "euclidean", "--set", "diagnostics.enabled=true"])
    assert result.exit_code != 0
    assert "--set overrides require --profile" in result.stderr


def test_pcct_build_command_invokes_builder(monkeypatch) -> None:
    runner = CliRunner()
    recorded = {}

    def fake_build_tree(**kwargs):
        recorded.update(kwargs)
        return object(), None, 0.25

    monkeypatch.setattr("cli.pcct.build_cli._build_tree", fake_build_tree)
    monkeypatch.setattr("cli.pcct.build_cli.benchmark_run", _fake_benchmark_run())

    result = runner.invoke(pcct_app, ["build", "--dimension", "4", "--tree-points", "16"])
    assert result.exit_code == 0
    assert recorded["dimension"] == 4
    assert recorded["tree_points"] == 16


def test_pcct_benchmark_repeats_runs(monkeypatch) -> None:
    runner = CliRunner()
    seeds: list[int] = []

    def fake_execute(options, run) -> QueryBenchmarkResult:
        seeds.append(options.seed)
        return QueryBenchmarkResult(
            elapsed_seconds=0.1,
            queries=options.queries,
            k=options.k,
            latency_ms=1.0 + options.seed,
            queries_per_second=1000.0,
            build_seconds=0.05,
        )

    monkeypatch.setattr("cli.pcct.benchmark_cli.execute_query_benchmark", fake_execute)
    monkeypatch.setattr("cli.pcct.benchmark_cli.benchmark_run", _fake_benchmark_run())

    result = runner.invoke(
        pcct_app,
        [
            "benchmark",
            "--dimension",
            "4",
            "--tree-points",
            "8",
            "--queries",
            "4",
            "--k",
            "1",
            "--metric",
            "euclidean",
            "--repeat",
            "2",
            "--seed",
            "3",
            "--seed-step",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert seeds == [3, 5]


def test_pcct_telemetry_render_command(tmp_path) -> None:
    log_path = tmp_path / "telemetry.jsonl"
    records = [
        {
            "schema_id": "covertreex.benchmark_batch.v2",
            "schema_version": 2,
            "run_id": "demo",
            "run_hash": "hash-a",
            "timestamp": 1.0,
            "batch_event_index": 0,
            "batch_index": 0,
            "batch_size": 4,
            "runtime": {"backend": "numpy", "metric": "residual"},
            "metadata": {"benchmark": "unit-test"},
            "traversal_ms": 10.0,
            "conflict_graph_ms": 5.0,
            "mis_ms": 1.0,
            "traversal_whitened_block_pairs": 100.0,
            "traversal_kernel_provider_pairs": 50.0,
            "traversal_whitened_block_ms": 2.0,
            "traversal_kernel_provider_ms": 1.0,
            "traversal_gate1_candidates": 32,
            "traversal_gate1_pruned": 8,
            "traversal_gate1_kept": 24,
            "conflict_pairwise_reused": 1,
        },
        {
            "schema_id": "covertreex.benchmark_batch.v2",
            "schema_version": 2,
            "run_id": "demo",
            "run_hash": "hash-a",
            "timestamp": 2.0,
            "batch_event_index": 1,
            "batch_index": 1,
            "batch_size": 4,
            "runtime": {"backend": "numpy", "metric": "residual"},
            "metadata": {"benchmark": "unit-test"},
            "traversal_ms": 12.0,
            "conflict_graph_ms": 4.0,
            "mis_ms": 1.5,
            "traversal_whitened_block_pairs": 80.0,
            "traversal_kernel_provider_pairs": 20.0,
            "traversal_whitened_block_ms": 2.5,
            "traversal_kernel_provider_ms": 0.5,
            "traversal_gate1_candidates": 30,
            "traversal_gate1_pruned": 5,
            "traversal_gate1_kept": 25,
            "conflict_pairwise_reused": 1,
        },
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    runner = CliRunner()
    result_json = runner.invoke(pcct_app, ["telemetry", "render", str(log_path), "--format", "json"])
    assert result_json.exit_code == 0
    assert '"run_id": "demo"' in result_json.stdout

    result_md = runner.invoke(
        pcct_app,
        ["telemetry", "render", str(log_path), "--format", "md", "--show", "fields"],
    )
    assert result_md.exit_code == 0
    assert "Measurement fields" in result_md.stdout
    assert "traversal_ms" in result_md.stdout


def test_pcct_query_seeds_override_produces_stable_hash(tmp_path) -> None:
    runner = CliRunner()
    log_one = tmp_path / "run_one.jsonl"
    log_two = tmp_path / "run_two.jsonl"
    common_args = [
        "query",
        "--profile",
        "default",
        "--dimension",
        "2",
        "--tree-points",
        "16",
        "--batch-size",
        "8",
        "--queries",
        "8",
        "--k",
        "1",
        "--baseline",
        "none",
        "--global-seed",
        "123",
        "--residual-grid-seed",
        "777",
        "--seed",
        "0",
    ]

    result_one = runner.invoke(pcct_app, common_args + ["--log-file", str(log_one)])
    assert result_one.exit_code == 0, result_one.output
    result_two = runner.invoke(pcct_app, common_args + ["--log-file", str(log_two)])
    assert result_two.exit_code == 0, result_two.output

    first_payload = json.loads(log_one.read_text(encoding="utf-8").splitlines()[0])
    second_payload = json.loads(log_two.read_text(encoding="utf-8").splitlines()[0])
    assert first_payload["run_hash"] == second_payload["run_hash"]
    assert first_payload["seed_pack"].get("residual_grid") == 777
