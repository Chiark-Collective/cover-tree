from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.random import Generator, default_rng

from covertreex import reset_residual_metric
from covertreex import config as cx_config
from covertreex.algo import batch_insert, batch_insert_prefix_doubling
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.metrics import build_residual_backend, configure_residual_correlation
from covertreex.queries.knn import knn
from covertreex.telemetry import (
    BenchmarkLogWriter,
    ResidualScopeCapRecorder,
    generate_run_id,
    resolve_artifact_path,
    timestamped_artifact,
)
from cli.runtime import runtime_from_args
from covertreex.baseline import (
    BaselineCoverTree,
    ExternalCoverTreeBaseline,
    GPBoostCoverTreeBaseline,
    has_external_cover_tree,
    has_gpboost_cover_tree,
)
from tests.utils.datasets import gaussian_points


@dataclass(frozen=True)
class QueryBenchmarkResult:
    elapsed_seconds: float
    queries: int
    k: int
    latency_ms: float
    queries_per_second: float
    build_seconds: float | None = None


@dataclass(frozen=True)
class BaselineComparison:
    name: str
    build_seconds: float
    elapsed_seconds: float
    latency_ms: float
    queries_per_second: float


def _resolve_backend() -> TreeBackend:
    runtime = cx_config.runtime_config()
    if runtime.backend == "jax":
        return TreeBackend.jax(precision=runtime.precision)
    if runtime.backend == "numpy":
        return TreeBackend.numpy(precision=runtime.precision)
    raise NotImplementedError(f"Backend '{runtime.backend}' is not supported yet.")


def _generate_backend_points(
    rng: Generator,
    count: int,
    dimension: int,
    *,
    backend: TreeBackend,
) -> np.ndarray:
    samples = gaussian_points(rng, count, dimension, dtype=np.float64)
    return backend.asarray(samples, dtype=backend.default_float)


def _resolve_artifact_arg(path: str | None) -> str | None:
    if not path:
        return None
    return str(resolve_artifact_path(path, category="benchmarks"))


def _build_tree(
    *,
    dimension: int,
    tree_points: int,
    batch_size: int,
    seed: int,
    prebuilt_points: np.ndarray | None = None,
    log_writer: BenchmarkLogWriter | None = None,
    scope_cap_recorder: "ResidualScopeCapRecorder | None" = None,
    build_mode: str = "batch",
) -> Tuple[PCCTree, np.ndarray, float]:
    backend = _resolve_backend()
    tree = PCCTree.empty(dimension=dimension, backend=backend)

    if build_mode == "prefix":
        if prebuilt_points is not None:
            points_np = np.asarray(prebuilt_points, dtype=np.float64, copy=False)
        else:
            rng = default_rng(seed)
            points_np = gaussian_points(rng, tree_points, dimension, dtype=np.float64)
        batch = backend.asarray(points_np, dtype=backend.default_float)
        start = time.perf_counter()
        tree, prefix_result = batch_insert_prefix_doubling(
            tree,
            batch,
            backend=backend,
            mis_seed=seed,
            shuffle_seed=seed,
        )
        build_seconds = time.perf_counter() - start
        if log_writer is not None:
            runtime = cx_config.runtime_config()
            schedule = runtime.prefix_schedule
            for group_index, group in enumerate(prefix_result.groups):
                plan = group.plan
                if hasattr(plan.traversal, "parents"):
                    group_size = int(plan.traversal.parents.shape[0])
                else:
                    group_size = int(plan.traversal.levels.shape[0])
                extra = {
                    "prefix_group_index": group_index,
                    "prefix_factor": float(group.prefix_factor or 0.0),
                    "prefix_domination_ratio": float(group.domination_ratio or 0.0),
                    "prefix_schedule": schedule,
                }
                log_writer.record_batch(
                    batch_index=group_index,
                    batch_size=group_size,
                    plan=plan,
                    extra=extra,
                )
                if scope_cap_recorder is not None:
                    scope_cap_recorder.capture(plan)
        else:
            if scope_cap_recorder is not None:
                for group in prefix_result.groups:
                    scope_cap_recorder.capture(group.plan)
        return tree, points_np, build_seconds

    start = time.perf_counter()
    buffers: List[np.ndarray] = []
    idx = 0

    if prebuilt_points is not None:
        points_np = np.asarray(prebuilt_points, dtype=np.float64, copy=False)
        total = points_np.shape[0]
        while idx * batch_size < total:
            start_idx = idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_np = points_np[start_idx:end_idx]
            batch = backend.asarray(batch_np, dtype=backend.default_float)
            tree, plan = batch_insert(tree, batch, mis_seed=seed + idx)
            if log_writer is not None:
                log_writer.record_batch(
                    batch_index=idx,
                    batch_size=int(batch_np.shape[0]),
                    plan=plan,
                )
            if scope_cap_recorder is not None:
                scope_cap_recorder.capture(plan)
            buffers.append(np.asarray(batch))
            idx += 1
    else:
        rng = default_rng(seed)
        remaining = tree_points
        while remaining > 0:
            current = min(batch_size, remaining)
            batch = _generate_backend_points(
                rng,
                current,
                dimension,
                backend=backend,
            )
            tree, plan = batch_insert(tree, batch, mis_seed=seed + idx)
            if log_writer is not None:
                log_writer.record_batch(
                    batch_index=idx,
                    batch_size=current,
                    plan=plan,
                )
            if scope_cap_recorder is not None:
                scope_cap_recorder.capture(plan)
            buffers.append(np.asarray(batch))
            remaining -= current
            idx += 1

    build_seconds = time.perf_counter() - start
    if buffers:
        points_np = np.concatenate(buffers, axis=0)
    else:
        points_np = np.empty((0, dimension), dtype=np.float64)
    return tree, points_np, build_seconds


def benchmark_knn_latency(
    *,
    dimension: int,
    tree_points: int,
    query_count: int,
    k: int,
    batch_size: int,
    seed: int,
    prebuilt_points: np.ndarray | None = None,
    prebuilt_tree: PCCTree | None = None,
    prebuilt_queries: np.ndarray | None = None,
    build_seconds: float | None = None,
    log_writer: BenchmarkLogWriter | None = None,
    scope_cap_recorder: "ResidualScopeCapRecorder | None" = None,
    build_mode: str = "batch",
) -> Tuple[PCCTree, QueryBenchmarkResult]:
    tree_build_seconds: float | None = None
    if prebuilt_tree is None:
        tree, _, tree_build_seconds = _build_tree(
            dimension=dimension,
            tree_points=tree_points,
            batch_size=batch_size,
            seed=seed,
            prebuilt_points=prebuilt_points,
            log_writer=log_writer,
            scope_cap_recorder=scope_cap_recorder,
            build_mode=build_mode,
        )
    else:
        tree = prebuilt_tree
        tree_build_seconds = build_seconds

    if scope_cap_recorder is not None and tree_build_seconds is not None:
        scope_cap_recorder.annotate(tree_build_seconds=tree_build_seconds)

    backend = tree.backend
    if prebuilt_queries is None:
        query_rng = default_rng(seed + 1)
        queries = _generate_backend_points(
            query_rng,
            query_count,
            dimension,
            backend=backend,
        )
    else:
        queries = backend.asarray(
            prebuilt_queries, dtype=backend.default_float
        )
    start = time.perf_counter()
    knn(tree, queries, k=k)
    elapsed = time.perf_counter() - start
    qps = query_count / elapsed if elapsed > 0 else float("inf")
    latency = (elapsed / query_count) * 1e3 if query_count else 0.0
    return tree, QueryBenchmarkResult(
        elapsed_seconds=elapsed,
        queries=query_count,
        k=k,
        latency_ms=latency,
        queries_per_second=qps,
        build_seconds=tree_build_seconds,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark batched k-NN query latency for the PCCT implementation."
    )
    parser.add_argument("--dimension", type=int, default=8, help="Dimensionality of points.")
    parser.add_argument(
        "--tree-points",
        type=int,
        default=16_384,
        help="Number of points to populate the tree with before querying.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size used while constructing the tree.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=1024,
        help="Number of query points to evaluate.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of neighbours to request.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier propagated to telemetry artifacts (default: auto-generated).",
    )
    parser.add_argument(
        "--metric",
        choices=("euclidean", "residual"),
        default="euclidean",
        help="Distance metric to benchmark (configures residual caches when 'residual').",
    )
    parser.add_argument(
        "--residual-lengthscale",
        type=float,
        default=1.0,
        help="RBF kernel lengthscale for synthetic residual caches.",
    )
    parser.add_argument(
        "--residual-variance",
        type=float,
        default=1.0,
        help="RBF kernel variance for synthetic residual caches.",
    )
    parser.add_argument(
        "--residual-inducing",
        type=int,
        default=512,
        help="Number of inducing points to use when building residual caches.",
    )
    parser.add_argument(
        "--residual-chunk-size",
        type=int,
        default=512,
        help="Chunk size for residual kernel streaming.",
    )
    parser.add_argument(
        "--baseline",
        choices=("none", "sequential", "gpboost", "external", "both", "all"),
        default="none",
        help=(
            "Include baseline comparisons. Install '.[baseline]' for the external library and "
            "'numba' extra for the GPBoost baseline. Options: 'sequential', 'gpboost', "
            "'external', 'both' (sequential + external), 'all' (sequential + gpboost + external)."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write per-batch telemetry as JSON lines to the specified path.",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable JSONL batch telemetry output (enabled by default).",
    )
    parser.add_argument(
        "--batch-order",
        choices=("natural", "random", "hilbert"),
        default=None,
        help="Override COVERTREEX_BATCH_ORDER for this run.",
    )
    parser.add_argument(
        "--batch-order-seed",
        type=int,
        default=None,
        help="Override COVERTREEX_BATCH_ORDER_SEED for this run.",
    )
    parser.add_argument(
        "--prefix-schedule",
        choices=("doubling", "adaptive"),
        default=None,
        help="Override COVERTREEX_PREFIX_SCHEDULE for this run.",
    )
    parser.add_argument(
        "--build-mode",
        choices=("batch", "prefix"),
        default="batch",
        help="Choose the tree construction strategy (standard batch inserts or prefix doubling).",
    )
    parser.add_argument(
        "--residual-gate",
        choices=("off", "lookup"),
        default=None,
        help="Residual-only: automatically configure Gate-1 (e.g. 'lookup' wires sparse traversal + lookup table).",
    )
    parser.add_argument(
        "--residual-gate-lookup-path",
        type=str,
        default="docs/data/residual_gate_profile_diag0.json",
        help="Lookup JSON used when --residual-gate=lookup (default: diag0 profile).",
    )
    parser.add_argument(
        "--residual-gate-margin",
        type=float,
        default=0.02,
        help="Safety margin added to lookup thresholds when --residual-gate=lookup.",
    )
    parser.add_argument(
        "--residual-gate-cap",
        type=float,
        default=0.0,
        help="Optional radius cap passed to the lookup preset (0 keeps existing env/default).",
    )
    parser.add_argument(
        "--residual-scope-caps",
        type=str,
        default=None,
        help="Residual-only: JSON file describing per-level scope radius caps.",
    )
    parser.add_argument(
        "--residual-scope-cap-default",
        type=float,
        default=None,
        help="Residual-only: fallback radius cap applied when no per-level cap matches.",
    )
    parser.add_argument(
        "--residual-scope-cap-output",
        type=str,
        default=None,
        help="Residual-only: write derived per-level scope caps to this JSON file.",
    )
    parser.add_argument(
        "--residual-scope-cap-percentile",
        type=float,
        default=0.5,
        help="Quantile (0-1) used when deriving new scope caps (default: median).",
    )
    parser.add_argument(
        "--residual-scope-cap-margin",
        type=float,
        default=0.05,
        help="Safety margin added to the sampled percentile when deriving scope caps.",
    )
    return parser.parse_args()


def _run_sequential_baseline(points: np.ndarray, queries: np.ndarray, *, k: int) -> BaselineComparison:
    start_build = time.perf_counter()
    tree = BaselineCoverTree.from_points(points)
    build_seconds = time.perf_counter() - start_build
    start = time.perf_counter()
    tree.knn(queries, k=k, return_distances=False)
    elapsed = time.perf_counter() - start
    qps = queries.shape[0] / elapsed if elapsed > 0 else float("inf")
    latency = (elapsed / queries.shape[0]) * 1e3 if queries.shape[0] else 0.0
    return BaselineComparison(
        name="sequential",
        build_seconds=build_seconds,
        elapsed_seconds=elapsed,
        latency_ms=latency,
        queries_per_second=qps,
    )


def _run_external_baseline(points: np.ndarray, queries: np.ndarray, *, k: int) -> BaselineComparison:
    if not has_external_cover_tree():
        raise RuntimeError("External cover tree baseline requested but `covertree` is not available.")
    start_build = time.perf_counter()
    tree = ExternalCoverTreeBaseline.from_points(points)
    build_seconds = time.perf_counter() - start_build
    start = time.perf_counter()
    tree.knn(queries, k=k, return_distances=False)
    elapsed = time.perf_counter() - start
    qps = queries.shape[0] / elapsed if elapsed > 0 else float("inf")
    latency = (elapsed / queries.shape[0]) * 1e3 if queries.shape[0] else 0.0
    return BaselineComparison(
        name="external",
        build_seconds=build_seconds,
        elapsed_seconds=elapsed,
        latency_ms=latency,
        queries_per_second=qps,
    )


def _run_gpboost_baseline(points: np.ndarray, queries: np.ndarray, *, k: int) -> BaselineComparison:
    if not has_gpboost_cover_tree():
        raise RuntimeError(
            "GPBoost cover tree baseline requested but 'numba' extra is not installed."
        )
    start_build = time.perf_counter()
    tree = GPBoostCoverTreeBaseline.from_points(points)
    build_seconds = time.perf_counter() - start_build
    start = time.perf_counter()
    tree.knn(queries, k=k, return_distances=False)
    elapsed = time.perf_counter() - start
    qps = queries.shape[0] / elapsed if elapsed > 0 else float("inf")
    latency = (elapsed / queries.shape[0]) * 1e3 if queries.shape[0] else 0.0
    return BaselineComparison(
        name="gpboost",
        build_seconds=build_seconds,
        elapsed_seconds=elapsed,
        latency_ms=latency,
        queries_per_second=qps,
    )


def run_baseline_comparisons(
    points: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    mode: str,
) -> List[BaselineComparison]:
    queries = np.asarray(queries, dtype=float)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    results: List[BaselineComparison] = []
    if mode in ("sequential", "both", "all"):
        results.append(_run_sequential_baseline(points, queries, k=k))
    if mode in ("gpboost", "all"):
        results.append(_run_gpboost_baseline(points, queries, k=k))
    if mode in ("external", "both", "all"):
        results.append(_run_external_baseline(points, queries, k=k))
    return results


def main() -> None:
    args = _parse_args()
    if args.residual_gate and args.metric != "residual":
        raise ValueError("--residual-gate presets are only supported when --metric residual is selected.")
    cli_runtime = runtime_from_args(args)
    runtime_snapshot = cli_runtime.describe()
    cli_runtime.activate()
    run_id = args.run_id or generate_run_id()
    log_metadata = {
        "benchmark": "cli.queries",
        "dimension": args.dimension,
        "tree_points": args.tree_points,
        "batch_size": args.batch_size,
        "queries": args.queries,
        "k": args.k,
        "metric": args.metric,
        "build_mode": args.build_mode,
        "baseline": args.baseline,
    }
    log_writer: BenchmarkLogWriter | None = None
    scope_cap_recorder: ResidualScopeCapRecorder | None = None

    try:
        log_path: str | None
        if args.no_log_file:
            log_path = None
        elif args.log_file:
            log_path = _resolve_artifact_arg(args.log_file)
        else:
            log_path = str(
                timestamped_artifact(
                    category="benchmarks",
                    prefix=f"queries_{run_id}",
                    suffix=".jsonl",
                )
            )
        if log_path:
            print(f"[queries] writing batch telemetry to {log_path}")
            log_writer = BenchmarkLogWriter(
                log_path,
                run_id=run_id,
                runtime=runtime_snapshot,
                metadata=log_metadata,
            )

        scope_cap_output = _resolve_artifact_arg(args.residual_scope_cap_output)
        if args.metric == "residual" and scope_cap_output:
            runtime_config = cx_config.runtime_config()
            scope_cap_recorder = ResidualScopeCapRecorder(
                output=scope_cap_output,
                percentile=args.residual_scope_cap_percentile,
                margin=args.residual_scope_cap_margin,
                radius_floor=runtime_config.residual_radius_floor,
            )
            scope_cap_recorder.annotate(
                run_id=run_id,
                log_file=log_path,
                tree_points=args.tree_points,
                batch_size=args.batch_size,
                scope_chunk_target=runtime_config.scope_chunk_target,
                scope_chunk_max_segments=runtime_config.scope_chunk_max_segments,
                residual_scope_cap_default=args.residual_scope_cap_default,
                seed=args.seed,
                build_mode=args.build_mode,
            )

        point_rng = default_rng(args.seed)
        points_np = gaussian_points(point_rng, args.tree_points, args.dimension, dtype=np.float64)
        query_rng = default_rng(args.seed + 1)
        queries_np = gaussian_points(query_rng, args.queries, args.dimension, dtype=np.float64)

        if args.metric == "residual":
            residual_backend = build_residual_backend(
                points_np,
                seed=args.seed,
                inducing_count=args.residual_inducing,
                variance=args.residual_variance,
                lengthscale=args.residual_lengthscale,
                chunk_size=args.residual_chunk_size,
            )
            configure_residual_correlation(residual_backend)

        tree, result = benchmark_knn_latency(
            dimension=args.dimension,
            tree_points=args.tree_points,
            query_count=args.queries,
            k=args.k,
            batch_size=args.batch_size,
            seed=args.seed,
            prebuilt_points=points_np,
            prebuilt_queries=queries_np,
            log_writer=log_writer,
            scope_cap_recorder=scope_cap_recorder,
            build_mode=args.build_mode,
        )

        print(
            f"pcct | build={result.build_seconds:.4f}s "
            f"queries={result.queries} k={result.k} "
            f"time={result.elapsed_seconds:.4f}s "
            f"latency={result.latency_ms:.4f}ms "
            f"throughput={result.queries_per_second:,.1f} q/s"
        )

        if args.baseline != "none":
            baseline_results = run_baseline_comparisons(
                points_np,
                queries_np,
                k=args.k,
                mode=args.baseline,
            )
            for baseline in baseline_results:
                slowdown = (
                    baseline.latency_ms / result.latency_ms if result.latency_ms else float("inf")
                )

                print(
                    f"baseline[{baseline.name}] | build={baseline.build_seconds:.4f}s "
                    f"time={baseline.elapsed_seconds:.4f}s "
                    f"latency={baseline.latency_ms:.4f}ms "
                    f"throughput={baseline.queries_per_second:,.1f} q/s "
                    f"slowdown={slowdown:.3f}x"
                )
    finally:
        reset_residual_metric()
        cx_config.reset_runtime_context()
        if scope_cap_recorder is not None:
            scope_cap_recorder.dump()
        if log_writer is not None:
            log_writer.close()


__all__ = [
    "BaselineComparison",
    "QueryBenchmarkResult",
    "_build_tree",
    "benchmark_knn_latency",
    "run_baseline_comparisons",
    "main",
]


if __name__ == "__main__":
    main()
