from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from cli.pcct.runtime_config import runtime_from_args
from covertreex.algo import batch_delete, batch_insert
from covertreex.core.tree import PCCTree, get_runtime_backend
from covertreex.telemetry import (
    BATCH_OPS_RESULT_SCHEMA_ID,
    generate_run_id,
    resolve_artifact_path,
    timestamped_artifact,
)


@dataclass(frozen=True)
class BenchmarkResult:
    mode: Literal["insert", "delete"]
    elapsed_seconds: float
    batches: int
    batch_size: int
    points_processed: int
    throughput_points_per_sec: float


def _write_result_artifact(
    path: Path,
    *,
    run_id: str,
    runtime_snapshot: dict[str, Any],
    args: argparse.Namespace,
    result: BenchmarkResult,
) -> None:
    payload = {
        "schema_id": BATCH_OPS_RESULT_SCHEMA_ID,
        "run_id": run_id,
        "timestamp": time.time(),
        "mode": result.mode,
        "batches": result.batches,
        "batch_size": result.batch_size,
        "points_processed": result.points_processed,
        "elapsed_seconds": result.elapsed_seconds,
        "throughput_points_per_sec": result.throughput_points_per_sec,
        "parameters": {
            "dimension": args.dimension,
            "seed": args.seed,
            "bootstrap_batches": args.bootstrap_batches,
        },
        "runtime": runtime_snapshot,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _generate_batch(key: jax.Array, batch_size: int, dimension: int) -> jnp.ndarray:
    backend = get_runtime_backend()
    return jax.random.normal(key, (batch_size, dimension), dtype=backend.default_float)


def benchmark_insert(
    *,
    dimension: int,
    batch_size: int,
    batches: int,
    seed: int,
) -> Tuple[PCCTree, BenchmarkResult]:
    backend = get_runtime_backend()
    tree = PCCTree.empty(dimension=dimension, backend=backend)
    key = jax.random.PRNGKey(seed)
    start = time.perf_counter()
    for idx in range(batches):
        key, subkey = jax.random.split(key)
        batch = _generate_batch(subkey, batch_size, dimension)
        tree, _ = batch_insert(tree, batch, mis_seed=seed + idx)
    elapsed = time.perf_counter() - start
    points = batch_size * batches
    throughput = points / elapsed if elapsed > 0 else float("inf")
    return tree, BenchmarkResult(
        mode="insert",
        elapsed_seconds=elapsed,
        batches=batches,
        batch_size=batch_size,
        points_processed=points,
        throughput_points_per_sec=throughput,
    )


def benchmark_delete(
    base_tree: PCCTree,
    *,
    batch_size: int,
    batches: int,
    seed: int,
) -> Tuple[PCCTree, BenchmarkResult]:
    rng = np.random.default_rng(seed)
    tree = base_tree
    start = time.perf_counter()
    completed_batches = 0
    for idx in range(batches):
        if tree.num_points < batch_size:
            break
        indices = rng.choice(tree.num_points, size=batch_size, replace=False)
        tree, _ = batch_delete(tree, indices)
        completed_batches += 1
    elapsed = time.perf_counter() - start
    points = batch_size * completed_batches
    throughput = points / elapsed if elapsed > 0 else float("inf")
    return tree, BenchmarkResult(
        mode="delete",
        elapsed_seconds=elapsed,
        batches=completed_batches,
        batch_size=batch_size,
        points_processed=points,
        throughput_points_per_sec=throughput,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark batch insert/delete throughput for the PCCT implementation."
    )
    parser.add_argument(
        "mode",
        choices=("insert", "delete"),
        help="Operation to benchmark.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=8,
        help="Dimensionality of generated points.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of points per batch.",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=20,
        help="Number of batches to execute.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for data generation.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier embedded in telemetry outputs.",
    )
    parser.add_argument(
        "--bootstrap-batches",
        type=int,
        default=20,
        help="Initial insert batches used to populate the tree before delete benchmarks.",
    )
    parser.add_argument(
        "--log-json",
        type=str,
        default="",
        help="Optional path to write a JSON summary for the run (defaults to artifacts/benchmarks).",
    )
    parser.add_argument(
        "--no-log-json",
        action="store_true",
        help="Disable JSON summary emission.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cli_runtime = runtime_from_args(
        SimpleNamespace(
            metric="euclidean",
            backend="jax",
            precision="float64",
        )
    )
    runtime_snapshot = cli_runtime.describe()
    cli_runtime.activate()
    run_id = args.run_id or generate_run_id(prefix="batchops")
    if args.no_log_json:
        log_path = None
    elif args.log_json:
        log_path = resolve_artifact_path(args.log_json, category="benchmarks")
    else:
        log_path = timestamped_artifact(
            category="benchmarks",
            prefix=f"batch_ops_{run_id}",
            suffix=".json",
        )
    mode: Literal["insert", "delete"] = args.mode  # type: ignore[assignment]

    if mode == "insert":
        _, result = benchmark_insert(
            dimension=args.dimension,
            batch_size=args.batch_size,
            batches=args.batches,
            seed=args.seed,
        )
    else:
        tree, _ = benchmark_insert(
            dimension=args.dimension,
            batch_size=args.batch_size,
            batches=args.bootstrap_batches,
            seed=args.seed,
        )
        _, result = benchmark_delete(
            tree,
            batch_size=args.batch_size,
            batches=args.batches,
            seed=args.seed + 1,
        )

    print(
        f"{result.mode} | batches={result.batches} "
        f"batch_size={result.batch_size} "
        f"points={result.points_processed} "
        f"time={result.elapsed_seconds:.4f}s "
        f"throughput={result.throughput_points_per_sec:,.1f} pts/s"
    )
    if log_path:
        _write_result_artifact(
            log_path,
            run_id=run_id,
            runtime_snapshot=runtime_snapshot,
            args=args,
            result=result,
        )
        print(f"[batch_ops] wrote summary to {log_path} (run_id={run_id})")


if __name__ == "__main__":
    main()
