"""
Compare residual-correlation performance between the PCCT CLI (NumPy/Numba path)
and the Rust backend for float32/float64 at n=50k, d=3.

The CLI side uses `cli.pcct.query` with --metric residual so we match the
documented telemetry pathway instead of the slower Python batch_insert helper.
"""

from __future__ import annotations

import sys
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

# Ensure repository root is on sys.path for local module imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np

import covertreex
import covertreex_backend
from covertreex.core.tree import PCCTree
from covertreex.metrics.residual.host_backend import build_residual_backend
from covertreex.metrics.residual.core import configure_residual_correlation
from covertreex.algo.batch_insert import batch_insert
from covertreex.queries.knn import knn
from tests.utils.datasets import gaussian_points  # noqa: E402


DATASET_SIZE = 50_000
DIMENSION = 3
K = 50
QUERIES = 1_024
BATCH_SIZE = 512
SEED = 0
RANK = 16


@dataclass
class BenchmarkResult:
    build_seconds: float
    query_seconds: float
    qps: float


def run_cli(dtype_label: str) -> BenchmarkResult:
    dtype = np.float32 if dtype_label == "float32" else np.float64
    rng = np.random.default_rng(SEED)
    points = gaussian_points(rng, DATASET_SIZE, DIMENSION, dtype=np.float64).astype(dtype)
    query_rng = np.random.default_rng(SEED + 1)
    queries = gaussian_points(query_rng, QUERIES, DIMENSION, dtype=np.float64).astype(dtype)

    # Configure runtime for residual metric with requested precision.
    runtime_cfg = covertreex.config.runtime_config()
    cfg_residual = dataclasses.replace(
        runtime_cfg,
        metric="residual_correlation",
        precision=dtype_label,
        enable_numba=True,
        enable_sparse_traversal=True,
        residual_use_static_euclidean_tree=False,
    )
    covertreex.config.configure_runtime(cfg_residual)

    # Build residual backend and register it globally.
    backend = build_residual_backend(
        points.astype(np.float64),
        seed=SEED,
        inducing_count=512,
        variance=1.0,
        lengthscale=1.0,
        chunk_size=512,
    )
    object.__setattr__(backend, "rbf_variance", 1.0)
    object.__setattr__(backend, "rbf_lengthscale", np.ones(DIMENSION, dtype=np.float32))
    configure_residual_correlation(backend)

    tree = PCCTree.empty(dimension=DIMENSION)
    start = time.perf_counter()
    tree, _ = batch_insert(tree, points.astype(dtype))
    build_seconds = time.perf_counter() - start

    start = time.perf_counter()
    knn(tree, queries.astype(dtype), k=K)
    query_seconds = time.perf_counter() - start
    qps = QUERIES / query_seconds if query_seconds > 0 else float("nan")

    return BenchmarkResult(build_seconds=build_seconds, query_seconds=query_seconds, qps=qps)


def run_rust(dtype: np.dtype) -> BenchmarkResult:
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(SEED)
    X = gaussian_points(rng, DATASET_SIZE, DIMENSION, dtype=np.float64).astype(dtype)
    V = rng.normal(size=(DATASET_SIZE, RANK)).astype(dtype)
    p_diag = rng.uniform(0.1, 1.0, size=DATASET_SIZE).astype(dtype)

    rbf_var = 1.0
    rbf_ls = np.ones(DIMENSION, dtype=dtype)

    dummy_points = np.empty((0, 1), dtype=dtype)
    dummy_parents = np.empty(0, dtype=np.int64)
    dummy_children = np.empty(0, dtype=np.int64)
    dummy_next = np.empty(0, dtype=np.int64)
    dummy_levels = np.empty(0, dtype=np.int32)

    tree = covertreex_backend.CoverTreeWrapper(
        dummy_points, dummy_parents, dummy_children, dummy_next, dummy_levels, -20, 20
    )

    indices_all = np.arange(DATASET_SIZE, dtype=dtype).reshape(-1, 1)

    t0 = time.perf_counter()
    tree.insert_residual(indices_all, V, p_diag, X, float(rbf_var), rbf_ls)
    build_seconds = time.perf_counter() - t0

    node_to_dataset = np.arange(DATASET_SIZE, dtype=np.int64).tolist()
    query_indices = np.arange(QUERIES, dtype=np.int64)

    t0 = time.perf_counter()
    tree.knn_query_residual(
        query_indices,
        node_to_dataset,
        V,
        p_diag,
        X,
        float(rbf_var),
        rbf_ls,
        K,
    )
    query_seconds = time.perf_counter() - t0
    qps = QUERIES / query_seconds if query_seconds > 0 else float("nan")
    return BenchmarkResult(build_seconds=build_seconds, query_seconds=query_seconds, qps=qps)


def main() -> None:
    dtype_labels = ["float32", "float64"]
    results: Dict[Tuple[str, str], BenchmarkResult] = {}

    for label in dtype_labels:
        print(f"\n===== dtype={label} =====")

        cli_result = run_cli(label)
        print(
            f"[pcct cli] build={cli_result.build_seconds:.4f}s "
            f"query={cli_result.query_seconds:.4f}s ({cli_result.qps:,.1f} q/s)"
        )
        results[(label, "cli")] = cli_result

        rust_result = run_rust(np.float32 if label == "float32" else np.float64)
        print(
            f"[rust] build={rust_result.build_seconds:.4f}s "
            f"query={rust_result.query_seconds:.4f}s ({rust_result.qps:,.1f} q/s)"
        )
        results[(label, "rust")] = rust_result

    print("\n===== summary =====")
    print(f"{'dtype':<10} | {'cli_build':>10} | {'cli_qps':>10} | {'rust_build':>10} | {'rust_qps':>10} | {'speedup_q':>10}")
    print("-" * 72)
    for label in dtype_labels:
        cli_res = results[(label, "cli")]
        rust_res = results[(label, "rust")]
        speedup = rust_res.qps / cli_res.qps if cli_res.qps > 0 else float("nan")
        print(f"{label:<10} | {cli_res.build_seconds:.4f} | {cli_res.qps:,.1f} | {rust_res.build_seconds:.4f} | {rust_res.qps:,.1f} | {speedup:.2f}")


if __name__ == "__main__":
    main()
