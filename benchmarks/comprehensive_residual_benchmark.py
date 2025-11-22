"""
Comprehensive Benchmark: 50k Points, Residual Correlation.
Comparing:
1. Python/Numba: Static Euclidean Build + Residual Query (Baseline)
2. Rust: Static Euclidean Build + Residual Query (Hybrid)
3. Rust: Dynamic Residual Build + Residual Query (Target Optimization)

Produces a table of results.
"""

from __future__ import annotations

import sys
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import numpy as np

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import covertreex
from covertreex.core.tree import PCCTree
from covertreex.metrics.residual.host_backend import build_residual_backend
from covertreex.metrics.residual.core import configure_residual_correlation
from covertreex.algo.batch_insert import batch_insert
from covertreex.queries.knn import knn
from tests.utils.datasets import gaussian_points

# Configuration
DATASET_SIZE = 50_000
DIMENSION = 3
K = 50
QUERIES = 2_000
BATCH_SIZE = 512
SEED = 42
WARMUP_SIZE = 1_000

logging.basicConfig(level=logging.WARNING)

@dataclass
class Result:
    variant: str
    build_time: float
    query_time: float
    build_qps: float
    query_qps: float

def setup_backend(points: np.ndarray):
    backend = build_residual_backend(
        points,
        seed=SEED,
        inducing_count=512,
        variance=1.0,
        lengthscale=1.0,
        chunk_size=512,
    )
    object.__setattr__(backend, "rbf_variance", 1.0)
    object.__setattr__(backend, "rbf_lengthscale", np.ones(DIMENSION, dtype=np.float32))
    configure_residual_correlation(backend)
    return backend

def run_variant(
    variant_name: str,
    points: np.ndarray,
    query_indices: np.ndarray,
    enable_rust: bool,
    build_metric: str, # 'euclidean' or 'residual_correlation'
    use_static_tree: bool, # For query
    build_mode: str = "coordinates", # "coordinates" or "indices"
) -> Result:
    print(f"  Running {variant_name}...")
    
    # Configure
    runtime_cfg = covertreex.config.runtime_config()
    cfg = dataclasses.replace(
        runtime_cfg,
        metric=build_metric,
        precision="float32",
        enable_numba=True,
        enable_rust=enable_rust,
        enable_sparse_traversal=False,
        enable_diagnostics=False,
        log_level="WARNING",
        batch_order_strategy="natural",
        residual_use_static_euclidean_tree=use_static_tree,
    )
    covertreex.config.configure_runtime(cfg)
    
    # Backend Setup
    backend = setup_backend(points)
    
    # Build
    dim = 1 if build_mode == "indices" else DIMENSION
    tree = PCCTree.empty(dimension=dim)
    
    batch_points_source = points.astype(np.float32, copy=False)
    
    start_build = time.perf_counter()
    total = batch_points_source.shape[0]
    
    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        if build_mode == "indices":
            # Create indices array (N, 1)
            chunk = np.arange(i, end, dtype=np.float32).reshape(-1, 1)
        else:
            chunk = batch_points_source[i:end]
            
        tree, _ = batch_insert(tree, chunk)
        
    build_time = time.perf_counter() - start_build
    
    # Query
    queries = np.asarray(query_indices, dtype=np.int64).reshape(-1, 1)
    
    # If using indices mode, we need to make sure queries are compatible?
    # knn takes `query_points`. 
    # If tree has indices, queries should be indices?
    # In our case, `queries` IS indices (N, 1) of int64.
    # knn will convert to float32 if needed.
    # So it should be fine.
    
    start_query = time.perf_counter()
    knn(tree, queries, k=K)
    query_time = time.perf_counter() - start_query
    
    return Result(
        variant=variant_name,
        build_time=build_time,
        query_time=query_time,
        build_qps=total / build_time if build_time > 0 else 0,
        query_qps=QUERIES / query_time if query_time > 0 else 0,
    )

def main():
    print(f"Benchmark Suite: 50k Points, Residual Correlation")
    print(f"Platform: Linux | CPU")
    
    rng = np.random.default_rng(SEED)
    points = gaussian_points(rng, DATASET_SIZE, DIMENSION, dtype=np.float64)
    query_rng = np.random.default_rng(SEED + 1)
    query_indices = query_rng.integers(0, DATASET_SIZE, size=QUERIES, endpoint=False, dtype=np.int64)
    
    # WARMUP (Smoke Test)
    print("\n>>> SMOKE TEST (N=1000) <<<")
    warmup_points = points[:WARMUP_SIZE]
    warmup_queries = query_rng.integers(0, WARMUP_SIZE, size=100, endpoint=False, dtype=np.int64)
    
    try:
        run_variant("Warmup-Py", warmup_points, warmup_queries, False, "euclidean", True, "coordinates")
        run_variant("Warmup-Rust", warmup_points, warmup_queries, True, "residual_correlation", False, "indices")
        print(">>> SMOKE TEST PASSED <<<")
    except Exception as e:
        print(f"\n!!! SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # BENCHMARK
    results = []
    
    # 1. Python Baseline: Static Euclidean Tree + Residual Query
    try:
        res = run_variant(
            "Py-Static (Euc/Res)", 
            points, query_indices, 
            enable_rust=False, 
            build_metric="euclidean", 
            use_static_tree=True,
            build_mode="coordinates"
        )
        results.append(res)
    except Exception as e:
        print(f"Py-Static Failed: {e}")

    # 2. Rust Static: Static Euclidean Tree + Residual Query
    try:
        res = run_variant(
            "Rust-Static (Euc/Res)", 
            points, query_indices, 
            enable_rust=True, 
            build_metric="euclidean", 
            use_static_tree=True,
            build_mode="coordinates"
        )
        results.append(res)
    except Exception as e:
        print(f"Rust-Static Failed: {e}")

    # 3. Rust Dynamic: Dynamic Residual Tree + Residual Query
    # This needs indices build mode!
    try:
        res = run_variant(
            "Rust-Dynamic (Res/Res)", 
            points, query_indices, 
            enable_rust=True, 
            build_metric="residual_correlation", 
            use_static_tree=False,
            build_mode="indices"
        )
        results.append(res)
    except Exception as e:
        print(f"Rust-Dynamic Failed: {e}")


    # REPORT
    print("\n" + "="*85)
    print(f"{ 'Variant':<25} | { 'Build (s)':>10} | { 'B-QPS':>8} | { 'Query (s)':>10} | { 'Q-QPS':>8} | { 'Speedup':>7}")
    print("-" * 85)
    
    baseline_qps = results[0].query_qps if results else 1.0
    
    for res in results:
        speedup = res.query_qps / baseline_qps if baseline_qps > 0 else 0.0
        print(f"{res.variant:<25} | {res.build_time:10.4f} | {res.build_qps:8.0f} | {res.query_time:10.4f} | {res.query_qps:8.0f} | {speedup:7.2f}x")
    print("="*85)

if __name__ == "__main__":
    main()
