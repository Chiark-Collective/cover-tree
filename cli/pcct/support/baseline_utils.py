from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import numpy as np

from covertreex.baseline import (
    BaselineCoverTree,
    ExternalCoverTreeBaseline,
    GPBoostCoverTreeBaseline,
    MlpackCoverTreeBaseline,
    ScikitLearnBaseline,
    ScipyBaseline,
    has_external_cover_tree,
    has_gpboost_cover_tree,
    has_mlpack_cover_tree,
    has_sklearn_baseline,
    has_scipy_baseline,
)


@dataclass(frozen=True)
class BaselineComparison:
    name: str
    build_seconds: float
    elapsed_seconds: float
    latency_ms: float
    queries_per_second: float


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


def _run_mlpack_baseline(points: np.ndarray, queries: np.ndarray, *, k: int) -> BaselineComparison:
    if not has_mlpack_cover_tree():
        raise RuntimeError(
            "mlpack cover tree baseline requested but mlpack bindings are not installed."
        )
    start_build = time.perf_counter()
    tree = MlpackCoverTreeBaseline.from_points(points)
    build_seconds = time.perf_counter() - start_build
    start = time.perf_counter()
    tree.knn(queries, k=k, return_distances=False)
    elapsed = time.perf_counter() - start
    qps = queries.shape[0] / elapsed if elapsed > 0 else float("inf")
    latency = (elapsed / queries.shape[0]) * 1e3 if queries.shape[0] else 0.0
    return BaselineComparison(
        name="mlpack",
        build_seconds=build_seconds,
        elapsed_seconds=elapsed,
        latency_ms=latency,
        queries_per_second=qps,
    )


def _run_sklearn_baseline(points: np.ndarray, queries: np.ndarray, *, k: int) -> BaselineComparison:
    if not has_sklearn_baseline():
        raise RuntimeError(
            "scikit-learn baseline requested but scikit-learn is not installed."
        )
    start_build = time.perf_counter()
    tree = ScikitLearnBaseline.from_points(points, algorithm="ball_tree")
    build_seconds = time.perf_counter() - start_build
    start = time.perf_counter()
    tree.knn(queries, k=k, return_distances=False)
    elapsed = time.perf_counter() - start
    qps = queries.shape[0] / elapsed if elapsed > 0 else float("inf")
    latency = (elapsed / queries.shape[0]) * 1e3 if queries.shape[0] else 0.0
    return BaselineComparison(
        name="sklearn_balltree",
        build_seconds=build_seconds,
        elapsed_seconds=elapsed,
        latency_ms=latency,
        queries_per_second=qps,
    )


def _run_scipy_baseline(points: np.ndarray, queries: np.ndarray, *, k: int) -> BaselineComparison:
    if not has_scipy_baseline():
        raise RuntimeError(
            "scipy baseline requested but scipy is not installed."
        )
    start_build = time.perf_counter()
    tree = ScipyBaseline.from_points(points)
    build_seconds = time.perf_counter() - start_build
    start = time.perf_counter()
    tree.knn(queries, k=k, return_distances=False)
    elapsed = time.perf_counter() - start
    qps = queries.shape[0] / elapsed if elapsed > 0 else float("inf")
    latency = (elapsed / queries.shape[0]) * 1e3 if queries.shape[0] else 0.0
    return BaselineComparison(
        name="scipy_ckdtree",
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
    
    modes = {m.strip() for m in mode.split(",")}
    run_all = "all" in modes
    results: List[BaselineComparison] = []

    if run_all or "sequential" in modes or "both" in modes:
        results.append(_run_sequential_baseline(points, queries, k=k))
    if run_all or "gpboost" in modes:
        results.append(_run_gpboost_baseline(points, queries, k=k))
    if run_all or "mlpack" in modes or "cover" in modes:
        results.append(_run_mlpack_baseline(points, queries, k=k))
    # "external" is the slow PyPI covertree. "both" and "cover" used to trigger it too.
    # We keep explicit "external" trigger, and "both"/"cover" for back-compat if needed,
    # but "all" will still trigger it unless we change that.
    # User requested to exclude it from "all" effectively, or just use explicit list in tool.
    # Let's make "all" include it for correctness of "all", but we will use explicit list in the tool.
    if run_all or "external" in modes or "both" in modes or "cover" in modes:
        results.append(_run_external_baseline(points, queries, k=k))
    if run_all or "sklearn" in modes:
        results.append(_run_sklearn_baseline(points, queries, k=k))
    if run_all or "scipy" in modes:
        results.append(_run_scipy_baseline(points, queries, k=k))
        
    return results


__all__ = ["BaselineComparison", "run_baseline_comparisons"]
