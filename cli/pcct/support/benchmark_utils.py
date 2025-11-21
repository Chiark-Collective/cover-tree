from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Tuple

import numpy as np
from numpy.random import Generator, default_rng

from covertreex import config as cx_config
from covertreex.algo import batch_insert, batch_insert_prefix_doubling
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.telemetry import BenchmarkLogWriter, ResidualScopeCapRecorder

from .runtime_utils import resolve_backend, measure_resources


def _ensure_context(context: cx_config.RuntimeContext | None) -> cx_config.RuntimeContext:
    existing = context or cx_config.current_runtime_context()
    if existing is not None:
        return existing
    return cx_config.runtime_context()


@dataclass(frozen=True)
class QueryBenchmarkResult:
    elapsed_seconds: float
    queries: int
    k: int
    latency_ms: float
    queries_per_second: float
    build_seconds: float | None = None
    cpu_user_seconds: float = 0.0
    cpu_system_seconds: float = 0.0
    rss_delta_bytes: int = 0


def _generate_backend_points(
    rng: Generator,
    count: int,
    dimension: int,
    *,
    backend: TreeBackend,
) -> np.ndarray:
    from tests.utils.datasets import gaussian_points
    samples = gaussian_points(rng, count, dimension, dtype=np.float64)
    return backend.asarray(samples, dtype=backend.default_float)


def _ensure_residual_backend(
    points: np.ndarray,
    context: cx_config.RuntimeContext,
) -> None:
    from covertreex.metrics.residual import (
        ResidualCorrHostData,
        configure_residual_correlation,
        get_residual_backend,
    )
    try:
        get_residual_backend()
        return
    except RuntimeError:
        pass
    
    # Auto-configure dummy backend
    N, D = points.shape
    # Rank 16 default
    rank = 16
    rng = default_rng(42)
    V = rng.normal(size=(N, rank)).astype(np.float32)
    p_diag = rng.uniform(0.1, 1.0, size=N).astype(np.float32)
    kernel_diag = np.ones(N, dtype=np.float32)
    
    def dummy_provider(r, c):
        return np.zeros((r.size, c.size), dtype=np.float64)
        
    host_data = ResidualCorrHostData(
        v_matrix=V,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=dummy_provider,
        chunk_size=512
    )
    # Inject points for potential Numba usage or debugging
    object.__setattr__(host_data, "kernel_points_f32", points.astype(np.float32))
    
    configure_residual_correlation(host_data, context=context)


def build_tree(
    *,
    dimension: int,
    tree_points: int,
    batch_size: int,
    seed: int,
    prebuilt_points: np.ndarray | None = None,
    log_writer: BenchmarkLogWriter | None = None,
    scope_cap_recorder: "ResidualScopeCapRecorder | None" = None,
    build_mode: str = "batch",
    plan_callback: Callable[[Any, int, int], Mapping[str, Any] | None] | None = None,
    context: cx_config.RuntimeContext | None = None,
) -> Tuple[PCCTree, np.ndarray, float]:
    from tests.utils.datasets import gaussian_points
    
    resolved_context = _ensure_context(context)
    backend = resolve_backend(context=resolved_context)
    tree = PCCTree.empty(dimension=dimension, backend=backend)
    runtime = resolved_context.config
    
    is_residual = runtime.metric == "residual_correlation"
    
    # If residual, we MUST generate all points upfront to configure backend
    if is_residual and prebuilt_points is None:
        rng = default_rng(seed)
        prebuilt_points = gaussian_points(rng, tree_points, dimension, dtype=np.float64)
        
    if is_residual:
        _ensure_residual_backend(prebuilt_points, resolved_context)

    if build_mode == "prefix":
        if prebuilt_points is not None:
            points_np = np.asarray(prebuilt_points, dtype=np.float64, copy=False)
        else:
            rng = default_rng(seed)
            points_np = gaussian_points(rng, tree_points, dimension, dtype=np.float64)
            
        # For residual, pass INDICES
        if is_residual:
            batch_data = np.arange(points_np.shape[0], dtype=np.int64).reshape(-1, 1)
            tree = PCCTree.empty(dimension=1, backend=backend)
            batch = backend.asarray(batch_data, dtype=backend.default_int)
        else:
            batch = backend.asarray(points_np, dtype=backend.default_float)
            
        start = time.perf_counter()
        tree, prefix_result = batch_insert_prefix_doubling(
            tree,
            batch,
            backend=backend,
            mis_seed=seed,
            shuffle_seed=seed,
            context=resolved_context,
        )
        build_seconds = time.perf_counter() - start
        schedule = runtime.prefix_schedule
        for group_index, group in enumerate(prefix_result.groups):
            plan = group.plan
            if hasattr(plan.traversal, "parents"):
                group_size = int(plan.traversal.parents.shape[0])
            else:
                group_size = int(plan.traversal.levels.shape[0])
            extra_payload: Mapping[str, Any] | None = None
            if plan_callback is not None:
                extra_payload = plan_callback(plan, group_index, group_size)
            prefix_extra = {
                "prefix_group_index": group_index,
                "prefix_factor": float(group.prefix_factor or 0.0),
                "prefix_domination_ratio": float(group.domination_ratio or 0.0),
                "prefix_schedule": schedule,
            }
            if extra_payload:
                prefix_extra.update(extra_payload)
            if log_writer is not None:
                log_writer.record_batch(
                    batch_index=group_index,
                    batch_size=group_size,
                    plan=plan,
                    extra=prefix_extra,
                )
            if scope_cap_recorder is not None:
                scope_cap_recorder.capture(plan)
            return tree, points_np, build_seconds

    start = time.perf_counter()
    buffers: list[np.ndarray] = []
    idx = 0

    if prebuilt_points is not None:
        points_np = np.asarray(prebuilt_points, dtype=np.float64, copy=False)
        total = points_np.shape[0]
        while idx * batch_size < total:
            start_idx = idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            
            if is_residual:
                batch_np = np.arange(start_idx, end_idx, dtype=np.int64).reshape(-1, 1)
                batch = backend.asarray(batch_np, dtype=backend.default_int)
                if idx == 0:
                     tree = PCCTree.empty(dimension=1, backend=backend)
            else:
                batch_np = points_np[start_idx:end_idx]
                batch = backend.asarray(batch_np, dtype=backend.default_float)
                
            tree, plan = batch_insert(
                tree,
                batch,
                mis_seed=seed + idx,
                context=resolved_context,
            )
            extra_payload = None
            if plan_callback is not None:
                extra_payload = plan_callback(plan, idx, int(batch_np.shape[0]))
            if log_writer is not None:
                log_writer.record_batch(
                    batch_index=idx,
                    batch_size=int(batch_np.shape[0]),
                    plan=plan,
                    extra=extra_payload,
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
            tree, plan = batch_insert(
                tree,
                batch,
                mis_seed=seed + idx,
                context=resolved_context,
            )
            extra_payload = None
            if plan_callback is not None:
                extra_payload = plan_callback(plan, idx, current)
            if log_writer is not None:
                log_writer.record_batch(
                    batch_index=idx,
                    batch_size=current,
                    plan=plan,
                    extra=extra_payload,
                )
            if scope_cap_recorder is not None:
                scope_cap_recorder.capture(plan)
            buffers.append(np.asarray(batch))
            remaining -= current
            idx += 1

    build_seconds = time.perf_counter() - start
    if prebuilt_points is not None:
        pass
    elif buffers:
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
    plan_callback: Callable[[Any, int, int], Mapping[str, Any] | None] | None = None,
    context: cx_config.RuntimeContext | None = None,
) -> Tuple[PCCTree, QueryBenchmarkResult]:
    from covertreex.queries.knn import knn
    
    resolved_context = _ensure_context(context)
    runtime = resolved_context.config
    is_residual = (
        runtime.metric == "residual_correlation"
        or runtime.residual_use_static_euclidean_tree
    )
    tree_build_seconds: float | None = None
    if prebuilt_tree is None:
        tree, _, tree_build_seconds = build_tree(
            dimension=dimension,
            tree_points=tree_points,
            batch_size=batch_size,
            seed=seed,
            prebuilt_points=prebuilt_points,
            log_writer=log_writer,
            scope_cap_recorder=scope_cap_recorder,
            build_mode=build_mode,
            plan_callback=plan_callback,
            context=resolved_context,
        )
    else:
        tree = prebuilt_tree
        tree_build_seconds = build_seconds

    if scope_cap_recorder is not None and tree_build_seconds is not None:
        scope_cap_recorder.annotate(tree_build_seconds=tree_build_seconds)

    backend = tree.backend
    if prebuilt_queries is None:
        query_rng = default_rng(seed + 1)
        if is_residual:
            indices = query_rng.integers(
                0,
                tree.num_points,
                size=query_count,
                endpoint=False,
                dtype=np.int64,
            ).reshape(-1, 1)
            queries = backend.asarray(indices, dtype=backend.default_int)
        else:
            queries = _generate_backend_points(
                query_rng,
                query_count,
                dimension,
                backend=backend,
            )
    else:
        qp = np.asarray(prebuilt_queries)
        if is_residual and qp.dtype.kind in {"i", "u"}:
            queries = backend.asarray(qp, dtype=backend.default_int)
        else:
            queries = backend.asarray(prebuilt_queries, dtype=backend.default_float)
    
    with measure_resources() as query_stats:
        knn(tree, queries, k=k, context=resolved_context)
    
    elapsed = query_stats['wall']
    qps = query_count / elapsed if elapsed > 0 else float("inf")
    latency = (elapsed / query_count) * 1e3 if query_count else 0.0
    return tree, QueryBenchmarkResult(
        elapsed_seconds=elapsed,
        queries=query_count,
        k=k,
        latency_ms=latency,
        queries_per_second=qps,
        build_seconds=tree_build_seconds,
        cpu_user_seconds=query_stats['user'],
        cpu_system_seconds=query_stats['system'],
        rss_delta_bytes=query_stats['rss_delta'],
    )


__all__ = [
    "QueryBenchmarkResult",
    "build_tree",
    "benchmark_knn_latency",
]
