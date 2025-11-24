from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from covertreex.algo.batch.types import BatchInsertPlan, BatchInsertTimings
from covertreex.algo.conflict import ConflictGraph, ConflictGraphTimings
from covertreex.algo.mis import MISResult
from covertreex.algo.traverse import TraversalResult, TraversalTimings
from covertreex.telemetry.logs import BenchmarkLogWriter


def _as_int64(seq: Any, *, fallback: int | None = None) -> np.ndarray:
    arr = np.asarray(seq, dtype=np.int64)
    if arr.ndim == 0:
        arr = arr.reshape(-1)
    if fallback is not None and arr.size == 0:
        arr = np.asarray(fallback, dtype=np.int64)
    return arr


def build_rust_plan(payload: Mapping[str, Any]) -> BatchInsertPlan:
    parents = _as_int64(payload.get("parents", ()))
    levels = _as_int64(payload.get("levels", ()))
    batch_size = int(payload.get("batch_size", parents.size))
    scope_indptr = _as_int64(
        payload.get("scope_indptr", np.zeros(batch_size + 1, dtype=np.int64))
    )
    scope_indices = _as_int64(payload.get("scope_indices", ()))
    conflict_indptr = _as_int64(
        payload.get("conflict_indptr", np.zeros(batch_size + 1, dtype=np.int64))
    )
    conflict_indices = _as_int64(payload.get("conflict_indices", ()))
    selected = _as_int64(payload.get("selected", ()))
    dominated = _as_int64(payload.get("dominated", ()))
    timings_map = payload.get("timings", {}) or {}

    traversal_timings = TraversalTimings(
        pairwise_seconds=float(timings_map.get("pairwise_seconds", 0.0)),
        mask_seconds=float(timings_map.get("mask_seconds", 0.0)),
        semisort_seconds=float(timings_map.get("semisort_seconds", 0.0)),
        chain_seconds=float(timings_map.get("chain_seconds", 0.0)),
        nonzero_seconds=float(timings_map.get("nonzero_seconds", 0.0)),
        sort_seconds=float(timings_map.get("sort_seconds", 0.0)),
        assemble_seconds=float(timings_map.get("assemble_seconds", 0.0)),
        tile_seconds=float(timings_map.get("tile_seconds", 0.0)),
        build_wall_seconds=float(
            timings_map.get(
                "build_wall_seconds", timings_map.get("traversal_seconds", 0.0)
            )
        ),
        scope_chunk_segments=int(
            timings_map.get("scope_chunk_segments", batch_size)
        ),
        scope_chunk_emitted=int(
            timings_map.get("scope_chunk_emitted", parents.size)
        ),
        scope_chunk_max_members=int(
            timings_map.get("scope_chunk_max_members", parents.size)
        ),
        scope_chunk_scans=int(timings_map.get("scope_chunk_scans", 0)),
        scope_chunk_points=int(timings_map.get("scope_chunk_points", 0)),
        scope_chunk_dedupe=int(timings_map.get("scope_chunk_dedupe", 0)),
        scope_chunk_saturated=int(
            timings_map.get("scope_chunk_saturated", 0)
        ),
        scope_cache_hits=int(timings_map.get("scope_cache_hits", 0)),
        scope_cache_prefetch=int(timings_map.get("scope_cache_prefetch", 0)),
        scope_budget_start=int(timings_map.get("scope_budget_start", 0)),
        scope_budget_final=int(timings_map.get("scope_budget_final", 0)),
        scope_budget_escalations=int(
            timings_map.get("scope_budget_escalations", 0)
        ),
        scope_budget_early_terminate=int(
            timings_map.get("scope_budget_early_terminate", 0)
        ),
        whitened_block_pairs=int(
            timings_map.get("whitened_block_pairs", 0)
        ),
        whitened_block_seconds=float(
            timings_map.get("whitened_block_seconds", 0.0)
        ),
        whitened_block_calls=int(
            timings_map.get("whitened_block_calls", 0)
        ),
        kernel_provider_pairs=int(
            timings_map.get("kernel_provider_pairs", 0)
        ),
        kernel_provider_seconds=float(
            timings_map.get("kernel_provider_seconds", 0.0)
        ),
        kernel_provider_calls=int(
            timings_map.get("kernel_provider_calls", 0)
        ),
    )

    conflict_timings = ConflictGraphTimings(
        pairwise_seconds=float(timings_map.get("pairwise_seconds", 0.0)),
        scope_group_seconds=float(timings_map.get("scope_group_seconds", 0.0)),
        adjacency_seconds=float(
            timings_map.get(
                "adjacency_seconds", timings_map.get("conflict_graph_seconds", 0.0)
            )
        ),
        annulus_seconds=float(timings_map.get("annulus_seconds", 0.0)),
        adjacency_membership_seconds=float(
            timings_map.get("adjacency_membership_seconds", 0.0)
        ),
        adjacency_targets_seconds=float(
            timings_map.get("adjacency_targets_seconds", 0.0)
        ),
        adjacency_scatter_seconds=float(
            timings_map.get("adjacency_scatter_seconds", 0.0)
        ),
        adjacency_filter_seconds=float(
            timings_map.get("adjacency_filter_seconds", 0.0)
        ),
        adjacency_sort_seconds=float(
            timings_map.get("adjacency_sort_seconds", 0.0)
        ),
        adjacency_dedup_seconds=float(
            timings_map.get("adjacency_dedup_seconds", 0.0)
        ),
        adjacency_extract_seconds=float(
            timings_map.get("adjacency_extract_seconds", 0.0)
        ),
        adjacency_csr_seconds=float(
            timings_map.get("adjacency_csr_seconds", 0.0)
        ),
        adjacency_total_pairs=float(
            timings_map.get("adjacency_total_pairs", conflict_indices.size)
        ),
        adjacency_candidate_pairs=float(
            timings_map.get("adjacency_candidate_pairs", conflict_indices.size)
        ),
        adjacency_max_group_size=float(
            timings_map.get("adjacency_max_group_size", 0.0)
        ),
        scope_bytes_h2d=int(timings_map.get("scope_bytes_h2d", 0)),
        scope_bytes_d2h=int(timings_map.get("scope_bytes_d2h", 0)),
        scope_groups=int(timings_map.get("scope_groups", 0)),
        scope_groups_unique=int(timings_map.get("scope_groups_unique", 0)),
        scope_domination_ratio=float(
            timings_map.get("scope_domination_ratio", 0.0)
        ),
        scope_chunk_segments=int(
            timings_map.get("scope_chunk_segments", batch_size)
        ),
        scope_chunk_emitted=int(
            timings_map.get("scope_chunk_emitted", parents.size)
        ),
        scope_chunk_max_members=int(
            timings_map.get("scope_chunk_max_members", parents.size)
        ),
        scope_chunk_pair_cap=int(
            timings_map.get("scope_chunk_pair_cap", 0)
        ),
        scope_chunk_pairs_before=int(
            timings_map.get("scope_chunk_pairs_before", conflict_indices.size)
        ),
        scope_chunk_pairs_after=int(
            timings_map.get("scope_chunk_pairs_after", conflict_indices.size)
        ),
        scope_chunk_pair_merges=int(
            timings_map.get("scope_chunk_pair_merges", 0)
        ),
        mis_seconds=float(timings_map.get("mis_seconds", 0.0)),
        pairwise_reused=int(timings_map.get("pairwise_reused", 0)),
        grid_cells=int(payload.get("conflict_grid_cells", 0)),
        grid_leaders_raw=int(payload.get("conflict_grid_leaders_raw", 0)),
        grid_leaders_after=int(payload.get("conflict_grid_leaders_after", 0)),
        grid_local_edges=int(payload.get("conflict_grid_local_edges", 0)),
        arena_bytes=int(timings_map.get("arena_bytes", 0)),
        degree_cap=int(timings_map.get("degree_cap", 0)),
        degree_pruned_pairs=int(
            timings_map.get("degree_pruned_pairs", 0)
        ),
    )

    conflict_graph = ConflictGraph(
        indptr=conflict_indptr,
        indices=conflict_indices,
        pairwise_distances=np.zeros(conflict_indices.shape[0], dtype=np.float32),
        scope_indptr=scope_indptr,
        scope_indices=scope_indices,
        radii=np.zeros(batch_size, dtype=np.float32),
        annulus_bounds=np.zeros((0, 2), dtype=np.float32),
        annulus_bins=np.zeros(0, dtype=np.int64),
        annulus_bin_indptr=np.zeros(1, dtype=np.int64),
        annulus_bin_indices=np.zeros(0, dtype=np.int64),
        annulus_bin_ids=np.zeros(0, dtype=np.int64),
        timings=conflict_timings,
        forced_selected=None,
        forced_dominated=None,
        grid_cells=int(payload.get("conflict_grid_cells", 0)),
        grid_leaders_raw=int(payload.get("conflict_grid_leaders_raw", 0)),
        grid_leaders_after=int(payload.get("conflict_grid_leaders_after", 0)),
        grid_local_edges=int(payload.get("conflict_grid_local_edges", 0)),
    )

    conflict_scopes = tuple(() for _ in range(batch_size))
    traversal = TraversalResult(
        parents=parents,
        levels=levels,
        conflict_scopes=conflict_scopes,
        scope_indptr=scope_indptr,
        scope_indices=scope_indices,
        timings=traversal_timings,
        residual_cache=None,
        engine=str(payload.get("engine", "rust-pcct2")),
        gate_active=bool(payload.get("gate_active", False)),
    )

    mis_result = MISResult(
        independent_set=selected,
        iterations=int(timings_map.get("mis_iterations", 1)),
    )

    plan_timings = BatchInsertTimings(
        traversal_seconds=float(timings_map.get("traversal_seconds", 0.0)),
        conflict_graph_seconds=float(
            timings_map.get("conflict_graph_seconds", 0.0)
        ),
        mis_seconds=float(timings_map.get("mis_seconds", 0.0)),
    )

    return BatchInsertPlan(
        traversal=traversal,
        conflict_graph=conflict_graph,
        mis_result=mis_result,
        selected_indices=selected,
        dominated_indices=dominated,
        level_summaries=tuple(),
        timings=plan_timings,
        batch_permutation=None,
        batch_order_strategy=str(
            payload.get("batch_order_strategy", "rust-pcct2")
        ),
        batch_order_metrics=dict(payload.get("batch_order_metrics", {}) or {}),
    )


def record_rust_batch(
    log_writer: BenchmarkLogWriter, payload: Mapping[str, Any]
) -> None:
    plan = build_rust_plan(payload)
    log_writer.record_batch(
        batch_index=int(payload.get("batch_index", 0)),
        batch_size=int(payload.get("batch_size", 0)),
        plan=plan,
        extra=payload.get("extra"),
    )


__all__ = ["build_rust_plan", "record_rust_batch"]
