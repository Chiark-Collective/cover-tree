from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import time

from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.logging import get_logger
from covertreex.queries._knn_numba import knn_numba, materialise_tree_view_cached

from ..._traverse_numba import NUMBA_TRAVERSAL_AVAILABLE, build_scopes_numba
from ..._traverse_sparse_numba import (
    NUMBA_SPARSE_TRAVERSAL_AVAILABLE,
    collect_sparse_scopes,
    collect_sparse_scopes_csr,
)
from ..._scope_numba import build_scope_csr_from_pairs
from ...semisort import select_topk_by_level
from ..base import (
    TraversalResult,
    TraversalTimings,
    TraversalStrategy,
    block_until_ready,
    collect_distances,
    empty_result,
)
from .common import _collect_next_chain
from .registry import register_traversal_strategy

LOGGER = get_logger("algo.traverse.euclidean")


class _EuclideanDenseTraversal(TraversalStrategy):
    def collect(
        self,
        tree: PCCTree,
        batch: Any,
        *,
        backend: TreeBackend,
        runtime: Any,
    ) -> TraversalResult:
        return _collect_euclidean_dense(tree, batch, backend=backend, runtime=runtime)


class _EuclideanSparseTraversal(TraversalStrategy):
    def collect(
        self,
        tree: PCCTree,
        batch: Any,
        *,
        backend: TreeBackend,
        runtime: Any,
    ) -> TraversalResult:
        return _collect_euclidean_sparse(tree, batch, backend=backend, runtime=runtime)


def _collect_euclidean_dense(
    tree: PCCTree,
    batch: Any,
    *,
    backend: TreeBackend,
    runtime: Any,
) -> TraversalResult:
    xp = backend.xp
    batch_size = int(batch.shape[0]) if batch.size else 0

    start = time.perf_counter()
    distances = collect_distances(tree, batch, backend)
    block_until_ready(distances)
    pairwise_seconds = time.perf_counter() - start

    start = time.perf_counter()
    parents = xp.argmin(distances, axis=1).astype(backend.default_int)
    levels = tree.top_levels[parents]

    base_radius = xp.power(2.0, levels.astype(backend.default_float) + 1.0)
    si_values = tree.si_cache[parents]
    radius = xp.maximum(base_radius, si_values)
    node_indices = xp.arange(tree.num_points, dtype=backend.default_int)
    parent_mask = node_indices[None, :] == parents[:, None]
    within_radius = distances <= radius[:, None]
    mask = xp.logical_or(within_radius, parent_mask)
    block_until_ready(mask)
    mask_seconds = time.perf_counter() - start

    parents_np = np.asarray(parents, dtype=np.int64)
    top_levels_np = np.asarray(tree.top_levels, dtype=np.int64)
    next_cache_np = np.asarray(tree.next_cache, dtype=np.int64)

    mask_np = np.asarray(backend.to_numpy(mask), dtype=bool)
    if mask_np.ndim != 2:
        mask_np = np.reshape(mask_np, (batch_size, tree.num_points))

    chunk_target = int(runtime.scope_chunk_target)
    scope_limit = chunk_target if chunk_target > 0 else 0

    use_numba = runtime.enable_numba and NUMBA_TRAVERSAL_AVAILABLE

    if use_numba:
        numba_start = time.perf_counter()
        scope_indptr_np, scope_indices_np = build_scopes_numba(
            mask_np,
            parents_np,
            next_cache_np,
            top_levels_np,
        )
        numba_end = time.perf_counter()
        semisort_seconds = numba_end - start
        chain_seconds = numba_end - numba_start
        nonzero_seconds = 0.0
        sort_seconds = 0.0
        assemble_seconds = 0.0
    else:
        unique_parents = {int(p) for p in parents_np if int(p) >= 0}
        chain_update_start = time.perf_counter()
        chain_map = {
            parent: _collect_next_chain(tree, parent, next_cache=next_cache_np)
            for parent in unique_parents
        }
        if chain_map:
            for idx, parent in enumerate(parents_np):
                if parent >= 0:
                    chain = chain_map.get(int(parent))
                    if chain:
                        mask_np[idx, list(chain)] = True
        chain_seconds = time.perf_counter() - chain_update_start

        nonzero_start = time.perf_counter()
        row_ids, col_ids = np.nonzero(mask_np)
        nonzero_seconds = time.perf_counter() - nonzero_start

        sort_start = time.perf_counter()
        if row_ids.size:
            counts = np.bincount(row_ids, minlength=batch_size)
            offsets = np.zeros(batch_size + 1, dtype=np.int64)
            np.cumsum(counts, out=offsets[1:])
            grouped_cols = np.empty(row_ids.size, dtype=np.int64)
            cursors = offsets[:-1].copy()
            for idx in range(row_ids.size):
                row = row_ids[idx]
                pos = cursors[row]
                grouped_cols[pos] = col_ids[idx]
                cursors[row] = pos + 1
            selections: list[np.ndarray] = []
            scope_indptr_list = [0]
            cursor = 0
            limit_value = scope_limit if scope_limit > 0 else 0
            for row in range(batch_size):
                start = offsets[row]
                end = offsets[row + 1]
                if start == end:
                    scope_indptr_list.append(cursor)
                    continue
                row_values = grouped_cols[start:end].astype(np.int64, copy=False)
                selected = select_topk_by_level(
                    row_values,
                    top_levels_np[row_values],
                    limit_value,
                )
                parent_idx = parents_np[row] if row < parents_np.size else -1
                selected = _ensure_parent_in_selection(
                    selected,
                    row_values,
                    parent_idx,
                    top_levels_np,
                    limit_value,
                )
                selections.append(selected)
                cursor += selected.size
                scope_indptr_list.append(cursor)

            scope_indptr_np = np.asarray(scope_indptr_list, dtype=np.int64)
            if cursor > 0:
                scope_indices_np = np.concatenate(selections).astype(np.int64, copy=False)
            else:
                scope_indices_np = np.empty(0, dtype=np.int64)
        else:
            scope_indptr_np = np.zeros(batch_size + 1, dtype=np.int64)
            scope_indices_np = np.zeros(0, dtype=np.int64)
        sort_seconds = time.perf_counter() - sort_start

        semisort_seconds = time.perf_counter() - start
        assemble_seconds = 0.0

    assemble_start = time.perf_counter()
    conflict_scopes = tuple(
        tuple(int(x) for x in scope_indices_np[scope_indptr_np[i] : scope_indptr_np[i + 1]])
        for i in range(batch_size)
    )
    scope_indptr_arr = backend.asarray(scope_indptr_np, dtype=backend.default_int)
    scope_indices_arr = backend.asarray(scope_indices_np, dtype=backend.default_int)
    assemble_seconds += time.perf_counter() - assemble_start

    LOGGER.debug(
        "Traversal assigned parents %s at levels %s",
        backend.to_numpy(parents),
        backend.to_numpy(levels),
    )

    return TraversalResult(
        parents=backend.device_put(parents),
        levels=backend.device_put(levels),
        conflict_scopes=conflict_scopes,
        scope_indptr=scope_indptr_arr,
        scope_indices=scope_indices_arr,
        timings=TraversalTimings(
            pairwise_seconds=pairwise_seconds,
            mask_seconds=mask_seconds,
            semisort_seconds=semisort_seconds,
            chain_seconds=chain_seconds,
            nonzero_seconds=nonzero_seconds,
            sort_seconds=sort_seconds,
            assemble_seconds=assemble_seconds,
        ),
        engine="euclidean_dense",
        gate_active=False,
    )


def _collect_euclidean_sparse(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: TreeBackend,
    runtime: Any,
) -> TraversalResult:
    queries_np = np.asarray(backend.to_numpy(batch_points), dtype=np.float64)
    if queries_np.ndim == 1:
        queries_np = queries_np[None, :]

    batch_size = int(queries_np.shape[0])
    if batch_size == 0:
        return empty_result(backend, 0)

    view = materialise_tree_view_cached(tree)

    chunk_target = int(runtime.scope_chunk_target)
    scope_limit = chunk_target if chunk_target > 0 else 0

    parent_start = time.perf_counter()
    indices, _distances = knn_numba(view, queries_np, k=1, return_distances=True)
    pairwise_seconds = time.perf_counter() - parent_start

    parents_np = np.asarray(indices, dtype=np.int64).reshape(batch_size)
    top_levels_np = view.top_levels
    levels_np = np.full(batch_size, -1, dtype=np.int64)
    valid_mask = parents_np >= 0
    if np.any(valid_mask):
        levels_np[valid_mask] = top_levels_np[parents_np[valid_mask]]

    base_radii = np.zeros(batch_size, dtype=np.float64)
    base_radii[valid_mask] = np.power(2.0, levels_np[valid_mask].astype(np.float64) + 1.0)

    si_values = np.zeros(batch_size, dtype=np.float64)
    if view.si_cache.size:
        within_si = np.logical_and(valid_mask, parents_np < view.si_cache.shape[0])
        si_values[within_si] = view.si_cache[parents_np[within_si]]

    radii_np = np.maximum(base_radii, si_values)

    # chunk_target / scope_limit already computed above

    scope_start = time.perf_counter()
    if chunk_target > 0:
        scope_indptr_np, scope_indices_np, chunk_segments, chunk_emitted, chunk_max_members = collect_sparse_scopes_csr(
            view,
            queries_np,
            parents_np,
            radii_np,
            chunk_target,
        )
        scope_seconds = time.perf_counter() - scope_start
        scope_indptr_np = scope_indptr_np.astype(np.int64, copy=False)
        scope_indices_np = scope_indices_np.astype(np.int64, copy=False)
        if scope_limit > 0 and scope_indices_np.size:
            counts = np.diff(scope_indptr_np).astype(np.int64, copy=False)
            owners_arr = np.repeat(np.arange(batch_size, dtype=np.int64), counts)
            scope_indptr_np, scope_indices_np = build_scope_csr_from_pairs(
                owners_arr,
                scope_indices_np.astype(np.int64, copy=False),
                batch_size,
                limit=scope_limit,
                top_levels=top_levels_np,
                parents=parents_np,
            )
        chunk_max_members = max(chunk_max_members, _max_scope_size(scope_indptr_np))
        semisort_seconds = 0.0
        tile_seconds = scope_seconds
    else:
        scopes = collect_sparse_scopes(view, queries_np, parents_np, radii_np)
        scope_seconds = time.perf_counter() - scope_start

        owners_chunks: list[np.ndarray] = []
        member_chunks: list[np.ndarray] = []
        for row, scope in enumerate(scopes):
            nodes = np.asarray(scope, dtype=np.int64)
            if nodes.size:
                owners_chunks.append(np.full(nodes.size, row, dtype=np.int64))
                member_chunks.append(nodes)
        if owners_chunks:
            owners_arr = np.concatenate(owners_chunks)
            members_arr = np.concatenate(member_chunks)
            scope_indptr_np, scope_indices_np = build_scope_csr_from_pairs(
                owners_arr,
                members_arr,
                batch_size,
                limit=scope_limit,
                top_levels=top_levels_np,
                parents=parents_np,
            )
        else:
            scope_indptr_np = np.zeros(batch_size + 1, dtype=np.int64)
            scope_indices_np = np.empty(0, dtype=np.int64)
        chunk_segments = batch_size
        chunk_emitted = 0
        chunk_max_members = _max_scope_size(scope_indptr_np)
        semisort_seconds = scope_seconds
        tile_seconds = 0.0

    if chunk_target > 0:
        # ensure chunk metrics have sensible defaults when scopes are empty
        if chunk_max_members == 0 and scope_indices_np.size:
            chunk_max_members = _max_scope_size(scope_indptr_np)

    conflict_scopes_tuple = _conflict_scopes_from_csr(scope_indptr_np, scope_indices_np)

    parents_arr = backend.asarray(parents_np.astype(np.int64), dtype=backend.default_int)
    levels_arr = backend.asarray(levels_np.astype(np.int64), dtype=backend.default_int)
    scope_indptr_arr = backend.asarray(scope_indptr_np.astype(np.int64), dtype=backend.default_int)
    scope_indices_arr = backend.asarray(scope_indices_np.astype(np.int64), dtype=backend.default_int)

    return TraversalResult(
        parents=backend.device_put(parents_arr),
        levels=backend.device_put(levels_arr),
        conflict_scopes=conflict_scopes_tuple,
        scope_indptr=backend.device_put(scope_indptr_arr),
        scope_indices=backend.device_put(scope_indices_arr),
        timings=TraversalTimings(
            pairwise_seconds=pairwise_seconds,
            mask_seconds=0.0,
            semisort_seconds=semisort_seconds,
            chain_seconds=0.0,
            nonzero_seconds=0.0,
            sort_seconds=0.0,
            assemble_seconds=0.0,
            tile_seconds=tile_seconds,
            scope_chunk_segments=int(chunk_segments),
            scope_chunk_emitted=int(chunk_emitted),
            scope_chunk_max_members=int(chunk_max_members),
        ),
        engine="euclidean_sparse",
        gate_active=False,
    )


def _ensure_parent_in_selection(
    selection: np.ndarray,
    row_values: np.ndarray,
    parent_idx: int,
    top_levels_np: np.ndarray,
    limit: int,
) -> np.ndarray:
    if parent_idx < 0 or row_values.size == 0:
        return selection
    if not np.any(row_values == parent_idx):
        return selection
    if selection.size == 0:
        return np.asarray([parent_idx], dtype=np.int64)
    if np.any(selection == parent_idx):
        return selection
    augmented = np.concatenate([selection, np.asarray([parent_idx], dtype=np.int64)])
    keep = limit if limit > 0 else 0
    return select_topk_by_level(augmented, top_levels_np[augmented], keep)



def _conflict_scopes_from_csr(
    scope_indptr: np.ndarray,
    scope_indices: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    batch_size = scope_indptr.size - 1
    scopes: list[tuple[int, ...]] = []
    for row in range(batch_size):
        start = int(scope_indptr[row])
        end = int(scope_indptr[row + 1])
        if start == end:
            scopes.append(())
        else:
            scopes.append(tuple(int(x) for x in scope_indices[start:end]))
    return tuple(scopes)


def _max_scope_size(scope_indptr: np.ndarray) -> int:
    if scope_indptr.size <= 1:
        return 0
    diffs = scope_indptr[1:] - scope_indptr[:-1]
    if diffs.size == 0:
        return 0
    return int(np.max(diffs))


register_traversal_strategy(
    "euclidean_sparse_numba",
    predicate=lambda runtime, backend: (
        runtime.enable_sparse_traversal
        and runtime.enable_numba
        and runtime.metric == "euclidean"
        and NUMBA_SPARSE_TRAVERSAL_AVAILABLE
        and backend.name == "numpy"
    ),
    factory=_EuclideanSparseTraversal,
)


register_traversal_strategy(
    "euclidean_dense",
    predicate=lambda runtime, backend: runtime.metric == "euclidean",
    factory=_EuclideanDenseTraversal,
)


__all__ = [
    "_EuclideanDenseTraversal",
    "_EuclideanSparseTraversal",
]
