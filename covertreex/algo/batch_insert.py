from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

import jax.numpy as jnp

from covertreex.algo.conflict_graph import ConflictGraph, build_conflict_graph
from covertreex.algo.mis import MISResult, run_mis
from covertreex.algo.traverse import TraversalResult, traverse_collect_scopes
from covertreex.core.tree import PCCTree, TreeBackend, TreeLogStats


@dataclass(frozen=True)
class BatchInsertPlan:
    traversal: TraversalResult
    conflict_graph: ConflictGraph
    mis_result: MISResult
    selected_indices: Any
    dominated_indices: Any
    level_summaries: tuple["LevelSummary", ...]


@dataclass(frozen=True)
class LevelSummary:
    level: int
    candidates: Any
    selected: Any
    dominated: Any


@dataclass(frozen=True)
class PrefixBatchGroup:
    permutation_indices: Any
    plan: BatchInsertPlan


@dataclass(frozen=True)
class PrefixBatchResult:
    permutation: Any
    groups: Tuple[PrefixBatchGroup, ...]


def _recompute_level_offsets(backend: TreeBackend, top_levels: Any) -> Any:
    """Return refreshed level offsets in descending level order."""

    top_levels_np = np.asarray(backend.to_numpy(top_levels), dtype=np.int64)
    if top_levels_np.size == 0:
        return backend.asarray([0], dtype=backend.default_int)
    top_levels_np = np.clip(top_levels_np, 0, None)
    max_level = int(top_levels_np.max())
    counts = np.bincount(top_levels_np, minlength=max_level + 1)
    counts_desc = counts[::-1]  # highest level first
    offsets = np.concatenate([[0], np.cumsum(counts_desc, dtype=np.int64)])
    return backend.asarray(offsets, dtype=backend.default_int)


def plan_batch_insert(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: Optional[TreeBackend] = None,
    mis_seed: int | None = None,
) -> BatchInsertPlan:
    backend = backend or tree.backend
    traversal = traverse_collect_scopes(tree, batch_points, backend=backend)
    conflict_graph = build_conflict_graph(tree, traversal, batch_points, backend=backend)
    mis_result = run_mis(backend, conflict_graph, seed=mis_seed)
    selected_indices = jnp.where(mis_result.independent_set == 1)[0]
    dominated_indices = jnp.where(mis_result.independent_set == 0)[0]

    levels_np = np.asarray(backend.to_numpy(traversal.levels), dtype=np.int64)
    selected_np = np.asarray(backend.to_numpy(selected_indices), dtype=np.int64)
    dominated_np = np.asarray(backend.to_numpy(dominated_indices), dtype=np.int64)
    clamped_levels = np.maximum(levels_np, 0)
    unique_levels = np.unique(clamped_levels)
    level_summaries = []
    for lvl in unique_levels:
        mask = clamped_levels == lvl
        candidate_idx = np.nonzero(mask)[0]
        if candidate_idx.size == 0:
            continue
        selected_mask = np.isin(candidate_idx, selected_np, assume_unique=False)
        selected_idx = candidate_idx[selected_mask]
        dominated_idx = candidate_idx[~selected_mask]
        level_summaries.append(
            LevelSummary(
                level=int(lvl),
                candidates=backend.asarray(candidate_idx, dtype=backend.default_int),
                selected=backend.asarray(selected_idx, dtype=backend.default_int),
                dominated=backend.asarray(dominated_idx, dtype=backend.default_int),
            )
        )

    return BatchInsertPlan(
        traversal=traversal,
        conflict_graph=conflict_graph,
        mis_result=mis_result,
        selected_indices=selected_indices,
        dominated_indices=dominated_indices,
        level_summaries=tuple(level_summaries),
    )


def batch_insert(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: Optional[TreeBackend] = None,
    mis_seed: int | None = None,
) -> tuple[PCCTree, BatchInsertPlan]:
    backend = backend or tree.backend
    plan = plan_batch_insert(tree, batch_points, backend=backend, mis_seed=mis_seed)

    total_new_candidates = int(plan.selected_indices.size + plan.dominated_indices.size)
    if total_new_candidates == 0:
        return tree, plan

    xp = backend.xp
    batch = backend.asarray(batch_points, dtype=backend.default_float)
    selected_points = batch[plan.selected_indices]
    selected_levels = plan.traversal.levels[plan.selected_indices]
    selected_levels = xp.maximum(
        selected_levels, xp.zeros_like(selected_levels, dtype=backend.default_int)
    )
    selected_parents = plan.traversal.parents[plan.selected_indices]
    selected_si = plan.conflict_graph.radii[plan.selected_indices]

    dominated_points = batch[plan.dominated_indices]
    dominated_levels = plan.traversal.levels[plan.dominated_indices]
    dominated_levels = xp.maximum(
        dominated_levels
        - xp.ones_like(dominated_levels, dtype=backend.default_int),
        xp.zeros_like(dominated_levels, dtype=backend.default_int),
    )
    dominated_parents = plan.traversal.parents[plan.dominated_indices]
    dominated_si = plan.conflict_graph.radii[plan.dominated_indices]

    inserted_points = xp.concatenate([selected_points, dominated_points], axis=0)
    inserted_levels = xp.concatenate([selected_levels, dominated_levels], axis=0)
    inserted_parents = xp.concatenate([selected_parents, dominated_parents], axis=0)
    inserted_si = xp.concatenate([selected_si, dominated_si], axis=0)

    selected_batch_indices = np.asarray(
        backend.to_numpy(plan.selected_indices), dtype=np.int64
    )
    dominated_batch_indices = np.asarray(
        backend.to_numpy(plan.dominated_indices), dtype=np.int64
    )
    inserted_parents_np = np.asarray(
        backend.to_numpy(inserted_parents), dtype=np.int64
    )

    if dominated_batch_indices.size:
        selected_to_global: dict[int, int] = {
            int(batch_idx): int(tree.num_points + offset)
            for offset, batch_idx in enumerate(selected_batch_indices)
        }

        graph_indptr = np.asarray(
            backend.to_numpy(plan.conflict_graph.indptr), dtype=np.int64
        )
        graph_indices = np.asarray(
            backend.to_numpy(plan.conflict_graph.indices), dtype=np.int64
        )
        mis_mask = np.asarray(
            backend.to_numpy(plan.mis_result.independent_set), dtype=np.int8
        )
        batch_np = np.asarray(backend.to_numpy(batch), dtype=float)

        num_selected = int(selected_batch_indices.size)
        for offset, batch_idx in enumerate(dominated_batch_indices):
            start = graph_indptr[batch_idx]
            end = graph_indptr[batch_idx + 1]
            neighbors = graph_indices[start:end]
            candidate: list[tuple[float, int]] = []
            for nb in neighbors:
                if mis_mask[nb] != 1:
                    continue
                parent_idx = selected_to_global.get(int(nb))
                if parent_idx is None:
                    continue
                dist = float(
                    np.linalg.norm(batch_np[batch_idx] - batch_np[int(nb)], ord=2)
                )
                candidate.append((dist, parent_idx))
            if candidate:
                candidate.sort(key=lambda item: item[0])
                inserted_parents_np[num_selected + offset] = candidate[0][1]

        inserted_parents = backend.asarray(
            inserted_parents_np, dtype=backend.default_int
        )

    inserted_levels_np = np.asarray(backend.to_numpy(inserted_levels), dtype=np.int64)
    parent_levels_np = np.empty_like(inserted_parents_np)
    dominated_levels_np = np.asarray(
        backend.to_numpy(dominated_levels), dtype=np.int64
    )

    total_inserted = inserted_levels_np.shape[0]
    for idx_parent, parent in enumerate(inserted_parents_np):
        if parent < 0:
            parent_levels_np[idx_parent] = 0
        elif parent < tree.num_points:
            parent_levels_np[idx_parent] = int(tree.top_levels[parent])
        else:
            offset = parent - tree.num_points
            if offset < total_inserted:
                parent_levels_np[idx_parent] = int(inserted_levels_np[offset])
            else:
                parent_levels_np[idx_parent] = 0

    selected_count = selected_batch_indices.size
    dominated_count = dominated_batch_indices.size
    if dominated_count:
        for idx_dom in range(dominated_count):
            global_idx = selected_count + idx_dom
            parent_level = int(parent_levels_np[global_idx])
            candidate = int(dominated_levels_np[idx_dom])
            new_level = max(0, min(candidate, parent_level - 1))
            inserted_levels_np[selected_count + idx_dom] = new_level

    inserted_levels = backend.asarray(inserted_levels_np, dtype=backend.default_int)

    new_points = xp.concatenate([tree.points, inserted_points], axis=0)
    new_top_levels = xp.concatenate([tree.top_levels, inserted_levels], axis=0)
    new_parents = xp.concatenate([tree.parents, inserted_parents], axis=0)

    new_children = xp.concatenate(
        [
            tree.children,
            xp.full((inserted_points.shape[0],), -1, dtype=backend.default_int),
        ],
        axis=0,
    )
    new_si_cache = xp.concatenate([tree.si_cache, inserted_si], axis=0)
    new_next_cache = xp.concatenate(
        [
            tree.next_cache,
            xp.full((inserted_points.shape[0],), -1, dtype=backend.default_int),
        ],
        axis=0,
    )

    updated_children = new_children
    updated_next = new_next_cache
    parents_list = [int(p) for p in np.array(inserted_parents)]
    base_index = tree.num_points
    for offset, parent in enumerate(parents_list):
        if parent < 0:
            continue
        global_idx = base_index + offset
        prev_child = int(np.array(updated_children[parent]))
        prev_next = int(np.array(updated_next[parent]))
        updated_children = updated_children.at[parent].set(global_idx)
        updated_next = updated_next.at[parent].set(global_idx)
        if prev_child >= 0:
            updated_next = updated_next.at[global_idx].set(prev_child)
        elif prev_next >= 0:
            updated_next = updated_next.at[global_idx].set(prev_next)
        else:
            updated_next = updated_next.at[global_idx].set(-1)

    new_children = updated_children
    new_next_cache = updated_next

    new_level_offsets = _recompute_level_offsets(backend, new_top_levels)

    total_inserted = int(inserted_points.shape[0])
    stats = TreeLogStats(
        num_batches=tree.stats.num_batches + 1,
        num_insertions=tree.stats.num_insertions + total_inserted,
        num_deletions=tree.stats.num_deletions,
        num_conflicts_resolved=tree.stats.num_conflicts_resolved + int(plan.conflict_graph.num_edges // 2),
    )

    new_tree = tree.replace(
        points=new_points,
        top_levels=new_top_levels,
        parents=new_parents,
        children=new_children,
        level_offsets=new_level_offsets,
        si_cache=new_si_cache,
        next_cache=new_next_cache,
        stats=stats,
    )

    return new_tree, plan


def _prefix_slices(length: int) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    size = 1
    start = 0
    while start < length:
        end = min(start + size, length)
        slices.append((start, end))
        start = end
        size = min(size * 2, length - start if length - start > 0 else size)
    return slices


def batch_insert_prefix_doubling(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: Optional[TreeBackend] = None,
    mis_seed: int | None = None,
    shuffle_seed: int | None = None,
) -> tuple[PCCTree, PrefixBatchResult]:
    """Insert a batch using prefix-doubling sub-batches.

    Randomly permutes `batch_points`, then processes prefix groups of doubling
    size (1, 2, 4, …) to mirror Algorithm 4 from Gu et al. Returns the final
    tree together with the permutation metadata for downstream inspection."""

    backend = backend or tree.backend
    batch_np = np.asarray(backend.to_numpy(batch_points))
    batch_size = batch_np.shape[0]
    if batch_size == 0:
        empty_perm = backend.asarray([], dtype=backend.default_int)
        return tree, PrefixBatchResult(permutation=empty_perm, groups=tuple())

    rng = np.random.default_rng(shuffle_seed)
    permutation = np.arange(batch_size, dtype=np.int64)
    rng.shuffle(permutation)

    permuted = batch_np[permutation]
    slices = _prefix_slices(batch_size)

    current_tree = tree
    groups: list[PrefixBatchGroup] = []
    for idx, (start, end) in enumerate(slices):
        sub_batch = permuted[start:end]
        sub_seed = (mis_seed + idx) if mis_seed is not None else None
        current_tree, plan = batch_insert(
            current_tree, sub_batch, backend=backend, mis_seed=sub_seed
        )
        group_indices = permutation[start:end]
        groups.append(
            PrefixBatchGroup(
                permutation_indices=backend.asarray(
                    group_indices, dtype=backend.default_int
                ),
                plan=plan,
            )
        )

    return current_tree, PrefixBatchResult(
        permutation=backend.asarray(permutation, dtype=backend.default_int),
        groups=tuple(groups),
    )
