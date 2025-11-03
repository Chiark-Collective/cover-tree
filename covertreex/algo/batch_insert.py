from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

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
    return BatchInsertPlan(
        traversal=traversal,
        conflict_graph=conflict_graph,
        mis_result=mis_result,
        selected_indices=selected_indices,
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

    if plan.selected_indices.size == 0:
        return tree, plan

    xp = backend.xp
    batch = backend.asarray(batch_points, dtype=backend.default_float)
    selected_points = batch[plan.selected_indices]
    selected_levels = plan.traversal.levels[plan.selected_indices]
    selected_parents = plan.traversal.parents[plan.selected_indices]
    selected_si = plan.conflict_graph.radii[plan.selected_indices]

    new_points = xp.concatenate([tree.points, selected_points], axis=0)
    new_top_levels = xp.concatenate([tree.top_levels, selected_levels], axis=0)
    new_parents = xp.concatenate([tree.parents, selected_parents], axis=0)
    new_children = xp.concatenate(
        [
            tree.children,
            xp.full((plan.selected_indices.size,), -1, dtype=backend.default_int),
        ],
        axis=0,
    )
    new_si_cache = xp.concatenate([tree.si_cache, selected_si], axis=0)
    new_next_cache = xp.concatenate(
        [
            tree.next_cache,
            xp.full((plan.selected_indices.size,), -1, dtype=backend.default_int),
        ],
        axis=0,
    )

    stats = TreeLogStats(
        num_batches=tree.stats.num_batches + 1,
        num_insertions=tree.stats.num_insertions + int(plan.selected_indices.size),
        num_deletions=tree.stats.num_deletions,
        num_conflicts_resolved=tree.stats.num_conflicts_resolved + int(plan.conflict_graph.num_edges // 2),
    )

    new_tree = tree.replace(
        points=new_points,
        top_levels=new_top_levels,
        parents=new_parents,
        children=new_children,
        si_cache=new_si_cache,
        next_cache=new_next_cache,
        stats=stats,
    )

    return new_tree, plan
