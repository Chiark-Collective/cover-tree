from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.logging import get_logger

LOGGER = get_logger("algo.traverse")


@dataclass(frozen=True)
class TraversalResult:
    """Structured output produced by batched traversal."""

    parents: Any
    levels: Any
    conflict_scopes: Tuple[Tuple[int, ...], ...]
    scope_indptr: Any
    scope_indices: Any


def _broadcast_batch(backend: TreeBackend, batch_points: Any) -> Any:
    return backend.asarray(batch_points, dtype=backend.default_float)


def _collect_distances(tree: PCCTree, batch: Any, backend: TreeBackend) -> Any:
    xp = backend.xp
    diff = batch[:, None, :] - tree.points[None, :, :]
    return xp.sqrt(xp.sum(diff * diff, axis=-1))


def _empty_result(backend: TreeBackend, batch_size: int) -> TraversalResult:
    xp = backend.xp
    parents = backend.asarray(
        xp.full((batch_size,), -1), dtype=backend.default_int
    )
    levels = backend.asarray(
        xp.full((batch_size,), -1), dtype=backend.default_int
    )
    conflict_scopes: Tuple[Tuple[int, ...], ...] = tuple(() for _ in range(batch_size))
    scope_indptr = backend.asarray([0] * (batch_size + 1), dtype=backend.default_int)
    scope_indices = backend.asarray([], dtype=backend.default_int)
    return TraversalResult(
        parents=parents,
        levels=levels,
        conflict_scopes=conflict_scopes,
        scope_indptr=scope_indptr,
        scope_indices=scope_indices,
    )


def _collect_next_chain(tree: PCCTree, start: int) -> Tuple[int, ...]:
    if start < 0 or start >= tree.num_points:
        return ()
    chain: list[int] = []
    visited: set[int] = set()
    current = start
    while 0 <= current < tree.num_points and current not in visited:
        chain.append(current)
        visited.add(current)
        if tree.next_cache.size == 0:
            break
        nxt = int(tree.next_cache[current])
        if nxt < 0:
            break
        current = nxt
    return tuple(chain)


def traverse_collect_scopes(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: TreeBackend | None = None,
) -> TraversalResult:
    """Compute parent assignments and conflict scopes for a batch of points."""

    backend = backend or tree.backend
    batch = _broadcast_batch(backend, batch_points)
    batch_size = int(batch.shape[0]) if batch.size else 0

    if batch_size == 0:
        return _empty_result(backend, 0)

    if tree.is_empty():
        return _empty_result(backend, batch_size)

    xp = backend.xp
    distances = _collect_distances(tree, batch, backend)

    parents = xp.argmin(distances, axis=1).astype(backend.default_int)
    levels = tree.top_levels[parents]

    base_radius = xp.power(2.0, levels.astype(backend.default_float) + 1.0)
    si_values = tree.si_cache[parents]
    radius = xp.maximum(base_radius, si_values)
    node_indices = xp.arange(tree.num_points, dtype=backend.default_int)
    parent_mask = node_indices[None, :] == parents[:, None]
    within_radius = distances <= radius[:, None]
    mask = xp.logical_or(within_radius, parent_mask)

    parents_np = backend.to_numpy(parents)
    top_levels_np = backend.to_numpy(tree.top_levels)
    chain_map = {
        parent: _collect_next_chain(tree, int(parent))
        for parent in set(int(p) for p in parents_np)
        if parent >= 0
    }

    def semisort_scope(row_mask, idx: int) -> Tuple[int, ...]:
        indices_set = {j for j, flag in enumerate(row_mask) if flag}
        parent = int(parents_np[idx])
        if parent >= 0 and parent in chain_map:
            indices_set.update(chain_map[parent])
        sorted_indices = sorted(
            indices_set,
            key=lambda j: (-int(top_levels_np[j]), j),
        )
        return tuple(sorted_indices)

    conflict_scopes_list = [
        semisort_scope(row, idx) for idx, row in enumerate(backend.to_numpy(mask))
    ]

    scope_indptr = [0]
    scope_indices: List[int] = []
    for scope in conflict_scopes_list:
        scope_indices.extend(scope)
        scope_indptr.append(len(scope_indices))

    conflict_scopes = tuple(conflict_scopes_list)
    scope_indptr_arr = backend.asarray(scope_indptr, dtype=backend.default_int)
    scope_indices_arr = backend.asarray(scope_indices, dtype=backend.default_int)

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
    )
