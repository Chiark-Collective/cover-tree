from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Tuple

import numpy as np

from covertreex import config as cx_config
from covertreex.core._persistence_numba import (
    NUMBA_PERSISTENCE_AVAILABLE,
    apply_journal_cow,
)
from covertreex.core.tree import PCCTree, TreeBackend


@dataclass(frozen=True)
class SliceUpdate:
    """Descriptor for a copy-on-write update applied to a single array."""

    index: Tuple[Any, ...] | Any
    values: Any


def _ensure_array(backend: TreeBackend, values: Any, *, dtype: Any) -> Any:
    return backend.asarray(values, dtype=dtype)


def _normalise_index(index: Tuple[int, ...] | Tuple[slice, ...] | int | slice) -> Tuple[Any, ...]:
    if isinstance(index, tuple):
        return index
    return (index,)


def _required_length_along_axis(index: Tuple[Any, ...], current: int) -> int:
    if not index:
        return current
    primary = index[0]
    if isinstance(primary, slice):
        if primary.stop is None:
            return current
        return max(current, int(primary.stop))
    if isinstance(primary, int):
        if primary < 0:
            return current
        return max(current, int(primary) + 1)
    return current


def clone_array_segment(
    backend: TreeBackend, source: Any, updates: Iterable[SliceUpdate], *, dtype: Any
) -> Any:
    """Clone `source` and apply `updates` without mutating the original array."""

    target = backend.asarray(source, dtype=dtype)
    xp = backend.xp
    updates_list = list(updates)

    if updates_list:
        required_length = target.shape[0] if target.ndim >= 1 else 0
        for update in updates_list:
            index = _normalise_index(update.index)
            required_length = _required_length_along_axis(index, required_length)

        if target.ndim >= 1 and required_length > target.shape[0]:
            pad_shape = list(target.shape)
            pad_shape[0] = required_length - target.shape[0]
            pad = xp.zeros(tuple(pad_shape), dtype=dtype)
            target = xp.concatenate([target, pad], axis=0)

        for update in updates_list:
            index = _normalise_index(update.index)
            values = _ensure_array(backend, update.values, dtype=target.dtype)
            if hasattr(target, "at"):
                target = target.at[index].set(values)
            else:
                target = np.array(target, copy=True)
                target[index] = values

    return backend.device_put(target)


def clone_tree_with_updates(
    tree: PCCTree,
    *,
    points_updates: Iterable[SliceUpdate] = (),
    top_level_updates: Iterable[SliceUpdate] = (),
    parent_updates: Iterable[SliceUpdate] = (),
    child_updates: Iterable[SliceUpdate] = (),
    level_offset_updates: Iterable[SliceUpdate] = (),
    si_cache_updates: Iterable[SliceUpdate] = (),
    next_cache_updates: Iterable[SliceUpdate] = (),
) -> PCCTree:
    """Produce a new `PCCTree` with updates applied via copy-on-write semantics."""

    backend = tree.backend
    return tree.replace(
        points=clone_array_segment(
            backend, tree.points, points_updates, dtype=backend.default_float
        ),
        top_levels=clone_array_segment(
            backend, tree.top_levels, top_level_updates, dtype=backend.default_int
        ),
        parents=clone_array_segment(
            backend, tree.parents, parent_updates, dtype=backend.default_int
        ),
        children=clone_array_segment(
            backend, tree.children, child_updates, dtype=backend.default_int
        ),
        level_offsets=clone_array_segment(
            backend,
            tree.level_offsets,
            level_offset_updates,
            dtype=backend.default_int,
        ),
        si_cache=clone_array_segment(
            backend, tree.si_cache, si_cache_updates, dtype=backend.default_float
        ),
        next_cache=clone_array_segment(
            backend, tree.next_cache, next_cache_updates, dtype=backend.default_int
        ),
    )


def _pool_next_capacity(current: int, required: int) -> int:
    if required <= 0:
        return 0
    if current == 0:
        return max(16, required)
    return max(required, current * 2)


@dataclass
class JournalScratchPool:
    """Reusable scratch buffers for persistence journals."""

    min_capacity: int = 16
    _head_parents: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    _head_values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    _next_nodes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    _next_values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    _children: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))

    def _request(self, name: str, required: int) -> np.ndarray:
        buffer = getattr(self, name)
        if required <= 0:
            return buffer[:0]
        current = buffer.shape[0]
        if current < required:
            target = max(self.min_capacity, _pool_next_capacity(current, required))
            buffer = np.empty(target, dtype=np.int32)
            setattr(self, name, buffer)
        return buffer[:required]

    def head_parents(self, required: int) -> np.ndarray:
        return self._request("_head_parents", required)

    def head_values(self, required: int) -> np.ndarray:
        return self._request("_head_values", required)

    def next_nodes(self, required: int) -> np.ndarray:
        return self._request("_next_nodes", required)

    def next_values(self, required: int) -> np.ndarray:
        return self._request("_next_values", required)

    def children(self, required: int) -> np.ndarray:
        return self._request("_children", required)


DEFAULT_JOURNAL_POOL = JournalScratchPool()


@dataclass(frozen=True)
class PersistenceJournal:
    """Immutable snapshot of batched persistence mutations."""

    base_length: int
    inserted_points: np.ndarray
    inserted_levels: np.ndarray
    inserted_parents: np.ndarray
    inserted_si: np.ndarray
    head_parents: np.ndarray
    head_values: np.ndarray
    next_nodes: np.ndarray
    next_values: np.ndarray
    level_offsets: np.ndarray

    @property
    def inserted_count(self) -> int:
        return int(self.inserted_levels.shape[0])

    @property
    def total_length(self) -> int:
        return self.base_length + self.inserted_count


def _compute_level_offsets_incremental(
    current_offsets: np.ndarray, inserted_levels: np.ndarray
) -> np.ndarray:
    offsets_np = np.asarray(current_offsets, dtype=np.int64)
    inserted_np = np.asarray(inserted_levels, dtype=np.int64)

    if offsets_np.size <= 1:
        counts = np.zeros(1, dtype=np.int64)
    else:
        diffs = np.diff(offsets_np).astype(np.int64, copy=False)
        if diffs.size == 0:
            counts = np.zeros(1, dtype=np.int64)
        else:
            counts = diffs[::-1]

    max_existing = counts.shape[0] - 1
    max_inserted = int(np.max(inserted_np)) if inserted_np.size else -1
    target_level = max(max_existing, max_inserted)

    if target_level < 0:
        counts = np.zeros(1, dtype=np.int64)
    elif counts.shape[0] <= target_level:
        pad = target_level + 1 - counts.shape[0]
        counts = np.pad(counts, (0, pad))
    else:
        counts = counts[: target_level + 1]

    for level in inserted_np:
        lvl = int(level)
        if lvl < 0:
            continue
        if lvl >= counts.shape[0]:
            pad = lvl - counts.shape[0] + 1
            counts = np.pad(counts, (0, pad))
        counts[lvl] += 1

    if counts.sum() == 0:
        return np.asarray([0], dtype=np.int64)

    counts_desc = counts[::-1]
    result = np.empty(counts_desc.shape[0] + 1, dtype=np.int64)
    result[0] = 0
    np.cumsum(counts_desc, out=result[1:])
    return result


def _build_head_and_sibling_updates(
    inserted_parents: np.ndarray,
    base_children: np.ndarray,
    *,
    base_length: int,
    pool: JournalScratchPool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inserted_count = inserted_parents.shape[0]
    if inserted_count == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty, empty

    total_length = base_length + inserted_count

    current_children_buffer = pool.children(total_length)
    current_children = current_children_buffer[:total_length]
    np.copyto(current_children[:base_length], base_children[:base_length])
    if inserted_count:
        current_children[base_length:total_length] = -1

    head_parents_buffer = pool.head_parents(inserted_count)
    head_values_buffer = pool.head_values(inserted_count)
    next_nodes_buffer = pool.next_nodes(inserted_count)
    next_values_buffer = pool.next_values(inserted_count)

    head_count = 0
    for offset in range(inserted_count):
        parent = int(inserted_parents[offset])
        global_idx = base_length + offset
        next_nodes_buffer[offset] = global_idx

        if parent < 0 or parent >= total_length:
            next_values_buffer[offset] = -1
            continue

        prev_child = int(current_children[parent])
        head_parents_buffer[head_count] = parent
        head_values_buffer[head_count] = global_idx
        head_count += 1
        next_values_buffer[offset] = prev_child
        current_children[parent] = global_idx

    return (
        head_parents_buffer[:head_count],
        head_values_buffer[:head_count],
        next_nodes_buffer[:inserted_count],
        next_values_buffer[:inserted_count],
    )


def build_persistence_journal(
    tree: PCCTree,
    *,
    backend: TreeBackend | None = None,
    inserted_points: np.ndarray,
    inserted_levels: np.ndarray,
    inserted_parents: np.ndarray,
    inserted_si: np.ndarray,
    pool: JournalScratchPool | None = None,
) -> PersistenceJournal:
    backend = backend or tree.backend
    pool = pool or DEFAULT_JOURNAL_POOL

    inserted_points_np = np.asarray(inserted_points, dtype=float)
    inserted_levels_np = np.asarray(inserted_levels, dtype=np.int64)
    inserted_parents_np = np.asarray(inserted_parents, dtype=np.int64)
    inserted_si_np = np.asarray(inserted_si, dtype=float)

    base_length = tree.num_points
    children_np = np.asarray(backend.to_numpy(tree.children), dtype=np.int32)

    (
        head_parents_view,
        head_values_view,
        next_nodes_view,
        next_values_view,
    ) = _build_head_and_sibling_updates(
        inserted_parents_np.astype(np.int32, copy=False),
        children_np,
        base_length=base_length,
        pool=pool,
    )

    level_offsets_np = np.asarray(backend.to_numpy(tree.level_offsets), dtype=np.int64)
    next_level_offsets = _compute_level_offsets_incremental(
        level_offsets_np, inserted_levels_np
    )

    return PersistenceJournal(
        base_length=base_length,
        inserted_points=np.array(inserted_points_np, copy=True, order="C"),
        inserted_levels=inserted_levels_np.astype(np.int32, copy=True),
        inserted_parents=inserted_parents_np.astype(np.int32, copy=True),
        inserted_si=inserted_si_np.astype(float, copy=True),
        head_parents=np.array(head_parents_view, copy=True),
        head_values=np.array(head_values_view, copy=True),
        next_nodes=np.array(next_nodes_view, copy=True),
        next_values=np.array(next_values_view, copy=True),
        level_offsets=next_level_offsets.astype(np.int64, copy=True),
    )


def apply_persistence_journal(
    tree: PCCTree,
    journal: PersistenceJournal,
    *,
    backend: TreeBackend | None = None,
) -> PCCTree:
    backend = backend or tree.backend
    runtime = cx_config.runtime_config()

    if backend.name == "numpy":
        if runtime.enable_numba and NUMBA_PERSISTENCE_AVAILABLE:
            return _apply_journal_numba(tree, journal, backend=backend)
        return _apply_journal_numpy(tree, journal, backend=backend)

    return _apply_journal_clone(tree, journal, backend=backend)


def _apply_journal_numba(
    tree: PCCTree,
    journal: PersistenceJournal,
    *,
    backend: TreeBackend,
) -> PCCTree:
    base_length = journal.base_length
    inserted_count = journal.inserted_count
    total_length = journal.total_length

    points_np = np.asarray(backend.to_numpy(tree.points), dtype=float)
    parents_np = np.asarray(backend.to_numpy(tree.parents), dtype=np.int32)
    levels_np = np.asarray(backend.to_numpy(tree.top_levels), dtype=np.int32)
    children_np = np.asarray(backend.to_numpy(tree.children), dtype=np.int32)
    next_np = np.asarray(backend.to_numpy(tree.next_cache), dtype=np.int32)
    si_cache_np = np.asarray(backend.to_numpy(tree.si_cache), dtype=float)

    parents_out = np.empty(total_length, dtype=np.int32)
    levels_out = np.empty(total_length, dtype=np.int32)
    children_out = np.empty(total_length, dtype=np.int32)
    next_out = np.empty(total_length, dtype=np.int32)

    apply_journal_cow(
        parents_np,
        levels_np,
        children_np,
        next_np,
        journal.inserted_parents.astype(np.int32, copy=False),
        journal.inserted_levels.astype(np.int32, copy=False),
        journal.head_parents.astype(np.int32, copy=False),
        journal.head_values.astype(np.int32, copy=False),
        journal.next_nodes.astype(np.int32, copy=False),
        journal.next_values.astype(np.int32, copy=False),
        parents_out,
        levels_out,
        children_out,
        next_out,
        base_length,
    )

    if inserted_count:
        points_updated = np.concatenate([points_np, journal.inserted_points], axis=0)
        si_updated = np.concatenate([si_cache_np, journal.inserted_si], axis=0)
    else:
        points_updated = np.array(points_np, copy=True)
        si_updated = np.array(si_cache_np, copy=True)

    level_offsets_updated = journal.level_offsets

    return tree.replace(
        points=backend.asarray(points_updated, dtype=backend.default_float),
        top_levels=backend.asarray(levels_out, dtype=backend.default_int),
        parents=backend.asarray(parents_out, dtype=backend.default_int),
        children=backend.asarray(children_out, dtype=backend.default_int),
        level_offsets=backend.asarray(level_offsets_updated, dtype=backend.default_int),
        si_cache=backend.asarray(si_updated, dtype=backend.default_float),
        next_cache=backend.asarray(next_out, dtype=backend.default_int),
    )


def _apply_journal_numpy(
    tree: PCCTree,
    journal: PersistenceJournal,
    *,
    backend: TreeBackend,
) -> PCCTree:
    base_length = journal.base_length
    inserted_count = journal.inserted_count
    total_length = journal.total_length

    points_np = np.asarray(backend.to_numpy(tree.points), dtype=float)
    parents_np = np.asarray(backend.to_numpy(tree.parents), dtype=np.int32)
    levels_np = np.asarray(backend.to_numpy(tree.top_levels), dtype=np.int32)
    children_np = np.asarray(backend.to_numpy(tree.children), dtype=np.int32)
    next_np = np.asarray(backend.to_numpy(tree.next_cache), dtype=np.int32)
    si_cache_np = np.asarray(backend.to_numpy(tree.si_cache), dtype=float)

    parents_out = np.empty(total_length, dtype=np.int32)
    levels_out = np.empty(total_length, dtype=np.int32)
    children_out = np.empty(total_length, dtype=np.int32)
    next_out = np.empty(total_length, dtype=np.int32)

    parents_out[:base_length] = parents_np
    levels_out[:base_length] = levels_np
    children_out[:base_length] = children_np
    next_out[:base_length] = next_np

    if inserted_count:
        parents_out[base_length:] = journal.inserted_parents.astype(np.int32, copy=False)
        levels_out[base_length:] = journal.inserted_levels.astype(np.int32, copy=False)
        children_out[base_length:] = -1
        next_out[base_length:] = -1

    for parent, child in zip(journal.head_parents, journal.head_values):
        if 0 <= parent < children_out.shape[0]:
            children_out[int(parent)] = int(child)

    for node, value in zip(journal.next_nodes, journal.next_values):
        if 0 <= node < next_out.shape[0]:
            next_out[int(node)] = int(value)

    if inserted_count:
        points_updated = np.concatenate([points_np, journal.inserted_points], axis=0)
        si_updated = np.concatenate([si_cache_np, journal.inserted_si], axis=0)
    else:
        points_updated = np.array(points_np, copy=True)
        si_updated = np.array(si_cache_np, copy=True)

    level_offsets_updated = journal.level_offsets

    return tree.replace(
        points=backend.asarray(points_updated, dtype=backend.default_float),
        top_levels=backend.asarray(levels_out, dtype=backend.default_int),
        parents=backend.asarray(parents_out, dtype=backend.default_int),
        children=backend.asarray(children_out, dtype=backend.default_int),
        level_offsets=backend.asarray(level_offsets_updated, dtype=backend.default_int),
        si_cache=backend.asarray(si_updated, dtype=backend.default_float),
        next_cache=backend.asarray(next_out, dtype=backend.default_int),
    )


def _apply_journal_clone(
    tree: PCCTree,
    journal: PersistenceJournal,
    *,
    backend: TreeBackend,
) -> PCCTree:
    inserted_count = journal.inserted_count
    base_index = journal.base_length
    append_slice = slice(base_index, base_index + inserted_count)
    xp = backend.xp

    points_updates: list[SliceUpdate] = []
    top_level_updates: list[SliceUpdate] = []
    parent_updates: list[SliceUpdate] = []
    child_updates: list[SliceUpdate] = []
    next_updates: list[SliceUpdate] = []
    si_updates: list[SliceUpdate] = []

    if inserted_count:
        points_updates.append(
            SliceUpdate(
                index=(append_slice, slice(None)),
                values=backend.asarray(journal.inserted_points, dtype=backend.default_float),
            )
        )
        top_level_updates.append(
            SliceUpdate(
                index=(append_slice,),
                values=backend.asarray(journal.inserted_levels, dtype=backend.default_int),
            )
        )
        parent_updates.append(
            SliceUpdate(
                index=(append_slice,),
                values=backend.asarray(journal.inserted_parents, dtype=backend.default_int),
            )
        )
        si_updates.append(
            SliceUpdate(
                index=(append_slice,),
                values=backend.asarray(journal.inserted_si, dtype=backend.default_float),
            )
        )
        default_child_block = xp.full((inserted_count,), -1, dtype=backend.default_int)
        child_updates.append(SliceUpdate(index=(append_slice,), values=default_child_block))
        default_next_block = xp.full((inserted_count,), -1, dtype=backend.default_int)
        next_updates.append(SliceUpdate(index=(append_slice,), values=default_next_block))

    for parent, child in zip(journal.head_parents, journal.head_values):
        child_updates.append(SliceUpdate(index=(int(parent),), values=int(child)))

    for node, value in zip(journal.next_nodes, journal.next_values):
        next_updates.append(SliceUpdate(index=(int(node),), values=int(value)))

    level_offset_updates = [
        SliceUpdate(
            index=(slice(0, journal.level_offsets.shape[0]),),
            values=backend.asarray(journal.level_offsets, dtype=backend.default_int),
        )
    ]

    return clone_tree_with_updates(
        tree,
        points_updates=points_updates,
        top_level_updates=top_level_updates,
        parent_updates=parent_updates,
        child_updates=child_updates,
        level_offset_updates=level_offset_updates,
        si_cache_updates=si_updates,
        next_cache_updates=next_updates,
    )


__all__ = [
    "JournalScratchPool",
    "PersistenceJournal",
    "SliceUpdate",
    "apply_persistence_journal",
    "build_persistence_journal",
    "clone_array_segment",
    "clone_tree_with_updates",
    "DEFAULT_JOURNAL_POOL",
]
