from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import jax.numpy as jnp

from covertreex.core.tree import PCCTree, TreeBackend


@dataclass(frozen=True)
class SliceUpdate:
    """Descriptor for a copy-on-write update applied to a single array."""

    index: Tuple[int, ...]
    values: Any


def _ensure_array(backend: TreeBackend, values: Any, *, dtype: Any) -> Any:
    return backend.asarray(values, dtype=dtype)


def clone_array_segment(
    backend: TreeBackend, source: Any, updates: Iterable[SliceUpdate], *, dtype: Any
) -> Any:
    """Clone `source` and apply `updates` without mutating the original array."""

    target = backend.asarray(source, dtype=dtype)
    for update in updates:
        target = target.at[update.index].set(
            _ensure_array(backend, update.values, dtype=target.dtype)
        )
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
