from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from covertreex import config as cx_config
from covertreex.algo.conflict_graph import ConflictGraph
from covertreex.core.tree import TreeBackend


@dataclass(frozen=True)
class MISResult:
    independent_set: Any
    iterations: int


def _repeat_nodes(indptr: jnp.ndarray, dtype) -> jnp.ndarray:
    degrees = indptr[1:] - indptr[:-1]
    return jnp.repeat(jnp.arange(degrees.shape[0], dtype=dtype), degrees)


def run_mis(
    backend: TreeBackend,
    graph: ConflictGraph,
    *,
    seed: int | None = None,
) -> MISResult:
    """Luby-style MIS using JAX primitives."""

    num_nodes = graph.num_nodes
    if num_nodes == 0:
        empty = backend.asarray([], dtype=backend.default_int)
        return MISResult(independent_set=empty, iterations=0)

    runtime_seed = seed
    if runtime_seed is None:
        runtime_seed = cx_config.runtime_config().mis_seed or 0

    key = jax.random.PRNGKey(runtime_seed)
    indptr = graph.indptr
    indices = graph.indices
    dtype_int = backend.default_int
    dtype_float = backend.default_float

    sources = _repeat_nodes(indptr, dtype_int)

    def cond_fn(state):
        _, active, _, _ = state
        return jnp.any(active)

    def body_fn(state):
        key, active, selected, iterations = state
        key, subkey = jax.random.split(key)
        priorities = jax.random.uniform(subkey, (num_nodes,), dtype=dtype_float)

        neighbor_priorities = priorities[indices]
        neighbor_active = active[indices]
        neg_inf = jnp.full_like(neighbor_priorities, -jnp.inf)
        neighbor_priorities = jnp.where(neighbor_active, neighbor_priorities, neg_inf)

        max_neighbor = jnp.full((num_nodes,), -jnp.inf, dtype=dtype_float)
        max_neighbor = max_neighbor.at[sources].max(neighbor_priorities)

        winner_mask = jnp.logical_and(active, priorities >= max_neighbor)
        no_winner = jnp.logical_not(jnp.any(winner_mask))
        candidate_indices = jnp.where(active, jnp.arange(num_nodes, dtype=dtype_int), jnp.full((num_nodes,), num_nodes, dtype=dtype_int))
        fallback_idx = jnp.min(candidate_indices)
        fallback_mask = jnp.arange(num_nodes, dtype=dtype_int) == fallback_idx
        winner_mask = jnp.logical_or(winner_mask, jnp.logical_and(no_winner, fallback_mask))

        selected = jnp.logical_or(selected, winner_mask)

        winner_sources = winner_mask[sources]
        dominated = jnp.zeros_like(active)
        dominated = dominated.at[indices].max(winner_sources)

        active = jnp.logical_and(active, jnp.logical_not(winner_mask))
        active = jnp.logical_and(active, jnp.logical_not(dominated))

        iterations = iterations + jnp.int32(1)
        return key, active, selected, iterations

    init_state = (
        key,
        jnp.ones((num_nodes,), dtype=bool),
        jnp.zeros((num_nodes,), dtype=bool),
        jnp.int32(0),
    )
    _, _, selected, iterations = jax.lax.while_loop(cond_fn, body_fn, init_state)

    indicator = selected.astype(dtype_int)
    indicator = backend.device_put(indicator)
    iterations_int = int(iterations)

    return MISResult(independent_set=indicator, iterations=iterations_int)
