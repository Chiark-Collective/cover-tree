from typing import List, Tuple

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex.algo import build_conflict_graph, traverse_collect_scopes
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree, TreeLogStats


def _make_tree():
    backend = DEFAULT_BACKEND
    points = backend.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=backend.default_float,
    )
    top_levels = backend.asarray([2, 1, 1, 0, 0], dtype=backend.default_int)
    parents = backend.asarray([-1, 0, 0, 1, 3], dtype=backend.default_int)
    children = backend.asarray([1, 2, 3, 4, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 3, 5, 5, 5], dtype=backend.default_int)
    si_cache = backend.asarray([4.0, 2.0, 2.0, 1.0, 1.0], dtype=backend.default_float)
    next_cache = backend.asarray([1, 3, -1, 4, -1], dtype=backend.default_int)
    return PCCTree(
        points=points,
        top_levels=top_levels,
        parents=parents,
        children=children,
        level_offsets=level_offsets,
        si_cache=si_cache,
        next_cache=next_cache,
        stats=TreeLogStats(num_batches=1),
        backend=backend,
    )


def _collect_next_chain(tree: PCCTree, start: int) -> Tuple[int, ...]:
    chain: List[int] = []
    seen = set()
    current = start
    while 0 <= current < tree.num_points and current not in seen:
        chain.append(current)
        seen.add(current)
        nxt = int(tree.next_cache[current]) if tree.next_cache.size else -1
        if nxt < 0:
            break
        current = nxt
    return tuple(chain)


def test_traversal_matches_naive_computation():
    tree = _make_tree()
    batch = jnp.asarray([[2.2, 2.1], [0.9, 0.8], [3.1, 3.0]])

    result = traverse_collect_scopes(tree, batch)

    distances = jnp.linalg.norm(batch[:, None, :] - tree.points[None, :, :], axis=-1)
    parents_naive = jnp.argmin(distances, axis=1)
    levels_naive = tree.top_levels[parents_naive]

    assert tuple(result.parents.tolist()) == tuple(int(x) for x in parents_naive.tolist())
    assert tuple(result.levels.tolist()) == tuple(int(x) for x in levels_naive.tolist())

    radii = []
    for parent, level in zip(parents_naive.tolist(), levels_naive.tolist()):
        base = 2.0 ** (float(level) + 1.0)
        si = float(tree.si_cache[parent])
        radii.append(max(base, si))

    scopes = []
    for idx, parent in enumerate(parents_naive.tolist()):
        mask = []
        for node_idx in range(tree.num_points):
            within = distances[idx, node_idx] <= radii[idx] + 1e-9
            if within or node_idx == parent:
                mask.append(node_idx)
        mask = set(mask)
        mask.update(_collect_next_chain(tree, parent))
        scopes.append(
            tuple(sorted(mask, key=lambda j: (-int(tree.top_levels[j]), j)))
        )

    assert scopes == list(result.conflict_scopes)

    indptr = result.scope_indptr.tolist()
    indices = result.scope_indices.tolist()
    reconstructed = [
        tuple(indices[indptr[i] : indptr[i + 1]]) for i in range(len(scopes))
    ]
    assert reconstructed == scopes


def test_conflict_graph_matches_bruteforce_edges():
    tree = _make_tree()
    batch = jnp.asarray([[2.2, 2.1], [0.9, 0.8], [3.1, 3.0], [5.0, 5.0]])

    traversal = traverse_collect_scopes(tree, batch)
    graph = build_conflict_graph(tree, traversal, batch)

    distances = jnp.linalg.norm(batch[:, None, :] - batch[None, :, :], axis=-1)
    radii = []
    for parent, level in zip(traversal.parents.tolist(), traversal.levels.tolist()):
        if parent < 0:
            radii.append(float("inf"))
        else:
            base = 2.0 ** (float(level) + 1.0)
            si = float(tree.si_cache[parent])
            radii.append(max(base, si))

    scopes = traversal.conflict_scopes
    brute_edges = []
    for i in range(batch.shape[0]):
        for j in range(i + 1, batch.shape[0]):
            if not set(scopes[i]).intersection(scopes[j]):
                continue
            if float(distances[i, j]) <= min(radii[i], radii[j]) + 1e-9:
                brute_edges.append((i, j))

    # Reconstruct edges from CSR
    indptr = graph.indptr.tolist()
    indices = graph.indices.tolist()
    csr_edges = []
    for node in range(len(indptr) - 1):
        neighbors = indices[indptr[node] : indptr[node + 1]]
        for nb in neighbors:
            if node < nb:
                csr_edges.append((node, nb))

    assert sorted(brute_edges) == sorted(csr_edges)

    # Scope buffers must match traversal buffers
    assert graph.scope_indptr.tolist() == traversal.scope_indptr.tolist()
    assert graph.scope_indices.tolist() == traversal.scope_indices.tolist()


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_randomized_structural_invariants(seed):
    rng = jax.random.PRNGKey(seed)
    backend = DEFAULT_BACKEND

    num_points = 12
    points = jax.random.uniform(rng, (num_points, 3), dtype=backend.default_float)
    levels = jnp.arange(num_points, dtype=backend.default_int)[::-1] % 4
    parents = jnp.roll(jnp.arange(num_points, dtype=backend.default_int), -1)
    parents = parents.at[-1].set(-1)
    si_cache = jnp.full((num_points,), 2.0, dtype=backend.default_float)
    next_cache = jnp.roll(jnp.arange(num_points, dtype=backend.default_int), -1)
    next_cache = next_cache.at[-1].set(-1)
    level_offsets = jnp.concatenate(
        [jnp.asarray([0], dtype=backend.default_int), jnp.cumsum(jnp.ones((5,), dtype=backend.default_int) * (num_points // 4 + 1))]
    )

    tree = PCCTree(
        points=backend.asarray(points, dtype=backend.default_float),
        top_levels=backend.asarray(levels, dtype=backend.default_int),
        parents=backend.asarray(parents, dtype=backend.default_int),
        children=backend.asarray(jnp.roll(jnp.arange(num_points, dtype=backend.default_int), -1), dtype=backend.default_int),
        level_offsets=backend.asarray(level_offsets, dtype=backend.default_int),
        si_cache=backend.asarray(si_cache, dtype=backend.default_float),
        next_cache=backend.asarray(next_cache, dtype=backend.default_int),
        stats=TreeLogStats(num_batches=1),
        backend=backend,
    )

    batch = jax.random.uniform(rng, (5, 3), dtype=backend.default_float)
    traversal = traverse_collect_scopes(tree, batch)
    graph = build_conflict_graph(tree, traversal, batch)

    # Reconstruct scopes
    indptr = traversal.scope_indptr.tolist()
    indices = traversal.scope_indices.tolist()
    for idx, scope in enumerate(traversal.conflict_scopes):
        reconstructed = tuple(indices[indptr[idx] : indptr[idx + 1]])
        assert reconstructed == scope

    # Edge checks
    pairwise = jnp.linalg.norm(batch[:, None, :] - batch[None, :, :], axis=-1)
    radii = []
    for parent, level in zip(traversal.parents.tolist(), traversal.levels.tolist()):
        if parent < 0:
            radii.append(float("inf"))
        else:
            base = 2.0 ** (float(level) + 1.0)
            si = float(tree.si_cache[parent])
            radii.append(max(base, si))

    brute_edges = set()
    for i in range(batch.shape[0]):
        for j in range(i + 1, batch.shape[0]):
            if set(traversal.conflict_scopes[i]).intersection(traversal.conflict_scopes[j]):
                if float(pairwise[i, j]) <= min(radii[i], radii[j]) + 1e-9:
                    brute_edges.add((i, j))

    csr_edges = set()
    indptr_graph = graph.indptr.tolist()
    indices_graph = graph.indices.tolist()
    for node in range(len(indptr_graph) - 1):
        neighbors = indices_graph[indptr_graph[node] : indptr_graph[node + 1]]
        for nb in neighbors:
            if node < nb:
                csr_edges.add((node, nb))

    assert brute_edges == csr_edges
    assert graph.annulus_bins.shape == (batch.shape[0],)
    assert graph.annulus_bin_indices.shape == (batch.shape[0],)
    assert graph.annulus_bin_indptr.tolist()[-1] == batch.shape[0]
