import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex.algo.conflict_graph import ConflictGraph, build_conflict_graph
from covertreex.algo.traverse import traverse_collect_scopes
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree, TreeLogStats


def _sample_tree():
    backend = DEFAULT_BACKEND
    points = backend.asarray([[0.0, 0.0], [2.0, 2.0]], dtype=backend.default_float)
    top_levels = backend.asarray([1, 0], dtype=backend.default_int)
    parents = backend.asarray([-1, 0], dtype=backend.default_int)
    children = backend.asarray([1, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 2], dtype=backend.default_int)
    si_cache = backend.asarray([0.0, 0.0], dtype=backend.default_float)
    next_cache = backend.asarray([1, -1], dtype=backend.default_int)
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


def test_conflict_graph_builds_edges_from_shared_scopes():
    tree = _sample_tree()
    batch_points = [[2.1, 2.1], [2.4, 2.4], [5.0, 5.0]]

    traversal = traverse_collect_scopes(tree, batch_points)
    graph = build_conflict_graph(tree, traversal, batch_points)

    assert isinstance(graph, ConflictGraph)
    assert graph.num_nodes == 3

    # First two points should connect (within radius); third is isolated.
    assert graph.indptr.tolist() == [0, 1, 2, 2]
    assert graph.indices.tolist() == [1, 0]
    assert graph.scope_indptr.tolist() == [0, 1, 2, 3]
    assert graph.scope_indices.tolist() == [1, 1, 1]
    assert graph.radii.shape == (3,)
    assert graph.annulus_bounds.shape == (3, 2)
    assert graph.annulus_bins.shape == (3,)
    assert graph.annulus_bin_indices.shape == (3,)
    assert graph.annulus_bin_indptr.tolist()[-1] == 3
