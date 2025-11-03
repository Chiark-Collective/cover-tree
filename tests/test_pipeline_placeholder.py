import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex.algo import (
    build_conflict_graph,
    run_mis,
    traverse_collect_scopes,
)
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree, TreeLogStats


def _sample_tree():
    backend = DEFAULT_BACKEND
    points = backend.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=backend.default_float)
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


def test_placeholder_pipeline_executes():
    tree = _sample_tree()
    batch_points = [[2.0, 2.0], [3.0, 3.0]]

    traversal = traverse_collect_scopes(tree, batch_points)
    graph = build_conflict_graph(tree, traversal, batch_points)
    mis_result = run_mis(tree.backend, graph)

    assert graph.num_nodes == len(traversal.conflict_scopes) == 2
    assert mis_result.independent_set.shape == (2,)
