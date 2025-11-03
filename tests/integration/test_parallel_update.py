import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex.algo import batch_insert, plan_batch_insert
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree, TreeLogStats


def _setup_tree():
    backend = DEFAULT_BACKEND
    points = backend.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.5, 2.5],
        ],
        dtype=backend.default_float,
    )
    top_levels = backend.asarray([2, 1, 0], dtype=backend.default_int)
    parents = backend.asarray([-1, 0, 1], dtype=backend.default_int)
    children = backend.asarray([1, 2, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 2, 3], dtype=backend.default_int)
    si_cache = backend.asarray([4.0, 2.0, 1.0], dtype=backend.default_float)
    next_cache = backend.asarray([1, 2, -1], dtype=backend.default_int)
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


def test_plan_batch_insert_runs_pipeline():
    tree = _setup_tree()
    batch = jnp.asarray([[2.6, 2.6], [0.5, 0.4], [3.2, 3.1]])

    plan = plan_batch_insert(tree, batch, mis_seed=0)

    assert plan.selected_indices.shape[0] <= batch.shape[0]
    assert plan.conflict_graph.num_nodes == batch.shape[0]

    indicator = plan.mis_result.independent_set.tolist()
    selected = {idx for idx, flag in enumerate(indicator) if flag == 1}

    indptr = plan.conflict_graph.indptr.tolist()
    indices = plan.conflict_graph.indices.tolist()
    for node in selected:
        neighbors = set(indices[indptr[node] : indptr[node + 1]])
        assert neighbors.isdisjoint(selected)

    dominated = set(range(batch.shape[0])) - selected
    for node in dominated:
        neighbors = set(indices[indptr[node] : indptr[node + 1]])
        assert neighbors & selected

    new_tree, returned_plan = batch_insert(tree, batch, mis_seed=0)
    assert returned_plan.selected_indices.tolist() == plan.selected_indices.tolist()
    assert new_tree.num_points == tree.num_points + len(plan.selected_indices)
    assert tree.num_points == 3  # unchanged original tree

    appended = new_tree.points[-len(plan.selected_indices) :]
    expected = batch[plan.selected_indices]
    assert jnp.allclose(appended, expected)
