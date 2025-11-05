import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex.algo import batch_insert, batch_insert_prefix_doubling, plan_batch_insert
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


def _expected_level_offsets(levels):
    levels_np = np.asarray(levels, dtype=np.int64)
    if levels_np.size == 0:
        return [0]
    counts = np.bincount(levels_np, minlength=levels_np.max() + 1)
    counts_desc = counts[::-1]
    offsets = np.concatenate([[0], np.cumsum(counts_desc)])
    return offsets.tolist()


def test_plan_batch_insert_runs_pipeline():
    tree = _setup_tree()
    batch = jnp.asarray([[2.6, 2.6], [0.5, 0.4], [3.2, 3.1]])

    plan = plan_batch_insert(tree, batch, mis_seed=0)

    assert plan.selected_indices.shape[0] <= batch.shape[0]
    assert plan.conflict_graph.num_nodes == batch.shape[0]
    assert plan.dominated_indices.shape[0] + plan.selected_indices.shape[0] == batch.shape[0]
    summary_levels = {summary.level for summary in plan.level_summaries}
    expected_levels = {int(max(level, 0)) for level in plan.traversal.levels.tolist()}
    assert summary_levels == expected_levels
    candidate_count = sum(int(summary.candidates.shape[0]) for summary in plan.level_summaries)
    assert candidate_count == batch.shape[0]
    selected_count = sum(int(summary.selected.shape[0]) for summary in plan.level_summaries)
    dominated_count = sum(int(summary.dominated.shape[0]) for summary in plan.level_summaries)
    assert selected_count == plan.selected_indices.shape[0]
    assert dominated_count == plan.dominated_indices.shape[0]

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
    assert set(plan.dominated_indices.tolist()) == dominated

    new_tree, returned_plan = batch_insert(tree, batch, mis_seed=0)
    assert returned_plan.selected_indices.tolist() == plan.selected_indices.tolist()
    assert new_tree.num_points == tree.num_points + batch.shape[0]
    assert tree.num_points == 3  # unchanged original tree

    appended = new_tree.points[-batch.shape[0] :]
    expected = jnp.concatenate(
        [batch[plan.selected_indices], batch[plan.dominated_indices]], axis=0
    )
    assert jnp.allclose(appended, expected)


def test_batch_insert_updates_level_offsets_and_stats():
    tree = _setup_tree()
    batch = jnp.asarray([[2.4, 2.4], [3.5, 3.4], [0.2, 0.1]])

    new_tree, plan = batch_insert(tree, batch, mis_seed=0)

    assert new_tree.level_offsets.tolist() == _expected_level_offsets(
        new_tree.top_levels.tolist()
    )
    assert tree.level_offsets.tolist() == _expected_level_offsets(
        tree.top_levels.tolist()
    )
    assert new_tree.num_points == tree.num_points + batch.shape[0]

    total_inserted = int(plan.selected_indices.size + plan.dominated_indices.size)
    assert new_tree.stats.num_batches == tree.stats.num_batches + 1
    assert new_tree.stats.num_insertions == tree.stats.num_insertions + total_inserted
    assert new_tree.stats.num_deletions == tree.stats.num_deletions


def test_batch_insert_preserves_original_tree_buffers():
    tree = _setup_tree()
    original_points = tree.points.tolist()
    original_top_levels = tree.top_levels.tolist()
    original_parents = tree.parents.tolist()
    original_children = tree.children.tolist()
    original_level_offsets = tree.level_offsets.tolist()

    batch = jnp.asarray([[2.4, 2.4], [0.3, 0.2]])
    new_tree, _ = batch_insert(tree, batch, mis_seed=0)

    assert tree.points.tolist() == original_points
    assert tree.top_levels.tolist() == original_top_levels
    assert tree.parents.tolist() == original_parents
    assert tree.children.tolist() == original_children
    assert tree.level_offsets.tolist() == original_level_offsets

    assert new_tree.points.shape[0] == tree.points.shape[0] + batch.shape[0]


def test_batch_insert_on_empty_tree_sets_root_level():
    backend = DEFAULT_BACKEND
    empty = PCCTree.empty(dimension=2, backend=backend)
    batch = jnp.asarray([[1.0, 1.0], [2.0, 2.0]])

    new_tree, plan = batch_insert(empty, batch, mis_seed=0)

    assert not empty.points.shape[0]
    assert new_tree.num_points == batch.shape[0]
    assert all(level >= 0 for level in new_tree.top_levels.tolist())
    assert new_tree.level_offsets.tolist() == _expected_level_offsets(
        new_tree.top_levels.tolist()
    )
    summary_levels = {summary.level for summary in plan.level_summaries}
    assert summary_levels == {0}
    assert sum(int(summary.candidates.shape[0]) for summary in plan.level_summaries) == batch.shape[0]


def test_batch_insert_splices_child_chain_for_existing_parent():
    tree = _setup_tree()
    batch = jnp.asarray([[1.1, 1.1]])

    new_tree, plan = batch_insert(tree, batch, mis_seed=0)

    selected_parents = plan.traversal.parents[plan.selected_indices].tolist()
    assert selected_parents
    parent = int(selected_parents[0])
    assert parent >= 0

    new_idx = tree.num_points
    old_child = tree.children.tolist()[parent]

    assert int(new_tree.children[parent]) == new_idx
    assert int(new_tree.next_cache[parent]) == new_idx
    expected_next = old_child if old_child >= 0 else -1
    assert int(new_tree.next_cache[new_idx]) == expected_next


def test_batch_insert_sets_child_chain_for_parent_without_children():
    tree = _setup_tree()
    batch = jnp.asarray([[2.6, 2.6]])

    new_tree, plan = batch_insert(tree, batch, mis_seed=0)

    selected_parents = plan.traversal.parents[plan.selected_indices].tolist()
    assert selected_parents
    parent = int(selected_parents[0])
    assert parent >= 0

    new_idx = tree.num_points
    assert tree.children.tolist()[parent] == -1
    assert int(new_tree.children[parent]) == new_idx
    assert int(new_tree.next_cache[parent]) == new_idx
    assert int(new_tree.next_cache[new_idx]) == -1


def test_batch_insert_redistributes_dominated_levels():
    tree = _setup_tree()
    batch = jnp.asarray([[2.6, 2.6], [2.62, 2.61], [2.64, 2.63]])

    new_tree, plan = batch_insert(tree, batch, mis_seed=0)

    dominated_count = int(plan.dominated_indices.size)
    assert dominated_count > 0
    selected_count = int(plan.selected_indices.size)
    appended_levels = new_tree.top_levels[-batch.shape[0] :]
    dominated_new = np.asarray(appended_levels[selected_count:])
    dominated_original = np.asarray(plan.traversal.levels[plan.dominated_indices])
    assert np.all(dominated_new <= dominated_original)
    positive_mask = dominated_original > 0
    assert np.all(dominated_new[positive_mask] <= dominated_original[positive_mask] - 1)

    points = np.asarray(new_tree.points)
    levels = np.asarray(new_tree.top_levels)
    for offset in range(dominated_count):
        new_idx = tree.num_points + selected_count + offset
        lvl = int(dominated_new[offset])
        anchors_mask = levels >= lvl
        anchors_mask[new_idx] = False
        anchors = points[anchors_mask]
        if anchors.size == 0:
            continue
        if lvl == 0:
            continue
        distances = np.linalg.norm(anchors - points[new_idx], axis=1)
        assert np.all(distances > (2.0 ** lvl))


def test_batch_insert_persistence_across_versions():
    tree = _setup_tree()
    original_points = np.asarray(tree.points).copy()
    batch1 = jnp.asarray([[2.4, 2.4], [0.6, 0.5]])
    tree1, plan1 = batch_insert(tree, batch1, mis_seed=0)

    expected1 = np.asarray(
        jnp.concatenate(
            [batch1[plan1.selected_indices], batch1[plan1.dominated_indices]], axis=0
        )
    )
    appended1 = np.asarray(tree1.points)[tree.num_points :]
    assert np.allclose(appended1, expected1)

    batch2 = jnp.asarray([[3.1, 3.05], [3.4, 3.35], [0.1, 0.2]])
    tree2, plan2 = batch_insert(tree1, batch2, mis_seed=1)

    expected2 = np.asarray(
        jnp.concatenate(
            [batch2[plan2.selected_indices], batch2[plan2.dominated_indices]], axis=0
        )
    )

    # Original tree untouched
    assert np.allclose(np.asarray(tree.points), original_points)

    # First version unchanged by second insert
    assert np.allclose(
        np.asarray(tree1.points)[tree.num_points :], expected1
    )

    # Second version preserves first version region and appends new batch
    assert np.allclose(np.asarray(tree2.points)[: tree1.num_points], np.asarray(tree1.points))
    assert np.allclose(
        np.asarray(tree2.points)[tree1.num_points:], expected2
    )

    # Stats monotonic
    assert tree1.stats.num_batches == tree.stats.num_batches + 1
    assert tree2.stats.num_batches == tree1.stats.num_batches + 1


def test_batch_insert_prefix_doubling_matches_manual_sequence():
    tree = _setup_tree()
    batch = jnp.asarray(
        [
            [2.1, 2.0],
            [0.4, 0.5],
            [3.0, 3.1],
            [2.2, 2.3],
            [0.8, 0.7],
            [3.5, 3.6],
        ]
    )

    mis_seed = 7
    shuffle_seed = 11

    tree_pref, result = batch_insert_prefix_doubling(
        tree, batch, mis_seed=mis_seed, shuffle_seed=shuffle_seed
    )

    assert result.permutation.shape[0] == batch.shape[0]
    assert sorted(result.permutation.tolist()) == list(range(batch.shape[0]))

    tree_manual = tree
    for idx, group in enumerate(result.groups):
        pts = batch[jnp.asarray(group.permutation_indices.tolist())]
        tree_manual, _ = batch_insert(
            tree_manual, pts, mis_seed=mis_seed + idx
        )

    assert jnp.allclose(tree_pref.points, tree_manual.points)
    assert tree_pref.stats.num_batches == tree_manual.stats.num_batches
