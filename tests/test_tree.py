import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex import config as cx_config

cx_config.reset_runtime_config_cache()

from covertreex.core.tree import (
    PCCTree,
    TreeBackend,
    TreeLogStats,
    get_runtime_backend,
)


def _sample_tree() -> PCCTree:
    backend = get_runtime_backend()
    points = backend.array([[0.0, 0.0], [1.0, 1.0]])
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
        stats=TreeLogStats(num_batches=1, num_insertions=2),
        backend=backend,
    )


def test_empty_tree_has_expected_shapes():
    tree = PCCTree.empty(dimension=3)
    assert tree.is_empty()
    assert tree.dimension == 3
    assert tree.num_points == 0
    assert tree.level_offsets.shape == (1,)
    tree.validate()  # smoke-test shape checks


def test_replace_returns_new_instance():
    tree = PCCTree.empty(dimension=2)
    extended_offsets = tree.backend.asarray([0, 0], dtype=tree.backend.default_int)
    new_tree = tree.replace(level_offsets=extended_offsets)
    assert new_tree is not tree
    assert new_tree.level_offsets.shape == (2,)
    assert tree.level_offsets.shape == (1,)  # original remains untouched


def test_materialise_roundtrip():
    tree = _sample_tree()
    snapshot = tree.materialise()
    assert snapshot["backend"] == get_runtime_backend().name
    assert snapshot["points"].shape == (2, 2)
    assert snapshot["stats"]["num_insertions"] == 2


def test_to_backend_respects_precision():
    low_precision = TreeBackend.jax(precision="float32")
    tree = _sample_tree().to_backend(low_precision)
    assert tree.backend.default_float == jnp.float32
    assert tree.points.dtype == jnp.float32


def test_tree_backend_rejects_precision():
    with pytest.raises(ValueError):
        TreeBackend.jax(precision="float16")
