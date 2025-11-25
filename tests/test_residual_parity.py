from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
from numpy.random import default_rng

from covertreex import config as cx_config
from covertreex.engine import build_tree
from covertreex.metrics.residual import build_residual_backend
from covertreex.queries.knn import knn

try:
    import covertreex_backend  # type: ignore
except ImportError:  # pragma: no cover - optional backend
    covertreex_backend = None

pytestmark = pytest.mark.skipif(covertreex_backend is None, reason="Rust backend not built")


def _parity_context(monkeypatch: pytest.MonkeyPatch, *, enable_rust: bool) -> cx_config.RuntimeContext:
    monkeypatch.setenv("COVERTREEX_PRESET", "residual_parity")
    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    monkeypatch.setenv("COVERTREEX_ENABLE_RUST", "1" if enable_rust else "0")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_PARITY", "1")
    monkeypatch.setenv("COVERTREEX_RUST_QUERY_TELEMETRY", "1")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_DISABLE_FAST_PATHS", "1")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_USE_STATIC_EUCLIDEAN_TREE", "1")
    monkeypatch.setenv("COVERTREEX_PRECISION", "float64")
    cx_config.reset_runtime_config_cache()
    return cx_config.runtime_context()


def _make_dataset(dimension: int, seed: int, *, tree_points: int, queries: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = default_rng(seed)
    points = rng.normal(size=(tree_points, dimension)).astype(np.float64)
    query_ids = rng.integers(0, tree_points, size=queries, endpoint=False, dtype=np.int64).reshape(-1, 1)
    return points, query_ids


def _build_parity_trees(
    dimension: int,
    seed: int,
    *,
    tree_points: int,
    queries: int,
    batch_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    points, query_ids = _make_dataset(dimension, seed, tree_points=tree_points, queries=queries)
    backend = build_residual_backend(
        points,
        seed=seed,
        inducing_count=min(128, tree_points),
        variance=1.0,
        lengthscale=1.0,
        chunk_size=batch_size,
    )

    py_ctx = _parity_context(monkeypatch, enable_rust=False)
    python_tree = build_tree(
        points,
        engine="python-numba",
        context=py_ctx,
        batch_size=batch_size,
        seed=seed,
        residual_backend=backend,
        residual_params={"chunk_size": batch_size},
    )

    rust_ctx = _parity_context(monkeypatch, enable_rust=True)
    rust_tree = build_tree(
        points,
        engine="rust-hybrid",
        context=rust_ctx,
        batch_size=batch_size,
        seed=seed,
        residual_backend=backend,
        residual_params={"chunk_size": batch_size},
    )
    return python_tree, rust_tree, query_ids, py_ctx, rust_ctx


@pytest.mark.parametrize("dimension,seed", [(3, 0), (5, 11)])
@pytest.mark.parametrize("k_values", [(1, 5, 10)])
def test_residual_parity_matches_python(monkeypatch: pytest.MonkeyPatch, dimension: int, seed: int, k_values: Tuple[int, ...]):
    pytest.importorskip("numba")

    python_tree, rust_tree, query_ids, py_ctx, rust_ctx = _build_parity_trees(
        dimension=dimension,
        seed=seed,
        tree_points=1024,
        queries=64,
        batch_size=128,
        monkeypatch=monkeypatch,
    )

    for k in k_values:
        py_idx, py_dist = knn(
            python_tree.handle,
            query_ids,
            k=k,
            return_distances=True,
            context=py_ctx,
        )
        rust_idx, rust_dist = rust_tree.knn(
            query_ids,
            k=k,
            return_distances=True,
            context=rust_ctx,
        )

        py_idx = np.asarray(py_idx)
        py_dist = np.asarray(py_dist)
        rust_idx = np.asarray(rust_idx)
        rust_dist = np.asarray(rust_dist)

        np.testing.assert_array_equal(
            py_idx,
            rust_idx,
            err_msg=f"Neighbor indices diverged for k={k}, dim={dimension}, seed={seed}",
        )
        np.testing.assert_allclose(
            py_dist,
            rust_dist,
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"Distances diverged for k={k}, dim={dimension}, seed={seed}",
        )

    cx_config.reset_runtime_context()


_TELEMETRY_FIELDS = {
    "frontier_levels",
    "frontier_expanded",
    "yields",
    "caps_applied",
    "prunes_lower_bound",
    "prunes_lower_bound_chunks",
    "prunes_cap",
    "masked_dedup",
    "distance_evals",
    "budget_escalations",
    "budget_early_terminate",
    "level_cache_hits",
    "level_cache_misses",
    "block_sizes",
}


def test_residual_parity_telemetry_schema(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("numba")

    python_tree, rust_tree, query_ids, _py_ctx, rust_ctx = _build_parity_trees(
        dimension=3,
        seed=42,
        tree_points=256,
        queries=16,
        batch_size=64,
        monkeypatch=monkeypatch,
    )

    handle = rust_tree.handle
    # Ensure prior telemetry is clear
    if hasattr(handle.tree, "clear_last_query_telemetry"):
        handle.tree.clear_last_query_telemetry()

    _ = handle.tree.knn_query_residual(
        query_ids.reshape(-1),
        handle.node_to_dataset,
        handle.v_matrix,
        handle.p_diag,
        handle.coords,
        float(handle.rbf_variance),
        np.asarray(handle.rbf_lengthscale, dtype=handle.dtype),
        10,
    )
    telemetry = handle.tree.last_query_telemetry()
    assert telemetry is not None, "Rust parity telemetry missing"

    assert set(telemetry.keys()) == _TELEMETRY_FIELDS
    assert len(telemetry["frontier_levels"]) == len(telemetry["frontier_expanded"])
    assert telemetry["block_sizes"], "block_sizes should not be empty"
    assert all(isinstance(v, (int, float)) for v in telemetry["frontier_levels"])
    assert all(isinstance(v, (int, float)) for v in telemetry["block_sizes"])
    assert isinstance(telemetry["distance_evals"], int)
    assert isinstance(telemetry["prunes_lower_bound"], int)

    cx_config.reset_runtime_context()
