from __future__ import annotations

import numpy as np
import pytest

from covertreex import config as cx_config, reset_residual_metric
from covertreex.api import PCCT, Runtime
from covertreex.metrics import build_residual_backend, configure_residual_correlation
from covertreex.metrics.residual import set_residual_backend


@pytest.fixture(autouse=True)
def reset_runtime() -> None:
    cx_config.reset_runtime_context()
    yield
    cx_config.reset_runtime_context()
    reset_residual_metric()
    set_residual_backend(None)


def test_residual_numba_matches_rust_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a small deterministic dataset
    points = np.arange(64, dtype=np.float64).reshape(-1, 1)

    # Configure shared residual backend
    backend_state = build_residual_backend(
        points,
        seed=99,
        inducing_count=128,
        variance=1.0,
        lengthscale=1.0,
        chunk_size=64,
    )
    configure_residual_correlation(backend_state)

    # Parity mode for Rust to disable perf heuristics
    monkeypatch.setenv("COVERTREEX_RESIDUAL_PARITY", "1")
    monkeypatch.setenv("COVERTREEX_ENABLE_RUST", "1")

    base_runtime = Runtime(
        backend="numpy",
        precision="float64",
        metric="residual_correlation",
        enable_sparse_traversal=True,
        residual_use_static_euclidean_tree=True,
        batch_order="natural",
    )
    rt_numba = base_runtime.with_updates(enable_numba=True, enable_rust=False, engine="python-numba")
    rt_rust = base_runtime.with_updates(enable_numba=False, enable_rust=True, engine="rust-hilbert")

    # Build trees per engine to avoid layout differences in wrappers
    tree_numba = PCCT(rt_numba).fit(points, mis_seed=7)
    tree_rust = PCCT(rt_rust).fit(points, mis_seed=7)

    # Queries are dataset indices to avoid decode ambiguity
    queries = np.arange(16, dtype=np.int64).reshape(-1, 1)

    idx_numba, dist_numba = PCCT(rt_numba, tree_numba).knn(queries, k=1, return_distances=True)
    idx_rust, dist_rust = PCCT(rt_rust, tree_rust).knn(queries, k=1, return_distances=True)

    np.testing.assert_array_equal(idx_numba, idx_rust)
    np.testing.assert_allclose(dist_numba, dist_rust, atol=1e-6, rtol=0.0)
