import numpy as np
import pytest

try:
    import covertreex_backend
except ImportError:  # pragma: no cover - optional backend
    covertreex_backend = None


@pytest.mark.skipif(covertreex_backend is None, reason="Rust backend not built")
@pytest.mark.parametrize(
    "tree_dtype,payload_dtype",
    [
        (np.float32, np.float64),  # tree expects f32 but receives f64 payloads
        (np.float64, np.float32),  # tree expects f64 but receives f32 payloads
    ],
)
def test_residual_insert_and_query_accepts_mixed_dtypes(tree_dtype, payload_dtype):
    rng = np.random.default_rng(0)

    n = 8
    dim = 2
    rank = 3
    k = 2

    # Start with an empty tree in the target dtype.
    dummy_points = np.empty((0, 1), dtype=tree_dtype)
    dummy_parents = np.empty(0, dtype=np.int64)
    dummy_children = np.empty(0, dtype=np.int64)
    dummy_next = np.empty(0, dtype=np.int64)
    dummy_levels = np.empty(0, dtype=np.int32)
    tree = covertreex_backend.CoverTreeWrapper(
        dummy_points, dummy_parents, dummy_children, dummy_next, dummy_levels, -10, 10
    )

    # Payloads deliberately use the opposite dtype to exercise coercion.
    indices = np.arange(n, dtype=payload_dtype).reshape(-1, 1)
    v_matrix = rng.normal(size=(n, rank)).astype(payload_dtype)
    p_diag = np.full(n, 0.5, dtype=payload_dtype)
    coords = rng.normal(size=(n, dim)).astype(payload_dtype)
    rbf_var = 1.0
    rbf_ls = np.ones(dim, dtype=payload_dtype)

    tree.insert_residual(indices, v_matrix, p_diag, coords, rbf_var, rbf_ls)
    assert tree.point_count() == n

    node_to_dataset = np.arange(n, dtype=np.int64).tolist()
    queries = np.asarray([0, 1], dtype=np.int64)
    idx, dists = tree.knn_query_residual(
        queries,
        node_to_dataset,
        v_matrix,
        p_diag,
        coords,
        rbf_var,
        rbf_ls,
        k,
    )

    assert idx.shape == (queries.shape[0], k)
    assert dists.shape == (queries.shape[0], k)
    assert idx.dtype == np.int64
    expected_float = np.float32 if tree_dtype == np.float32 else np.float64
    assert dists.dtype == expected_float
