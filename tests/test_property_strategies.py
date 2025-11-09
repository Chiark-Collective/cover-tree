from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

from covertreex.algo.mis import batch_mis_seeds
from covertreex.algo.order.strategy import compute_batch_order

_float_elements = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
_shape_strategy = st.tuples(
    st.integers(min_value=0, max_value=32),
    st.integers(min_value=1, max_value=8),
)
_points_strategy = _shape_strategy.flatmap(
    lambda shape: hnp.arrays(dtype=np.float64, shape=shape, elements=_float_elements)
)


@settings(max_examples=50)
@given(points=_points_strategy, strategy=st.sampled_from(["natural", "random", "hilbert"]))
def test_batch_order_returns_valid_permutation(points: np.ndarray, strategy: str) -> None:
    result = compute_batch_order(points, strategy=strategy, seed=7)
    n_points = points.shape[0]
    if n_points <= 1 or strategy == "natural":
        assert result.permutation is None
    else:
        assert result.permutation is not None
        perm = result.permutation
        assert perm.shape == (n_points,)
        assert perm.dtype.kind in {"i", "u"}
        assert np.array_equal(np.sort(perm), np.arange(n_points))


@settings(max_examples=50)
@given(count=st.integers(min_value=0, max_value=64), seed=st.integers(min_value=0, max_value=1_000_000))
def test_batch_mis_seeds_are_deterministic(count: int, seed: int) -> None:
    first = batch_mis_seeds(count, seed=seed)
    second = batch_mis_seeds(count, seed=seed)
    assert first == second
    assert len(first) == count
    if count:
        assert len(set(first)) == count
