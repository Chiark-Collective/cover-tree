# Residual Parity Status — 2025-11-25

## Situation
- Cross‑engine parity is still failing: indices match, but distances diverge for a subset of queries when running the parity test.
- Failure reproduced with `pytest tests/test_residual_parity.py::test_residual_numba_matches_rust_parity --runxfail` (expected xfail; actual max |Δdist| ≈ 0.215).
- Observed differences: Rust returns `0.0` for some self‑distances where Python/Numba reports ~0.215–0.325. Max diff occurs on queries hitting those self entries.

## What was changed (today)
- `src/metric.rs`
  - Distance now uses `sqrt(1 - |rho|)` everywhere (previously a mix of squared and unsquared); `max_distance_hint` tightened to 1.0.
  - Added parity‑mode float32 accumulation path to reduce f64 drift: parity mode (or `COVERTREEX_RESIDUAL_FORCE_F32_MATH=1`) builds f32 copies of coords/V/p_diag and routes dot products through a small-lane accumulator.
  - Lifetime bounds updated to satisfy Rust compile after parity hooks.
- Rebuilt the extension with `maturin develop --release`.

## Current lead / hypothesis
- Python’s residual distances are produced via `distance_block_no_gate` (Numba) using:
  - `kernel_block` (float32), `dot_block` (float32), `p_diag` (float32), `eps=1e-9`
  - `rho = (kernel - dot) / sqrt(p_i * p_j)`, `dist = sqrt(max(0, 1 - |rho|))`
- Rust parity path still computes distance_idx directly from the residual metric using its own p_diag / v_norm handling; self distances collapse to zero because `(k_val - dot) ≈ 0` with tiny denominators, unlike the Python block path that yields ~0.215 due to the epsilon floor and how v_norm_sq is precomputed.
- Actionable next step: mirror the Python `distance_block_no_gate` formula inside Rust parity mode (distance_idx + batch helpers) using the same epsilon (1e-9) and the precomputed v_norm_sq/p_diag values, or call into a shared block routine. This should lift Rust self-distances to the Python values and close the 0.215 gap.

## Repro / Commands
```bash
# Build the Rust extension (release)
maturin develop --release

# Run the parity test (xfail expected; use --runxfail to see the diff)
pytest tests/test_residual_parity.py::test_residual_numba_matches_rust_parity -q --runxfail

# Quick manual check of distances
python - <<'PY'
import numpy as np, os
from covertreex.api import PCCT, Runtime
from covertreex.metrics import build_residual_backend, configure_residual_correlation

points = np.arange(64, dtype=np.float64).reshape(-1, 1)
b = build_residual_backend(points, seed=99, inducing_count=128, variance=1.0, lengthscale=1.0, chunk_size=64)
configure_residual_correlation(b)
os.environ["COVERTREEX_RESIDUAL_PARITY"] = "1"
os.environ["COVERTREEX_ENABLE_RUST"] = "1"

rt_base = Runtime(backend="numpy", precision="float64", metric="residual_correlation",
                  enable_sparse_traversal=True, residual_use_static_euclidean_tree=True,
                  batch_order="natural")
rt_numba = rt_base.with_updates(enable_numba=True, enable_rust=False, engine="python-numba")
rt_rust  = rt_base.with_updates(enable_numba=False, enable_rust=True,  engine="rust-hilbert")

tree_n = PCCT(rt_numba).fit(points, mis_seed=7)
tree_r = PCCT(rt_rust).fit(points, mis_seed=7)
queries = np.arange(16, dtype=np.int64).reshape(-1, 1)
_, dn = PCCT(rt_numba, tree_n).knn(queries, k=1, return_distances=True)
_, dr = PCCT(rt_rust,  tree_r).knn(queries, k=1, return_distances=True)
print("max |diff|", np.max(np.abs(dn - dr)))
print("numba first10", dn.flatten()[:10])
print("rust  first10", dr.flatten()[:10])
PY
```

## Relevant source touchpoints
- `src/metric.rs`: residual distance functions (`distance_idx`, `distances_sq_batch_idx_into_with_kth`, parity f32 paths); `max_distance_hint` definition.
- Python gold path for distances: `covertreex/metrics/_residual_numba.py::distance_block_no_gate` and `covertreex/metrics/residual/core.py::compute_residual_distances`.
- Parity test harness: `tests/test_residual_parity.py::test_residual_numba_matches_rust_parity`.

## Next steps (recommended)
1) In Rust parity mode, reimplement residual distance to exactly mirror Python’s block formula (use the same eps=1e-9 and p_diag/v_matrix cache; ensure float32 path). Verify self-distances move to ~0.215/0.325 where Python does.
2) Re-run `pytest ... --runxfail`; if green, drop xfail and update docs/bench logs.
3) Only after parity is green, rerun `benchmarks/run_residual_gold_standard.sh` to confirm perf hasn’t regressed; document commands/results per AGENTS guidelines.
