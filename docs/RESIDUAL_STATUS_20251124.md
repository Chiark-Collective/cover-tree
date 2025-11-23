# Residual Status Report: 2025-11-24

## Rust Fast Path Optimization

Implemented dense scope streamer, budget ladder, radius floor, and scope cap loading in the Rust backend (`rust-fast` engine).

### Changes
- **Algo:** Refactored `single_residual_knn_query` in `src/algo.rs` to use a tiled child loop (`stream_tile=64`), budget ladder (32/64/96), and radius clamping.
- **Caps:** Added `load_scope_caps` in `src/lib.rs` to load JSON scope caps from Python module via `COVERTREEX_RESIDUAL_SCOPE_CAP_PATH` env var.
- **Build:** `maturin develop --release`.

### Benchmark Results (Gold Standard)
- **Dataset:** 32,768 points, d=3, k=50, 1024 queries.
- **Config:** Caps enabled, Budget 32/64/96.

| Engine | Build Time | Query Time | Throughput | Note |
| :--- | :--- | :--- | :--- | :--- |
| `rust-fast` | 7.42s | 0.14s | **7,105 q/s** | Baseline Rust residual implementation. |
| `rust-pcct2` | **4.20s** | 0.21s | 4,971 q/s | Uses Hilbert ordering & telemetry build path. |

### Analysis
- **Optimization Success:** Both engines are now orders of magnitude faster than the previous unoptimized Rust state (~37 q/s).
- **Engine Discrepancy:** `rust-pcct2` builds nearly **2x faster** (likely due to Hilbert ordering reducing conflict graph complexity) but queries are **~30% slower** than `rust-fast`.
  - This might be due to the tree structure produced by Hilbert ordering interacting with the budget/pruning heuristics differently than the random ordering used in `rust-fast`.
  - Alternatively, there may be slight overheads in the `CoverTreeWrapper` dispatch or `auto` mode logic (though confirmed to default to tree traversal).
- **Next Steps:**
  - Investigate why Hilbert ordering yields slower queries (shallow tree? unbalanced?).
  - Implement Level Cache (key to closing gap with Numba's ~22k q/s).
