# Residual Status Report: 2025-11-24 (Update 2)

## Scope Caps Integrated into `rust-pcct2`

Corrected the `rust-pcct2` query path (`knn_query_residual_block`) to also load and apply scope caps when falling back to the tree algorithm (`COVERTREEX_RUST_PCCT2_SGEMM=tree` or `auto` without survivors).

### Changes
- **Algo:** Generic `scope_caps` parameter (`Option<&HashMap<i32, T>>`) in `algo.rs` to support both f32 and f64 paths.
- **Lib:** Updated `knn_query_residual_block` (F32 path) to:
  1.  Load scope caps via `load_scope_caps(py)`.
  2.  Pass `scope_caps.as_ref()` to `batch_residual_knn_query`.
- **Consistency:** Both `rust-fast` and `rust-pcct2` now respect the `COVERTREEX_RESIDUAL_SCOPE_CAP_PATH` environment variable and apply the dense streamer/budget ladder optimizations when using the tree traversal path.

### Benchmark Note
The previous `rust-fast` benchmark result (7,105 q/s) remains valid as it uses the `knn_query_residual` entry point, which was already updated. This update ensures `rust-pcct2` users also benefit from the optimizations.