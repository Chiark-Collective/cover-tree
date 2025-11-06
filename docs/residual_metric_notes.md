# Residual-Correlation Metric Integration (2025-11)

This note summarizes the current host-side implementation for the Vecchia residual-correlation metric inside `covertreex`.

## Host Caches & Configuration

We introduced `ResidualCorrHostData` in `covertreex/metrics/residual.py`. It packages the host-resident artefacts supplied by the VIF pipeline:

- `v_matrix` — the low-rank factors \( V = L_{mm}^{-1} K(X, U) \) for all training points.
- `p_diag` — per-point residual diagonals \( p_i = \max(K_{x_ix_i} - \|V_i\|^2, 10^{-9}) \).
- `kernel_diag` — raw kernel diagonals \( K_{x_ix_i} \) (provides a fallback for bound computations).
- `kernel_provider(rows, cols)` — a callable that returns the raw kernel slice \( K(X_{rows}, X_{cols}) \) over integer dataset indices.
- `point_decoder` — optional decoder that maps tree payloads to dataset indices (defaults to treating them as integer ids).
- `chunk_size` — preferred batch size for host streaming (defaults to 512).

`configure_residual_correlation(...)` installs the residual metric hooks. We intentionally keep Euclidean metrics untouched: the residual path is only active when `COVERTREEX_METRIC=residual_correlation` and custom caches are registered.

## Traversal Path

### Early-Exit Parent Search

- `_residual_find_parents` (in `covertreex/algo/traverse.py`) streams the tree indices in `chunk_size` tiles.
- For each chunk, we request the raw kernel block and feed it to the chunk kernel (`compute_distance_chunk` from `metrics/_residual_numba.py`).
- The kernel accumulates `V_i · V_j` incrementally, uses cached \( \|V_i\|^2 \) and \( p_i \) to bound the residual correlation, and aborts if the best possible distance still exceeds the caller’s current best. This replicates the residual bound from `_ResidualCorrBackend.lower_bound_train`.
- We track the minimum distance per query, yielding the same parent as the dense path.

### Streaming Scope Assembly

- `_collect_residual_scopes_streaming` reuses the chunk kernel to gather per-query conflict scopes.
- For each parent, we stream candidate tree nodes, apply the residual bound (lower bound via kernel diagonal) and exact distance checks only for survivors, and accumulate dataset indices into the scope. Parent chains (`Next`) are appended afterwards, ensuring deterministic ordering (descending level then index).
- Radii are derived the same way as the Euclidean path: \( \max(2^{\ell_i+1}, S_i) \).

## Conflict Graph Pipeline

### Pairwise Matrix

- Inside `build_conflict_graph`, when residual mode is active we decode dataset ids for the batch and materialise the full \( n \times n \) matrix by streaming kernel tiles through `compute_residual_distances_from_kernel`. This preserves compatibility with both dense and segmented builders.

### Adjacency Filter

- The dense adjacency builder (`_build_dense_adjacency`) now accepts an optional `residual_pairwise` matrix. When provided, the Numba helper receives the residual distances directly.
- The post-build radius filter no longer touches Euclidean norms. Instead, it groups outgoing edges by source, streams the kernel rows, and calls `compute_residual_distances_from_kernel` to prune edges one chunk at a time.
- The segmented builder piggybacks on the same residual matrix (used when `COVERTREEX_CONFLICT_GRAPH_IMPL=segmented`).

## Chunk Kernel (Numba)

`covertreex/metrics/_residual_numba.py` houses the Numba implementation. Given:

- query factor `v_query`, chunk factors `v_chunk`
- cached radii/diagonals `p_i`, `p_chunk`
- raw kernel entries

It emits both distances and a mask indicating which entries fall below a caller-specified radius. We expose `compute_residual_distances_with_radius` in `metrics/residual.py` so traversal and conflict graph code paths can reuse the same accelerated helper with CPU fallback.

## Tests

- `tests/test_metrics.py` exercises the chunk kernel (distance + radius masks) and validates that residual distances computed via kernel reuse match the dense path.
- `tests/test_traverse.py` now includes a residual sparse traversal regression (dense vs. streamed scopes) to ensure parents/levels/scopes remain consistent.
- `tests/test_conflict_graph.py` gained a residual parity check to confirm dense Euclidean and residual-aware models produce identical CSR structures.

## Operational Notes

- Residual mode requires `backend.name == "numpy"` (NumPy backend) until the GPU/JAX kernels are ported.
- The decoder must map tree payloads to dataset indices; if trees store transformed buffers, supply a custom `point_decoder` when creating `ResidualCorrHostData`.
- The chunk kernel honours `chunk_size`; tune it to balance host-side streaming vs. cache reuse.
- Setting `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1` now engages residual streaming automatically when the metric is residual-correlation; otherwise, traversal falls back to the dense mask path.

