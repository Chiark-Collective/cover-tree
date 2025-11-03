# Parallel Compressed Cover Tree Implementation Plan

## Context & Goal

This library is being designed as the neighbour-selection backbone for the Vecchia variational Gaussian process codebase (JAX, GPU-centric). At inference/training time we need:

- **Fast, persistent neighbour graphs** that can be swapped atomically during LBFGS iterations (async rebuild worklets).
- **GPU-friendly kernels** that play nicely with JAX (no host ↔ device ping-pong), yet keep CPU fallbacks for preprocessing.
- **Compressed cover-tree semantics** so we reuse state across epochs and avoid memory blow-up.
- **Parallel batch updates** to hide rebuild latency during Vecchia state refresh and to accelerate offline preparation.

Deliver a reusable “parallel compressed cover tree” (PCCT) library that combines:

- **Compressed cover tree** representation from Elkin–Kurlin (one node per point, distinctive descendant caches, O(1) `Children`/`Next` lookups).
- **Parallel batch updates** from Gu–Napier–Sun–Wang (prefix-doubling batches, conflict-graph MIS for insertion/deletion, path-copying persistence).
- **Unified backend** built on `jax.numpy`, with optional Numba acceleration for tight loops.

## Modules & Responsibilities

| Module | Purpose |
| ------ | ------- |
| `covertreex/config.py` | Centralised configuration surface: parse env vars, choose backend/device, set precision flags, expose `RuntimeConfig` used by tree/algo modules. |
| `covertreex/core/tree.py` | Immutable `PCCTree` dataclass holding points, per-point top levels, parent pointers, compressed child tables, per-level cover sets, cached `S_i` and `Next`, log counters, backend reference. |
| `covertreex/core/persistence.py` | Copy-on-write helpers to clone level slices and child segments when applying updates (path-copying). |
| `covertreex/core/metrics.py` | Backend registry with `jax.numpy` default; Numba kernels exposed behind feature flag; metric abstraction (pairwise & pointwise distances). |
| `covertreex/algo/traverse.py` | Batched traversal (Alg. 4 lines 5–9) returning `(parent, level)` decisions and conflict scopes `Π_parent`. |
| `covertreex/algo/conflict_graph.py` | Builds CSR adjacency for `L_k` using restricted annuli `(Π_{P_i} ∩ L_k)`; optional Euclidean grid/binning. |
| `covertreex/algo/mis.py` | Luby MIS on CSR (pure `jax.numpy` version, plus Numba acceleration path). |
| `covertreex/algo/batch_insert.py` | Prefix-doubling orchestration, per-level MIS selection, redistribution (ceil-log₂ distance), tree updates via persistence helpers. |
| `covertreex/algo/batch_delete.py` | Bottom-up uncovered set handling, MIS-driven promotions, new-root handling (Alg. 5). |
| `covertreex/queries/knn.py` | Corrected CCT k-NN (Alg. F.2) using cached `S_i` / `Next` tables; convenience 1-NN. |
| `tests/` | Pytest suite for invariants, traversal scopes, conflict graphs, MIS, batch ops, k-NN, persistence, backend parity. |
| `benchmarks/` | CLI scripts for insert/delete throughput, k-NN latency, distance-count scaling. |

## Backend Strategy

- **Primary API:** `jax.numpy` (`jnp`) for all core math; ensures eventual JIT/HLO portability.
- **Acceleration:** optional Numba wrappers for hotspots (distance kernels, MIS) with seamless fallback.
- **Future:** pure NumPy shim if needed for environments without JAX.

## Configuration & Environment

- **Env surface (`COVERTREEX_*`):** `BACKEND` (`jax`, `numpy`, future), `DEVICE` (e.g. `gpu:0,cpu:0`), `PRECISION` (`float64` default, `float32` alt), `ENABLE_NUMBA` (`0/1`), `LOG_LEVEL`.
- **Randomness:** `COVERTREEX_MIS_SEED` seeds Luby MIS for deterministic runs.
- **JAX flags:** gate `jax_enable_x64` via either `COVERTREEX_PRECISION` or inherited `JAX_ENABLE_X64`; respect downstream overrides instead of hard-coding.
- **Device detection:** inspect `jax.devices()` on import, honour `COVERTREEX_DEVICE` filters, fall back to CPU with debug log if GPUs unavailable.
- **Backend registry:** expose `RuntimeConfig` singleton that modules pull from to obtain `TreeBackend` objects (JAX primary, NumPy shim, optional Numba kernels keyed off `ENABLE_NUMBA`).
- **DX expectations:** deterministic configuration (cache results, avoid implicit JAX global state), environment warnings routed through logger (respect `LOG_LEVEL`).

## Implementation Tasks

1. **Configuration bootstrap**
   - Implement `covertreex/config.py` with `RuntimeConfig.from_env()` and unit tests for env parsing and device selection fallbacks.
   - Ensure JAX precision/device flags are applied exactly once and support explicit overrides from downstream libraries.
   - Provide helpers for obtaining `TreeBackend` instances and reporting the active configuration.

2. **Core structure & persistence**
   - Implement `PCCTree`, including cover-node caches, child tables, and `S_i` / `Next`.
   - Add copy-on-write utilities and tests verifying version isolation.

3. **Traversal & scopes**
   - Implement `traverse_collect_scopes` (vectorised over batch points).
   - Normalise ragged scopes with semisort/group-by utilities.
   - Tests: compare to sequential compressed insertion decisions.
   - Status: traversal emits semisorted scopes + CSR buffers (`scope_indptr`, `scope_indices`) with cached-radius floor and `Next` chain expansion. Next: integrate compressed-level pruning heuristics for batch redistribution.

4. **Conflict graph**
   - Build CSR from `Π_{P_i} ∩ L_k` with distance filtering (`<= 2^k`).
   - Add optional grid/binning for Euclidean spaces; fallback to pure scope scan.
   - Tests: adjacency matches brute-force edge detection.
   - Status: CSR builder live with distance-aware pruning, semisort buffers, and integer annulus bins (`annulus_bin_{indptr,indices,ids}`) ready for GPU kernels.

5. **MIS kernels**
   - Implement Luby MIS with `jax.numpy`.
   - Provide Numba-accelerated version sharing the same interface.
   - Tests: correctness (independence + maximality) on random graphs.
   - Status: seeded Luby MIS implemented with JAX primitives; follow-up: expose Numba path + rng batching for async rebuild.

6. **Batch insert (Alg. 4)**
   - Prefix-doubling grouping, per-level MIS loop, persistence-backed updates.
   - Redistribution: update `(P_j, ℓ_j)` via ceil-log₂ distance to new anchors.
   - Tests: invariants, distance counts, parity with sequential compressed build.

7. **Batch delete (Alg. 5)**
   - Implement uncovered-set processing, MIS promotion, new-root creation.
   - Tests similar to batch insert; verify node counts and invariants.

8. **k-NN queries**
   - Implement corrected CCT k-NN loop with cached `S_i` and `Next`.
   - Tests: compare against brute-force distances; handle ties deterministically.

9. **Diagnostics & benchmarks**
- Add logging hooks (conflict edges, MIS iterations, cache hit rates).
- Create benchmark scripts for insert/delete throughput and query latency.

## Testing & Validation

- **Invariant checks:** nesting, covering, separation, distinctive descendant consistency.
- **Cross-checks:** sequential compressed tree comparisons; brute-force k-NN.
- **Property tests:** persistence (old versions intact), MIS separation per level.
- **Complexity sanity:** track distance operations versus `O(n log n)` trend.
- **Backend parity:** ensure NumPy+jax vs. Numba give identical results.
- **Pipeline smoke tests:** traversal → conflict graph → MIS placeholders stitched together; to be upgraded to real invariants once kernels land.
  - Status: pipeline harness active; will evolve alongside semisort scopes and final MIS kernels.

## Milestones

1. `PCCTree` skeleton with traversal + invariants.
2. Batch insert end-to-end parity with sequential builder.
3. Batch delete parity and persistence validation.
4. k-NN queries with correctness suite.
5. Backend acceleration (Numba) + benchmarks.

Each milestone must land with tests and documentation referencing the relevant algorithms/lemmas from both source papers.

## Dependencies & Performance Considerations

- **Core dependencies**
  - `jax[cuda13] >= 0.5.3`: primary array backend; aligns with downstream `survi` environment. CUDA 13 wheel required for GPU support.
  - `jaxlib` matching build (auto-resolved via `jax` metapackage).
  - `numpy`: host-side manipulation (light usage).
  - `numba >= 0.61.2`: optional CPU acceleration for distance kernels/MIS; respect availability checks.
  - `optax`, `tfp-nightly[jax]`: upstream dependencies already in consumer app (no direct use here but avoid conflicts).
- **Optional adapters**
  - `scikit-learn >= 1.3`: only needed when `tree.as_sklearn()` is invoked.
  - `plotly`, `matplotlib`: for visualisation helpers; keep optional.
- **Environment alignment**
  - Python `~=3.12` (matching `survi`).
  - GPU runtime assumes CUDA ≥13 (as in downstream project); document fallback for CPU-only installs.
  - Ensure `jax.config.update("jax_enable_x64", True)` compatibility since Vecchia code relies on float64.
- **Performance targets**
  - Batch insert/delete: within 1.2× sequential compressed build time for n ≤ 32k, with scaling trend matching O(n log n).
  - Async rebuild: overlap ≥80 % of insert compute with simulated LBFGS workload (Tier C test).
  - GPU path: keep host-device traffic <5 % of total time (measured via `jax.profiler`).
  - Memory footprint: double-buffer overhead ≤ 1.5× active neighbour graph (points × k × 8 bytes).
- **Validation milestones**
  - Tier A/B integration tests before exposing APIs.
  - Tier C async + GPU tests gate release candidate.
  - Tier D (`Vecchia Refresh Loop Mini`, adapter compatibility) required before tagging v0.1.

## Integration Test Ladder

To keep the pipeline tight while hitting meaningful milestones, introduce three tiers of integration checks between unit tests and full end-to-end runs:

### Tier A – Structural Core
1. **Traversal + Cache Sanity**
   - After building the initial PCCTree, run batched `traverse_collect_scopes` and verify `(parent, level)` plus scope contents against the sequential traversal and the cached `S_i/Next` tables.
   - Ensures traversal logic, cover sets, and caches are aligned before MIS is involved.
   - Status: `tests/integration/test_structural_core.py::test_traversal_matches_naive_computation` plus randomized fixtures (`test_randomized_structural_invariants`) cover deterministic and stochastic trees.
2. **Scoped Conflict Graph**
   - For a fixed level `k`, construct `Π_{P_i} ∩ L_k` and resulting CSR edges, then compare against a brute-force “check all pairs” routine.
   - Confirms annulus restriction and edge generation behave before entering Luby MIS.
   - Status: `tests/integration/test_structural_core.py::test_conflict_graph_matches_bruteforce_edges` and randomized checks assert CSR edges and annulus bin metadata.

### Tier B – Parallel Update Mechanics
3. **Level-wise MIS Update**
   - Feed a controlled batch through `batch_insert`, capturing intermediate MIS selections; compare against a reference MIS (deterministic seed) and ensure post-update levels satisfy separation.
   - Verifies orchestration of prefix-doubling, per-level MIS, and redistribution without yet touching Vecchia.
   - Status: `plan_batch_insert` + `batch_insert` skeleton wired into integration test (`tests/integration/test_parallel_update.py`), appending selected nodes while keeping originals untouched; next steps: enforce per-level redistribution and persistence diff checks.
4. **Persistence Path Copy**
   - Execute successive updates, then diff consecutive tree versions to confirm only the intended level slices/child ranges changed and earlier versions remain queryable.
   - Confirms the path-copy strategy required for async rebuild.

### Tier C – Application Hooks
5. **Async Refresh Harness**
   - Spin two threads: main thread repeatedly queries neighbours; worker thread builds `tree.update(...)`. Validate that swap occurs only on `future.done()` and that queries never see partially updated state.
   - Direct precursor to plugging into `maybe_refresh_state`.
6. **GPU Builder Smoke**
   - Run build → `add` → `remove` → `knn` entirely on JAX device arrays; assert no implicit host round-trips and results match CPU baseline.

### Tier D – End-to-End Confidence
7. **Vecchia Refresh Loop Mini**
   - Mock LBFGS epochs with synthetic data: at each epoch, call `prefetch_scopes`, schedule async `update`, and check neighbour indices/dists against legacy output.
   - Validates caching, async scheduling, and Vecchia-specific tolerances.
8. **Adapter Compatibility**
   - Roundtrip through `as_sklearn()` and ANN-style adapters to ensure `add/update/query` sequences stay in sync with the underlying tree.

Each tier unlocks the next milestone; we only move forward once the relevant integration tests are green, ensuring issues are caught closer to their source rather than in full application runs.

## API & Developer Experience Sketch

### Core surface

```python
import jax.numpy as jnp
from covertreex import PCCTree

# hello world: build directly from data, Euclidean metric by default
tree = PCCTree.from_points(jnp.asarray(train_points))

# persistent updates return a fresh tree
tree2 = tree.update(insert=jnp.asarray(new_pts), delete=jnp.asarray(old_ids))

# queries
idx, dist = tree2.knn(jnp.asarray(query_pts), k=16, return_distances=True)

# optional mutable transaction for batched streaming updates
with tree2.writer() as txn:
    txn.insert(batch_pts)
    txn.delete(batch_ids)
tree3 = txn.commit()
```

### Key design points

- **Immutable default:** every mutating call (`update`, `add`, `remove`) returns a fresh `PCCTree`, keeping async rebuilds and concurrent analytics safe. Developers opt into mutation explicitly via `writer()`.
- **Ergonomic aliases:** `.add(points)`, `.remove(ids)`, and `.update(insert=..., delete=...)` all share the same engine; `.extend()` accepts iterables/iterators for streaming ingestion.
- **Backend-agnostic arrays:** accept `jax.Array`, `numpy.ndarray`, `cupy`, or `torch.Tensor`; data is coerced once through a lightweight registry (defaults auto-populated) so existing pipelines drop in without ceremony.
- **Metric hooks:** `"euclidean"`, `"manhattan"`, `"cosine"` ship ready; advanced users can register custom distance pairings via a single helper without touching core APIs.
- **Batch pipeline hooks:** `tree.prefetch_scopes(batch)` exposes the traversal/mutation plan so upstream schedulers (e.g., Ray, Dask) can partition work; `tree.stats()` surfaces conflict-graph size, MIS iterations, cache hits.
- **Interoperability adapters:**
  - `tree.as_sklearn()` → `sklearn.neighbors.KNeighborsMixin` wrapper for scikit-learn pipelines.
  - `tree.as_ann_index()` → `build/add/query` style adapter for ANN libraries.
  - `tree.export(level=k)` → JSON snapshot for tooling that expects explicit nodes.
- **Iterator UX:** `tree.walk(level=None)` yields `(node_id, level, point_index, children)` for debugging/inspection.
- **Configuration profiles:** `PCCTree.build(..., profile="low-latency")` toggles defaults (e.g., disable distance caching, shrink prefix batches) while `"throughput"` enables more aggressive caching and Numba kernels.

### Ergonomic helpers

- `covertreex.pipeline.BuildJob`: orchestrates staged ingest (chunked reading, prefix-doubling scheduling) with progress callbacks.
- `covertreex.inspect.compare(tree_a, tree_b)`: diff of invariants, overlap metrics, enabling regression tests for pipelines migrating from existing cover-tree libs.
- `covertreex.visualize.radial(tree, level=k)`: quick scatter/graph plotting for exploratory analysis (matplotlib-backed).

### DX enhancements to prioritise early

- Friendly error messages (`ValueError` with hints) when invariants would be violated (e.g., custom metric returning asymmetric distances).
- Autocomplete-friendly signatures (`update`, `add`, `remove`, `knn`, `radius`) with type hints and docstrings referencing algorithms.
- Lightweight progress hooks (`callbacks={"progress": fn}`) for batch builds—wired before heavy parallel work begins.
- Quickstart notebook showcasing build → update → query → adapter integration, runnable in Colab with minimal boilerplate.
- `tree.profile()` returning human-readable summary (levels, branching stats, cache sizes) for debugging.

### Documentation UX

- **Playbook-style guides:** “Replace your existing cover tree build with PCCTree”, “Batch updates during streaming ingestion”, “Hooking PCCTree into scikit-learn pipelines”.
- **Snippets-first:** every major method documented with minimal runnable code (works in REPL, Colab, or plain Python).
- **Reference to theory:** docstrings link back to algorithm/lemma identifiers so practitioners can align implementation with the papers when auditing correctness.
