# Residual Audit Implementation Plan

_Last updated: 2025-11-11_

This document tracks the concrete engineering work required to execute the November 2025 residual audit directions. Each section cites the current code, describes the implementation gaps, and outlines the changes we intend to land.

## Phase 0 — Hotspot Telemetry Injection

**Current code.** We log high-level `traversal_ms`, but we never break down how many point pairs flowed through the whitened-V multiply vs. the raw kernel provider. Without that insight it’s easy to optimise the wrong surface area.

**Plan.**
- Instrument `_collect_residual_scopes_streaming_*` (and the dense fallback) with four new counters: `whitened_block_pairs`, `whitened_block_ms`, `kernel_provider_pairs`, `kernel_provider_ms`.
- Emit the counters in `TraversalTimings` and add them to `tools/export_benchmark_diagnostics.py`.
- Define a single authoritative “build wall time” metric (e.g., measured around the full insertion call) so we stop reporting conflicting figures (257 s vs 968 s); log it next to the new counters for every run.
- Acceptance gate: a single Hilbert 32 k run should show these counters covering ≥80 % of total traversal time. Capture the numbers in the plan so later phases can compare.

**Key files:** [covertreex/algo/traverse/strategies.py](../covertreex/algo/traverse/strategies.py), [covertreex/algo/traverse/base.py](../covertreex/algo/traverse/base.py), [covertreex/algo/batch/insert.py](../covertreex/algo/batch/insert.py), [covertreex/telemetry/logs.py](../covertreex/telemetry/logs.py), [tools/export_benchmark_diagnostics.py](../tools/export_benchmark_diagnostics.py).

**Progress:** Not started.

## Phase 1 — Whitened-V Block GEMM (Primary Lever)

**Current code.** `_collect_residual_scopes_streaming_*` still calls `_distance_chunk` for every `(query, chunk)` pair, which loops over rank dimensions in Python/Numba and recomputes `V_i · V_j` column by column. There is no bulk multiply path; most CPU time is spent redoing dot products that could be a single SGEMM.

**Plan.**
- Introduce a dedicated helper (e.g., `compute_whitened_block(host_backend, query_ids, chunk_ids, *, workspace)`) that:
  1. Loads `gate_v32` (or a new `v_whitened_f32`) and corresponding norm squares (float32, contiguous).
  2. Batches queries into `QW = VW[query_ids]`, candidates into `TW = VW[chunk_ids]`.
  3. Executes `G = QW @ TW.T` via SGEMM, writing into a reusable workspace buffer.
  4. Forms `dist2 = norm_q[:,None] + norm_t[None,:] - 2 * G` and feeds that into the residual bound / mask builder.
- Define a small workspace struct (e.g., `ResidualWorkspace(g, dist2, mask)`), allocate it once per traversal thread (size tuned to `scope_chunk_target`), and reuse it per chunk to avoid allocator churn.
- Apply the gate / radius checks directly on the SGEMM output, pushing survivors straight into the CSR builder before we ever call the raw kernel provider.
- Acceptance criteria:
  * Median dominated-batch `traversal_ms` drops from ~0.7–1.4 s (dense) / 8.6 s (sparse) to ≤150 ms.
  * 32 k residual builds fall from “minutes” to “tens of seconds” even before any other optimisation lands.

**Key files:** [covertreex/metrics/residual/core.py](../covertreex/metrics/residual/core.py), [covertreex/metrics/_residual_numba.py](../covertreex/metrics/_residual_numba.py), [covertreex/algo/traverse/strategies.py](../covertreex/algo/traverse/strategies.py), [covertreex/algo/_residual_scope_numba.py](../covertreex/algo/_residual_scope_numba.py).

**Progress:** Not started.

## Phase 2 — GEMM-Based Raw Kernel Provider (Exact Fallback)

**Current code.** `covertreex/metrics/residual/host_backend.py:14-112` still defines `_rbf_kernel` via full broadcasting (`x[:, None, :] - y[None, :, :]`) and wires it directly into `kernel_provider`. Every traversal chunk therefore allocates a (rows × cols × dim) float64 tensor under the GIL.

**Plan.**
- Introduce a helper (e.g., `make_rbf_provider(points_f32, gamma, *, workbuf=None)`) that keeps the dataset in `float32`, precomputes `row_norms = (points_f32 ** 2).sum(1)`, and computes tiles via `A @ B.T` (`SGEMM`).
- Extend `ResidualCorrHostData` (or a sibling struct) with optional `kernel_points_f32`, `kernel_row_norms`, and `kernel_workbuf` so traversal can reuse the buffers without reallocation.
- Update `build_residual_backend(...)` to normalise inputs once (`np.asarray(points, dtype=np.float32, order="C")`), stash the contiguous copy plus norms, and expose the GEMM provider. The provider should accept `out` buffers to avoid repeated allocations inside `_collect_residual_scopes_streaming_*`.
- Ensure `kernel_provider` always returns `float32` tiles; downstream callers (`compute_residual_distances_from_kernel`) can promote to `float64` only when necessary.
- Add unit tests in `tests/test_metrics.py` to confirm the provider matches the existing `_rbf_kernel` numerically (within 1e-6 relative error) and stays in float32.
- Acceptance criteria:
  * `kernel_provider_ms` collapses to a handful of large calls per batch (10–50× faster).
  * Survivors only trigger the provider, confirming Phase 1 masks eliminated most work.

**Key files:** [covertreex/metrics/residual/host_backend.py](../covertreex/metrics/residual/host_backend.py), [covertreex/metrics/residual/core.py](../covertreex/metrics/residual/core.py), [covertreex/metrics/_residual_numba.py](../covertreex/metrics/_residual_numba.py), [tests/test_metrics.py](../tests/test_metrics.py).

**Progress:** Not started.

## Phase 3 — Enforce Residual Pairwise Reuse & Strategy Selection

**Current code.** `build_conflict_graph` (`covertreex/algo/conflict/runner.py:55-205`) already raises `ResidualPairwiseCacheError` when a residual batch arrives without cached pairwise distances, and `_collect_residual(...)` (`covertreex/algo/traverse/strategies.py:1204-1399`) constructs a `ResidualTraversalCache` every time it runs. The weak point is the selection predicate (`strategies.py:1405-1426`): residual traversal is only chosen when both `runtime.enable_sparse_traversal` and `runtime.enable_numba` are true. If a user invokes the residual metric with those toggles off (“dense” regressions often do), the scheduler falls back to the Euclidean strategies, no cache is produced, and conflict graph either recomputes kernels or aborts.

**Plan.**
- Relax the traversal predicate so `_ResidualTraversal` is selected for every NumPy residual run, independent of the sparse toggle; emulate “dense vs sparse” solely via `runtime.scope_chunk_target` and related caps.
- Make the Euclidean strategies fail fast when `runtime.metric == "residual_correlation"` so misconfigurations are immediately visible in tests.
- Teach `cli/queries` to auto-enable Numba + residual traversal (or print a clear error) whenever `--metric residual` is requested, ensuring telemetry consistently records `conflict_pairwise_reused=1`.
- Keep the existing `ResidualPairwiseCacheError` but add regression tests in `tests/test_conflict_graph.py` that confirm cache-less traversals raise, and that both scope-cap=0 (“dense”) and scope-cap>0 (“sparse”) modes reuse the cached block.
- Plumb the reuse flag through `covertreex/algo/batch/insert.py` and `covertreex/telemetry/logs.py` so diagnostics can assert every residual batch set `pairwise_reused=1`; expand `tools/export_benchmark_diagnostics.py` to treat any zero as a hard failure.
- Acceptance criteria:
  * Conflict graph timings stay ≤50 ms per dominated batch in both dense/sparse recipes.
  * Every residual batch logs `conflict_pairwise_reused=1`; CI fails otherwise.

**Key files:** [covertreex/algo/traverse/strategies.py](../covertreex/algo/traverse/strategies.py), [covertreex/algo/conflict/runner.py](../covertreex/algo/conflict/runner.py), [covertreex/algo/conflict/strategies.py](../covertreex/algo/conflict/strategies.py), [covertreex/algo/batch/insert.py](../covertreex/algo/batch/insert.py), [covertreex/telemetry/logs.py](../covertreex/telemetry/logs.py), [tools/export_benchmark_diagnostics.py](../tools/export_benchmark_diagnostics.py), [tests/test_conflict_graph.py](../tests/test_conflict_graph.py), [cli/queries.py](../cli/queries.py).

**Progress:** Not started.

## Phase 4 — Canonical Float32 Staging & Accuracy Guardrails

**Current code.** `ResidualCorrHostData` stores `v_matrix`, `p_diag`, and `kernel_diag` as float64 (`covertreex/metrics/residual/core.py:52-123`), and `configure_residual_correlation` re-computes `v_norm_sq` by re-casting the matrix back to float64. The host backend builder (`host_backend.py:86-109`) also emits float64 everywhere, so traversal drags twice the bandwidth it needs.

**Plan.**
- Switch `ResidualCorrHostData` to accept float32 matrices by default. Keep float64 buffers only where numerically necessary (e.g., when running Cholesky inside `build_residual_backend` or for auditing radii) and downcast once the stable factors are computed.
- Materialise and cache both `v_matrix_f32` and `v_matrix_f64` if Gate‑1 or audits still require float64 whitened vectors. Expose convenience properties (e.g., `.v_matrix_view(dtype)`) so Numba kernels (`metrics/_residual_numba.py`) can consume float32 without extra copies.
- Update `_compute_residual_distances_from_kernel` and `distance_block_no_gate` invocations to accept float32 inputs and cast internally only for intermediate numerics.
- Guard all `np.asarray(..., dtype=np.float64)` calls in `configure_residual_correlation` and `compute_*` helpers; only promote when the downstream API truly requires double precision.
- Add regression tests that load a float32 backend, run traversal, and confirm no implicit float64 upcasts occur (inspect `ndarray.dtype` on Cached arrays inside `ResidualTraversalCache`).
- Document a clear accuracy policy: e.g., “factorisation + audit remain float64; traversal/conflict run in float32 with tolerances ≤1e-4”. Add a test that audit mode never flags false positives after the dtype change.
- Acceptance criteria: memory footprint drops measurably (≈2× on caches), and audit passes remain clean.

**Key files:** [covertreex/metrics/residual/core.py](../covertreex/metrics/residual/core.py), [covertreex/metrics/residual/host_backend.py](../covertreex/metrics/residual/host_backend.py), [covertreex/metrics/_residual_numba.py](../covertreex/metrics/_residual_numba.py), [covertreex/algo/conflict/runner.py](../covertreex/algo/conflict/runner.py), [docs/residual_metric_notes.md](residual_metric_notes.md), [tests/test_metrics.py](../tests/test_metrics.py), [tests/test_conflict_graph.py](../tests/test_conflict_graph.py).

**Progress:** Not started.

## Phase 5 — Selection with Deterministic Tie-Breaks

**Current code.** Dense traversal still sorts entire CSR masks using `np.lexsort` (`covertreex/algo/traverse/strategies.py:184-199`). `covertreex/algo/semisort.py` likewise sorts every `(key, value)` pair even though the downstream MIS only needs the first `scope_limit` entries per query.

**Plan.**
- Introduce a utility (e.g., `select_topk_by_level(mask_row, levels, k)`) that performs `np.argpartition` to grab just the top-`scope_limit` candidates per query before a small local sort, and use it both in the Python fallback and the NumPy mask path.
- Within `_collect_residual_scopes_streaming_*`, stop appending entire `tree_positions` ranges when scopes already exceed `scope_limit`; apply `argpartition` on the `observed_radii` or per-level ordering to cap work eagerly.
- Extend the Numba CSR builder (`covertreex/algo/_scope_numba.py`) with a selection-based path when `scope_chunk_target` / `scope_limit` is non-zero; this keeps parity between dense and sparse configurations.
- After any `argpartition`, run a stable sort on the selected slice using `(level, index)` tie breakers so CSR output stays deterministic across runs.
- Update regression tests (e.g., `tests/test_traverse.py::test_residual_scope_limit`) to assert that the new selection path yields the same neighbor set as the full sort when `k` >= actual scope size, and that telemetry counters (`scope_chunk_max_members`) reflect the capped selection.
- Acceptance criteria: `semisort_seconds` approaches zero and MIS regressions stay absent due to deterministic ordering.

**Key files:** [covertreex/algo/traverse/strategies.py](../covertreex/algo/traverse/strategies.py), [covertreex/algo/_scope_numba.py](../covertreex/algo/_scope_numba.py), [covertreex/algo/semisort.py](../covertreex/algo/semisort.py), [tests/test_traverse.py](../tests/test_traverse.py), [docs/CORE_IMPLEMENTATIONS.md](CORE_IMPLEMENTATIONS.md).

**Progress:** Not started.

## Phase 6 — Remove Python Callouts from Hot Numba Loops

**Current code.** The streamer functions (`_collect_residual_scopes_streaming_serial/parallel` in `covertreex/algo/traverse/strategies.py:295-744`) repeatedly call the Python-level `kernel_provider` inside tight loops, and gate telemetry is updated via Python dataclasses on every chunk. Even though `_residual_scope_numba.py` provides Numba helpers, the surrounding loops stay in Python.

**Plan.**
- After landing the GEMM provider, restructure `_collect_residual_scopes_streaming_parallel` so that each block batches kernel requests: gather candidate indices into a contiguous array, pass them to the provider once per chunk, and immediately feed the resulting `float32` tile into a Numba `@njit` routine that filters survivors and updates telemetry structs stored in plain `np.ndarray`s rather than Python objects.
- Move gate telemetry accumulation into a lightweight struct of NumPy scalars; pass it by reference into the Numba kernel and only convert it back to `ResidualGateTelemetry` at the end of traversal.
- Ensure level-cache prefetching (`level_scope_cache` MAP) happens outside the Numba loop; once we enter `_distance_chunk`, no Python callbacks should be needed.
- Document the new “no Python in chunk loop” rule inside `docs/residual_metric_notes.md` and add a micro-benchmark under `benchmarks/` to assert we can stream 512×8 192 tiles without hitting the GIL.
- Acceptance criteria: nopython mode stays engaged; profiler shows negligible Python overhead in traversal once the preceding phases land.

**Key files:** [covertreex/algo/traverse/strategies.py](../covertreex/algo/traverse/strategies.py), [covertreex/algo/_residual_scope_numba.py](../covertreex/algo/_residual_scope_numba.py), [covertreex/metrics/residual/core.py](../covertreex/metrics/residual/core.py), [benchmarks/](../benchmarks).

**Progress:** Not started.

## Phase 7 — Threading & Runtime Guardrails

**Current code.** Neither `cli/queries.py` nor the residual builder constrains OpenMP / MKL / BLAS thread counts. When NumPy calls into MKL for the GEMM provider, it competes with Numba’s parallel loops, hurting determinism.

**Plan.**
- Inside `cli/queries.py` (around `main()` at lines 40-80) add a guard that sets `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, and `NUMBA_NUM_THREADS` to 1 (or a user-provided override) unless the environment already specifies values. Document the defaults in the CLI help text.
- For library consumers, expose `covertreex.runtime.configure_threading(max_threads)` so non-CLI entrypoints can opt in; when defaults need to be applied, log a short warning (“defaulting MKL_NUM_THREADS=1; override via env”) instead of silently overriding user settings.
- Update `docs/CORE_IMPLEMENTATIONS.md` with a note that residual builds assume single-threaded BLAS for kernel tiles, and mention how to raise the limit safely.
- Acceptance criteria: rerunning the same config on a quiet machine stays within ±3 % wall clock, and users retain explicit control via env or the new API.

**Key files:** [cli/queries.py](../cli/queries.py), [covertreex/runtime/__init__.py](../covertreex/runtime/__init__.py), [covertreex/runtime/config.py](../covertreex/runtime/config.py), [docs/CORE_IMPLEMENTATIONS.md](CORE_IMPLEMENTATIONS.md).

**Progress:** Not started.

---

Once these sections land we can revisit the lower-priority gate experiments outlined in `docs/RESIDUAL_PCCT_PLAN.md`, confident that the core traversal cost structure matches the audit recommendations.
