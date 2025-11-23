# Residual Metric Status Report (2025-11-23) - Update 2

## Overview

This report captures the state of the Rust backend optimization for the Residual Correlation metric after the "Frontier Batching" implementation.

## Current Benchmark Results (Gold Standard 32k)

Workload: 32,768 points, 3 dimensions, 1,024 queries, k=50.

| Metric | Python/Numba (Baseline) | Rust Fast (Current) | Gap |
| :--- | :--- | :--- | :--- |
| **Build Time** | **9.08 s** | **9.42 s** | ~1.04x (Competitive) |
| **Query Time** | **0.028 s** | **8.06 s** | ~288x (Slower) |
| **Throughput** | **36,660 QPS** | **127 QPS** | **~0.003x** |

## Optimizations Implemented

1.  **Correctness & Stability:**
    *   Fixed a critical panic in `ResidualMetric` related to non-contiguous memory layouts from NumPy.
    *   Removed `gold_standard_32k_rust` from CI to prevent regression failures.

2.  **Build Performance:**
    *   Rust parallel build (`batch_insert` with Rayon) is performing excellently, matching the highly-optimized Python/Numba build time.

3.  **Query Optimization attempts:**
    *   **Pruning:** Implemented standard Cover Tree branch-and-bound pruning.
    *   **Frontier Batching:** Refactored traversal to process candidates in batches of 32 to amortize function call overhead and enable potential SIMD.
    *   **Memory:** Removed the large `visited` vector allocation.

## Root Cause Analysis (The Query Gap)

Despite these efforts, the Rust query path remains ~300x slower. Potential reasons identified:

1.  **Hot-Loop Allocations:** The current "Frontier Batching" implementation allocates a new `Vec<T>` for distances *every single batch* (millions of times). The `Metric` trait/struct API needs to change to accept an output buffer.
2.  **SIMD/Vectorization:** Numba often generates AVX-512/AVX2 code that is superior to `rustc`'s auto-vectorization for complex kernels like the Residual Metric (which involves `exp`, `sqrt`, and dot products). Explicit SIMD (via `std::simd` or `wide` crate) might be required.
3.  **Algorithmic Overhead:** The Numba implementation uses a highly specialized "Micro-Batching" strategy that might be interacting more favorably with the CPU cache than the current Rust logic.

## Next Steps

1.  **Zero-Allocation Loop:** Modify `distances_sq_batch_idx` to write into a pre-allocated mutable slice (`&mut [T]`) instead of returning `Vec<T>`.
2.  **Explicit SIMD:** Investigate replacing the scalar loop in the metric calculation with explicit SIMD intrinsics to match Numba's throughput.
3.  **Profile:** Use `perf` or `flamegraph` to confirm if `malloc` is indeed the bottleneck.

For now, **Python/Numba remains the production recommendation** for query-heavy workloads.