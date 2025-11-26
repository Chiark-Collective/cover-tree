# Rust Residual Optimization Report - 2025-11-26

## Objective
Close the performance gap between Rust (~1.3k - 7k q/s) and Python-Numba (~40k q/s) for the Residual Cover Tree traversal, and validate the efficiency of "Pure" vs "Hybrid" tree construction.

## Experiments & Findings

### 1. Construction Performance (The "Star Tree" Bug)
- **Symptom:** Reports of $O(N^2)$ slowness when building trees with `residual_correlation`.
- **Root Cause:** A shortcut in `batch_insert` checked for `metric.max_distance_hint()`. For bounded metrics, it attached *all* points directly to the root to "save levels," creating a degenerate linked list.
- **Fix:** Removed the shortcut.
- **Result:** Build time for 32k points improved to **~4-6s**, matching/exceeding Python-Numba build speeds.

### 2. Traversal Strategy: Level-Synchronous vs. Heap BFS
- **Heap BFS:** Priority Queue based BFS (Numba-like). ~1.3k q/s. High overhead.
- **Level-Synchronous:** Processes nodes level-by-level. ~7k q/s (Baseline). Better locality.

### 3. Buffer Reuse (The "Chunking" Win)
- **Problem:** Allocating `cached_lb` (size N) for every query was a massive bottleneck.
- **Optimization:** `batch_insert` now uses `par_chunks`. A scratch buffer is allocated once per thread/chunk and reused.
- **Result:** **~12.8k q/s** peak (on smaller tests), stable **~9k q/s** on Gold Standard.

### 4. Pure vs. Hybrid Construction
- **Hybrid:** Build with Euclidean, Query with Residual.
- **Pure:** Build with Residual, Query with Residual.
- **Finding:** 
    - **Correctness:** Both produce identical k-NN results (verified via `tests/test_residual_pure_vs_hybrid_manual.py`).
    - **Performance:** "Pure" (specifically with Hilbert ordering) is faster to build (4.30s vs 5.83s) and slightly faster to query (8.9k vs 7.4k).
- **Conclusion:** The "Hybrid" heuristic is unnecessary. We should proceed with "Pure" construction using spatial reordering (Hilbert) for best results.

### 5. Dead Ends

- **Kernel Fusion:** Manual SIMD (`f32x8`) was slower (~6k q/s) than LLVM auto-vectorization.

- **Lazy Reset:** Generation counters for buffer clearing added net overhead.

- **Visited Set:** Deduplication bitsets added overhead for this density.



### 6. Data Reordering & Caching (The "Double-Mapping" Fix)

- **Problem:** The `ResidualMetric` relies on lookups into large external matrices (`V`, `Coords`). When trees are reordered (e.g., Hilbert), the *tree nodes* are contiguous, but the *data* they point to is scattered, causing cache thrashing.

- **Optimization:** 

    - Implemented `CachedResidualData` to store a copy of the V-matrix and coords, reordered to match the tree's BFS/Hilbert layout exactly.

    - Updated the query kernel to access this cached data linearly (`0..N`).

    - Removed the expensive `new_idx -> old_idx` mapping from the hot loop.

- **Result:** **~64,700 q/s**. This eliminated the last major bottleneck, surpassing Numba.



## Final Status (Gold Standard Benchmark)

*Dataset: 32k Points, 1k Queries, k=50*



| Implementation | Build Time | Query Throughput | Speedup (vs Numba) |

| :--- | :--- | :--- | :--- |

| **Python-Numba** | 7.42s | 41,087 q/s | 1.0x |

| **Rust (Natural)** | 6.31s | 7,302 q/s | 0.18x |

| **Rust (Hilbert) Optimized** | **1.08s** | **64,727 q/s** | **1.58x** |



## Conclusion

The Rust implementation now significantly outperforms the Python-Numba baseline in both build time (**~7x faster**) and query throughput (**~1.6x faster**). The key enablers were:

1.  **Algorithmic Parity:** Adopting Numba's dense bitset and buffer reuse strategies (via `SearchContext`).

2.  **Data Locality:** enforcing 1:1 mapping between tree node order and metric data layout in memory.

3.  **Parallelism:** Effective use of `rayon` for tree construction and batch queries.



## Future Work

- **Algorithmic Parity:** Verify if Numba's "Sparse" traversal (active in recent versions) is providing algorithmic advantages over the dense Level-Synchronous approach.

- **Dynamic Batching:** Further tune `stream_tile` sizes for extremely small or large batches.
