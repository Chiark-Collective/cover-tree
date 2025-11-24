# High-Performance Cover Tree Roadmap: Rust & Hybrid Architectures

**Date:** 2025-11-24
**Context:** Following the successful integration of the Dense Scope Streamer and Scope Caps into the Rust backend, we have achieved a ~200x speedup in Rust queries. However, a performance gap remains compared to the highly optimized Python/Numba query path. This document outlines two distinct avenues to close this gap and potentially surpass current state-of-the-art performance.

## Current State (Gold Standard 32k)

| Engine | Implementation | Order | Build Time | Query Time | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **python-numba** | Numba JIT | Natural | 16.10s | **0.05s** | **21,123 q/s** |
| **rust-natural** | Rust Native | Natural | 7.42s | 0.14s | 7,105 q/s |
| **rust-hilbert** | Rust Native | Hilbert | **4.20s** | 0.21s | 4,971 q/s |

---

## Avenue 1: Native Rust Optimization
**Goal:** Bring `rust-natural` and `rust-hilbert` query throughput up to parity with `python-numba` (~22k q/s) while maintaining pure Rust portability.

### 1. Level Caching (The "Missing Link")
The single largest differentiator between the implementations is the **Level Cache**.
*   **Python/Numba:** Maintains a cache of "valid scope members" from the parent level. When descending, it intersects the parent's scope with the child's potential scope. This often avoids re-scanning the entire child list or re-computing distances for points already known to be far.
*   **Rust (Current):** Re-scans children and re-evaluates heuristics at every level.
*   **Implementation Plan:**
    *   Implement a per-query (or thread-local) cache structure in Rust.
    *   Requires careful memory management to avoid allocations in the hot loop (e.g., reusing a `Vec<usize>` or a fixed-size buffer).
    *   Port the "semisort" or "masked append" logic to efficiently filter survivors from level $i$ to $i-1$.

### 2. Dynamic Block Sizing
*   **Python/Numba:** Dynamically adjusts the vector block size based on the number of active queries at a specific tree level. This maximizes SIMD utilization and minimizes work for "straggler" queries.
*   **Rust (Current):** Uses a fixed block size (default 64 or 256).
*   **Implementation Plan:**
    *   Refactor the batch driver to group queries by their current tree level.
    *   Process level-groups together, allowing the block size to adapt to the "wavefront" of the search.

### 3. Hilbert-Specific Tuning
`rust-hilbert` builds fastest but queries slowest.
*   **Hypothesis:** Hilbert ordering creates highly dense, spatially coherent sub-trees. The current "budget" heuristics (which stop searching after seeing $N$ survivors) might be saturating early on "local" points that aren't actually the nearest neighbors for a distant Maximin query, effectively "blinding" the search to better candidates in other branches.
*   **Fix:** Tune `BUDGET_UP` / `BUDGET_DOWN` specifically for Hilbert-ordered trees, or implement a "diversity" heuristic that forces the traversal to check sibling branches even if the local budget is full.

---

## Avenue 2: The "Super Engine" (Hybrid Architecture)
**Goal:** Combine the superior build speed of `rust-hilbert` with the superior query speed of `python-numba` to create an engine that dominates both metrics.

**Target Performance:** ~4.2s Build + ~0.05s Query = **~4.25s Total** (vs 16.1s current best).

### Architecture
The workflow would be:
1.  **Rust Build:** Use `rust-hilbert` logic to construct the Cover Tree. This handles the heavy lifting (conflict graph, rigorous contraction, reordering) in compiled, parallel Rust.
2.  **Zero-Copy Transfer:** Expose the internal struct-of-arrays (`parents`, `children`, `levels`, `points`) from Rust to Python.
    *   Use `PyO3` to return these vectors as NumPy arrays (or read-only views).
    *   This avoids serialization overhead.
3.  **Python Wrapper:** Construct a `PCCTree` object in Python wrapping these Rust-generated arrays.
4.  **Numba Query:** Pass this `PCCTree` to the existing `_ResidualTraversal` strategy. The Numba JIT kernels will run against the memory allocated by Rust.

### Implementation Steps
1.  **Expose Internals:** Add a method to `CoverTreeWrapper` (e.g., `export_arrays()`) that returns a tuple of `(parents, children, levels, points, next_nodes)` as `PyArray` objects.
2.  **Engine Binding:** Create a new engine class (e.g., `HybridRustNumbaEngine`) in `covertreex/engine.py`.
    *   `build()` calls `rust-hilbert` backend.
    *   Extracts arrays.
    *   Instantiates `PCCTree(backend=numpy, ...)` with the arrays.
3.  **Verify Memory Layout:** Ensure Rust `Vec<T>` layout matches what Numba expects (C-contiguous). (It should, as standard `Vec` is contiguous).

## Comparison & Recommendation

| Feature | Avenue 1 (Native Rust) | Avenue 2 (Hybrid) |
| :--- | :--- | :--- |
| **Query Speed** | High (Potential) | **Maximum (Proven)** |
| **Build Speed** | High | **Maximum** |
| **Complexity** | High (Algorithm porting) | Medium (Glue code) |
| **Portability** | **Pure Rust** | Requires Python/Numba |
| **Dependencies** | Low | High (Numba, LLVM) |

**Recommendation:**
1.  **Implement Avenue 2 (Hybrid) immediately.** It provides the highest immediate performance gain with the lowest algorithmic risk, leveraging code that is already proven to work (Numba query + Rust build).
2.  **Pursue Avenue 1 (Native)** as a long-term goal to remove the Python dependency for production deployments that require a standalone binary.
