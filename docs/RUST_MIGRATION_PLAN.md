# Plan: Migrating PCCT Core to Rust

**Status:** Proposal
**Date:** 2025-11-21
**Target:** High-performance backend for `covertreex` to enable scaling to $N=10^6+$ on CPU.

## 1. Motivation & Goal

The current Python/Numba implementation of the Parallel Compressed Cover Tree (PCCT) hits a performance wall on large datasets ($N > 100k$), particularly in low dimensions ($D=8$) where "crowding" forces deep trees. Construction times of ~10 minutes for 200k points are unacceptable for VIF training loops.

Profiling indicates the bottleneck is **Python Interpreter Overhead**:
-   Managing millions of node indices and list/dict structures.
-   Recursion depth limits and stack overhead.
-   The Global Interpreter Lock (GIL) preventing true multi-core scaling for complex logic like the Conflict Graph.

**Goal:** Replace the core algorithms (`batch_insert`, `knn`) with a **Rust extension module**. This aims to reduce build times from **minutes to seconds** and query latency to microseconds, matching or exceeding C++ reference implementations while maintaining a Pythonic API.

## 2. Architecture: The Hybrid Model

We will maintain `covertreex` as a Python package but introduce a compiled binary extension.

*   **`covertreex` (Python):** The user-facing API (`PCCT`, `RuntimeConfig`), plotting, telemetry, and high-level orchestration.
*   **`_covertreex_backend` (Rust):** A PyO3 module containing the heavy lifting.
    *   **Memory Management:** Owns the tree data buffers (flat `Vec<T>`).
    *   **Parallelism:** Uses `rayon` for thread-safe data parallelism.
    *   **SIMD:** Uses explicit or compiler-auto-vectorized SIMD for distances.

### Interop Strategy (PyO3 + NumPy)
We will use `numpy` arrays as the primary data exchange format.
-   **Zero-Copy:** Python passes `np.ndarray` to Rust. Rust interprets them as slices (`&[T]`) via `numpy` crate.
-   **Ownership:** The Rust `PCCTree` struct will likely own its internal vectors. When exposed to Python, we can either expose views or copy data out on demand. For maximum performance, the tree lives in Rust heap, and Python holds an opaque handle (PyCapsule/PyClass).

## 3. Phase 1: Infrastructure & Skeleton

**Objective:** Establish the build chain (`maturin`) and basic type wrapping.

1.  **Tooling:** Initialize a Rust crate inside the repo (e.g., `rust/`). Configure `pyproject.toml` to use `maturin` backend.
2.  **Types:** Define the `CoverTree` struct in Rust.
    ```rust
    struct CoverTree {
        points: Vec<f32>,      // Flattened N*D
        parents: Vec<i32>,     // N
        children: Vec<i32>,    // CSR-like or First-Child/Next-Sibling
        levels: Vec<i32>,      // N
        // ... other structural fields
    }
    ```
3.  **Bindings:** Expose `PCCTreeWrapper` class to Python via PyO3.
4.  **CI/CD:** Ensure `uv sync` or `pip install` builds the Rust extension.

## 4. Phase 2: Query Engine (Static Tree)

**Objective:** Port `knn` traversal first to validate performance gains on read-heavy workloads.

1.  **Distance Trait:** Define a `Metric` trait in Rust.
    ```rust
    trait Metric {
        fn distance(&self, idx_a: usize, idx_b: usize) -> f32;
        fn distance_to_point(&self, idx_a: usize, point: &[f32]) -> f32;
    }
    ```
2.  **Euclidean Impl:** Implement standard L2.
3.  **Traversal:** Port `knn_numba` logic to Rust.
    -   Heap management (Rust `BinaryHeap` is fast).
    -   Memory layout optimization (Structure of Arrays vs Array of Structures).
4.  **Benchmark:** Compare `rust_knn` vs `numba_knn`.

## 5. Phase 3: Construction (Batch Insert)

**Objective:** The core deliverable. Port the complex Parallel Batch Insert logic.

1.  **Conflict Graph:** Implement the graph construction in Rust.
    -   `rayon::par_iter` to compute pairwise distances and build adjacency lists.
    -   This is where Python overhead is currently highest (managing lists of edges). Rust `Vec<Vec<i32>>` or flat adjacency arrays will be 100x faster.
2.  **MIS (Maximal Independent Set):** Port the Luby/Greedy MIS algorithms. `rayon` is perfect for the parallel state updates.
3.  **Batch Orchestration:** Port the level-by-level descent logic.
4.  **Result:** `tree.fit(X)` runs fully in native code.

## 6. Phase 4: Residual Metric Integration (VIF)

**Objective:** Support the Dynamic Residual metric without crossing the Python/Rust boundary per-distance.

1.  **Data Passing:** The `ResidualCorrHostData` (V matrix, P diag) consists of large arrays. Pass these pointers to the Rust backend *once*.
2.  **Metric Impl:** Implement the `Metric` trait for `ResidualMetric`.
    -   Uses BLAS/Lapack (via `ndarray` or `cblas-sys`) or manual SIMD for `V_i . V_j` dot products.
    -   Since `V` is low rank ($R \approx 16-32$), manual AVX2/AVX512 loops might beat BLAS overhead for single-point distances.
3.  **Callback:** If the kernel is arbitrary Python code, we might need a callback.
    -   *Performance Risk:* Calling Python from Rust loop is slow.
    -   *Mitigation:* Support "Standard Kernels" (RBF, Matern) natively in Rust. Only fall back to Python for exotic kernels (accepting the perf hit).

## 7. Roadmap & Tasks

### Step 0: Setup
- [ ] Create `src/lib.rs` and `Cargo.toml`.
- [ ] Set up `maturin` build.

### Step 1: Euclidean MVP
- [ ] Implement `CoverTree` struct (owning data).
- [ ] Implement `knn` (brute force first, then tree).
- [ ] Expose to Python.

### Step 2: Fast Build
- [ ] Implement `batch_insert`.
- [ ] Benchmark against current Numba implementation.

### Step 3: Residual VIF
- [ ] Implement `ResidualMetric` struct.
- [ ] Bind it to `PCCTree` in Rust.
- [ ] Verify correlation/recall against Python baseline.

## 8. Why this solves the problem
By moving the **structure management** to Rust, we remove the $O(N)$ Python operations. Even if we perform $10^7$ distance calculations, doing them in a tight Rust loop with `rayon` overhead is negligible compared to doing them in Python loops or even Numba-invoked-from-Python loops.

This enables the "Rebuild Strategy" (GPBoost style) to be viable: if Build takes < 1s, we can rebuild every few steps.
