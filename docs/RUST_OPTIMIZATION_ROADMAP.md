# Optimization Roadmap: Closing the Gap to Numba

## Problem Statement
Current Rust throughput: **~10.3k q/s** (SIMD-optimized).
Target Numba throughput: **~39.0k q/s**.
Gap: **3.8x**.

Analysis shows the bottleneck has shifted from **Arithmetic Intensity** (solved by SIMD) to **Memory Latency** (the "Gather" problem).

## Evaluation of Approaches

We compared two architectural changes to address the gather bottleneck: **Dual-Tree Traversal** and **Contiguous Memory Layout**.

### 1. Dual-Tree Traversal (Batch-Query vs. Batch-Node)
*   **Concept:** Instead of processing one query against the tree, process a batch of queries ($Q$) against a batch of tree nodes ($N$).
*   **Mechanism:** Collect all children of a set of active nodes, gather their data *once*, and compute a $Q \times N$ distance matrix.
*   **Pros:**
    *   **Amortization:** The cost of fetching a node's data is shared across all queries in the batch.
    *   **Compute Density:** Allows use of SGEMM (Matrix-Matrix multiplication), efficiently utilizing FPU pipelines.
*   **Cons:**
    *   **Complexity:** Requires a complete rewrite of the traversal logic (`single_residual_knn_query` is deeply recursive/DFS). Managing the "frontier" for multiple queries simultaneously is non-trivial.
    *   **Pruning Granularity:** Batching reduces pruning effectiveness. If one query needs to visit a node, the whole batch effectively "visits" it (or complex masking is required), potentially wasting work.

### 2. Contiguous Memory Layout (Data-Oriented Design)
*   **Concept:** Restructure the tree and dataset so that data is accessed sequentially, not randomly.
*   **Mechanism:**
    1.  **CSR Topology:** Replace the Linked-List child structure (`first_child`, `next_node`) with a Compressed Sparse Row format (`children_offset[node]`, `children_len[node]`). Children of a node are stored contiguously in a `children` array.
    2.  **Dataset Reordering:** Permute the underlying `v_matrix` and `coords` arrays to match the tree's traversal order (e.g., BFS or DFS order).
*   **Pros:**
    *   **Zero Gather:** If children indices are contiguous ($i, i+1, \dots, i+8$), we can load data using **Vector Loads** (`vmovups`) instead of **Scalar Gathers**. This is a massive latency reduction.
    *   **Cache Locality:** Prefetchers can predict sequential access perfectly.
    *   **Simplicity:** The traversal logic remains mostly the same (DFS/BFS), just the iteration mechanism changes (`for child in slice` vs `while child != -1`).
*   **Cons:**
    *   **Preprocessing:** Requires re-indexing the dataset at build time (O(N) copy).

## Recommendation: Contiguous Memory Layout

**Winner: Contiguous Memory Layout (CSR + Reorder)**

### Reasoning based on Codebase Analysis:
1.  **Current Bottleneck:** `algo.rs` spends significant time in `metric.distances_sq_batch_idx_into...`. Inside this function, `p_indices` are random. The code explicitly gathers: `norms[k] = scaled_norms[p_idxs[k]]`. This scalar gather defeats the purpose of SIMD if the CPU is waiting for L3 cache lines on every load.
2.  **Traversal Overhead:** The current `CoverTreeData` uses a linked-list structure (`next_node`). This incurs a dependent load per child visited (`child = next_node[child]`). A CSR approach eliminates this entirely.
3.  **Feasibility:** Implementing Dual-Tree requires a paradigm shift. Implementing Layout Optimization is a transformation of the *storage*, keeping the *logic* largely similar.

### Implementation Plan
1.  **Refactor `CoverTreeData`:** Move to `children_offsets: Vec<usize>` and `children: Vec<NodeId>`.
2.  **Reorder Strategy:** When building the tree, output a permutation vector $P$ that maps `TreeIndex -> DatasetIndex`.
3.  **Permute Metric Data:** Create a `ReorderedResidualMetric` that stores `v_matrix` and `coords` in `TreeIndex` order.
4.  **SIMD Load:** Rewrite `distances_sq_...` to take a `start_idx` and `count` instead of `indices: &[usize]`. Use `std::slice::from_raw_parts` + `f32x8::from_slice` (aligned load).

**Expected Impact:** Closing the remaining 4x gap. Numba likely benefits from cache-friendly arrays or JIT-fused loops that hide gather latency. Explicitly fixing the layout in Rust will match or beat that performance.