# Benchmark Expansion Plan

## Objective
To rigorously evaluate the performance of the **Parallel Compressed Cover Tree (PCCT)** against established State-of-the-Art (SotA) implementations and standard Python baselines. This suite aims to validate PCCT's "SotA-competitive" status across varying scales, dimensions, and metrics.

## Target Baselines

We will compare PCCT against the following implementations, available via the `baseline` dependency group:

1.  **`mlpack` (C++ Bindings)**:
    *   *Role:* Primary SotA competitor for Cover Trees.
    *   *Metric:* Euclidean (L2).
    *   *Notes:* Known for stability and speed.

2.  **`scikit-learn` (BallTree / KDTree)**:
    *   *Role:* The "Standard" Python baseline.
    *   *Metric:* Euclidean (L2).
    *   *Notes:* Highly optimized Cython. `BallTree` handles higher dimensions better than `KDTree`.

3.  **`scipy.spatial.cKDTree`**:
    *   *Role:* Low-overhead, standard KD-Tree.
    *   *Metric:* Euclidean (L2).
    *   *Notes:* Very fast for low dimensions ($D < 20$).

4.  **`covertree` (PyPI)**:
    *   *Role:* Alternative Python/Cython Cover Tree implementation.
    *   *Metric:* L2.
    *   *Notes:* Drop-in replacement for `scipy`, provides a direct algorithmic comparison.

5.  **PCCT (Ours)**:
    *   *Variants:*
        *   `residual`: Complex metric, highly optimized parallel build.
        *   `euclidean_sparse`: The newly optimized O(N log N) path.

## Experimental Sweep

### 1. Synthetic Scaling (Gaussian)
Control dataset scale and dimension to stress-test complexity classes.

*   **Points (N):** `[10_000, 50_000, 100_000, 500_000, 1_000_000]`
*   **Dimension (D):** `[3, 16, 64, 128]`
*   **Queries (Q):** `10_000` (Fixed for throughput measurement)
*   **K:** `[1, 10]`

### 2. Realistic Data Proxies
Simulate real-world ANN workloads (even though we are doing exact NN).

*   **"Bio/Physics" Proxy:** 3D points, very large N (`1M`). High density locally.
*   **"Embedding" Proxy:** 128D points, medium N (`100k`). The "Curse of Dimensionality" test.

### 3. Metrics Captured
For each `(Impl, N, D, K)` tuple, record:
*   **Build Time (s):** Wall clock time to construct the index.
*   **Query Latency (ms):** Average latency per batch (or per query).
*   **Throughput (QPS):** Queries per second.
*   **Correctness:** Recall (should be 1.0 for exact methods).

## Implementation Strategy

1.  **Benchmark Driver:** Create `benchmarks/comprehensive_sweep.py` using the existing `cli.pcct` and `cli.pcct.support.benchmark_utils` infrastructure.
2.  **Adapters:** Ensure `scikit-learn` and `scipy` adapters exist in `covertreex.baseline` (we already have `mlpack` and `covertree` support, need to double check `scikit-learn`).
3.  **Execution:**
    *   Run sweeping script.
    *   Output JSONL telemetry.
    *   Generate comparison tables/plots.

## Action Items
- [ ] Verify `scikit-learn` and `scipy` adapters in `covertreex/baseline.py`.
- [ ] Create the sweep script.
- [ ] Run a "Smoke Test" sweep (small N) to verify all adapters work.
- [ ] Execute full sweep (Time permitting).
