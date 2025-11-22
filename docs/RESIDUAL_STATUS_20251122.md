# Residual Metric Status Report (2025-11-22)

## Overview

This report summarizes the current state of the **Residual Correlation** metric implementation, comparing the production Python/Numba path against the experimental Rust backend. The benchmark workload matches the historical "Gold Standard" (32,768 points, 3 dimensions, 1,024 queries, k=50).

## Key Findings

1.  **Python/Numba is the Performance Leader:**
    The current Python implementation (utilizing parallel conflict graphs and Numba kernels) has surpassed historical baselines.
    *   **Build Time:** 9.06s (vs historical ~24.2s)
    *   **Query Throughput:** ~36,222 QPS (vs historical ~21,700 QPS)
    *   **Speedup vs GPBoost Baseline:** ~727x

2.  **Rust Backend (Residual) is Underperforming:**
    The experimental `rust-fast` engine, which implements the full Cover Tree construction and traversal for the Residual metric in Rust, is currently significantly slower than the Python path.
    *   **Build Time:** ~37s (~4x slower than Python)
    *   **Query Throughput:** ~21 QPS (~1700x slower than Python)
    *   **Analysis:** The Rust implementation likely suffers from inefficient traversal logic or excessive synchronization overhead for this specific metric. It fails to leverage the structural optimizations (like the conflict graph) as effectively as the Python path for this workload.

3.  **Telemetry Gap:**
    The Rust backend does not currently emit the detailed, batch-level telemetry (traversal ms, conflict graph ms) required by the `export_benchmark_diagnostics.py` tool, causing automated regression jobs to fail when it is enabled.

## Recommendations

*   **Maintain Python as Default:** The `python-numba` engine should remain the default for all Residual metric workloads.
*   **Disable Rust in Regression:** The `gold_standard_32k_rust` job has been removed from the automated suite to prevent CI failures.
*   **Focus Rust Optimization:** Future work on the Rust backend should investigate the query traversal bottleneck. The "Static Tree" hybrid approach (Euclidean Build in Rust + Residual Query) showed promise in earlier ad-hoc tests (28k QPS) and might be a better direction than the pure Residual tree.

## Reproducing Results

To run the current gold standard benchmark (Python/Numba):

```bash
./benchmarks/run_residual_gold_standard.sh
```
