# Euclidean Metric Optimization Report

## Overview
Following the successful optimization of the Residual metric path, we applied similar principles to the **Euclidean (L2) metric** path. Previously, the default "Dense" Euclidean strategy used a brute-force $O(N^2)$ flat scan for parent finding and scope collection, causing poor scaling for datasets $N > 10k$.

## Optimization: Automatic Sparse Traversal Upgrade

We modified the `euclidean_dense` traversal strategy to automatically detect if **Numba** is available. If so, it now delegates the workload to the `euclidean_sparse` implementation, which utilizes:

1.  **Single-Tree Traversal (`knn_numba`)**: Finds parents in $O(\log N)$ using tree pruning (Triangle Inequality).
2.  **Sparse Scope Collection (`collect_sparse_scopes`)**: Finds conflict scopes using tree pruning, avoiding the construction of a dense $N \times N$ mask.

This effectively changes the complexity class of the default path from $O(N^2)$ to near $O(N \log N)$ without requiring user configuration.

### Configuration Change
We updated `RuntimeModel` to enable Numba **by default** if the library is installed. Previously, it defaulted to `False`, requiring explicit opt-in. This ensures that standard installations get the optimized performance out-of-the-box.

## Benchmark Results (N=32,768)

| System | Metric | Build Time | Query Throughput |
| :--- | :--- | :--- | :--- |
| **PCCT (Before)** | Euclidean | **~83.0s** | - |
| **PCCT (After)** | Euclidean | **11.94s** | **31,042 q/s** |
| **MLPack (Baseline)** | Euclidean | 10.97s | 3,038 q/s |

*   **Speedup:** Build time reduced by **~7x**.
*   **Parity:** PCCT build time is now comparable to the C++ MLPack baseline (11.9s vs 11.0s).
*   **Throughput:** PCCT retains its **10x** advantage in query throughput.

## Reproduction
```bash
python -m cli.pcct query --metric euclidean --tree-points 32768 --baseline mlpack
```
Look for `pcct | build=...`.

```