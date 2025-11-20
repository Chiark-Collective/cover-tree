# Benchmark Results: PCCT vs Baselines (2025-11-20)

This report summarizes the performance of the **Parallel Compressed Cover Tree (PCCT)** against standard Python libraries (`scikit-learn`, `scipy`) and the C++ `mlpack` library.

**Highlights:**
*   **High Dimensions (D=64):** PCCT is dominant, achieving **10-15x higher query throughput** than `scikit-learn` and `scipy`.
*   **Low Dimensions (D=8):** PCCT remains competitive, beating `scikit-learn` and trailing `scipy`'s highly optimized KD-tree by a factor of ~2x.
*   **Scalability:** PCCT maintains high throughput (~10k+ q/s) even as dimensions increase, whereas KD-trees degrade to brute-force speeds.

## 1. Summary Table

| Metric | N | D | Implementation | Build (s) | Throughput (q/s) | Speedup (vs PCCT) | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Euclidean** | **8,192** | **8** | **PCCT** | 1.00 | **104,288** | **1.0x** | Baseline |
| Euclidean | 8,192 | 8 | Scipy (cKDTree) | 0.001 | 73,945 | 0.71x | Fast low-D |
| Euclidean | 8,192 | 8 | Sklearn (BallTree) | 0.003 | 33,045 | 0.32x | |
| Euclidean | 8,192 | 8 | MLPack | 0.78 | 9,488 | 0.09x | C++ Cover Tree |
| | | | | | | | |
| **Euclidean** | **50,000** | **8** | **PCCT** | 26.97 | **21,202** | **1.0x** | |
| Euclidean | 50,000 | 8 | Scipy (cKDTree) | 0.010 | 21,559 | 1.02x | Tie |
| Euclidean | 50,000 | 8 | Sklearn (BallTree) | 0.022 | 16,200 | 0.76x | |
| | | | | | | | |
| **Euclidean** | **100,000** | **8** | **PCCT** | 111.38 | **10,840** | **1.0x** | |
| Euclidean | 100,000 | 8 | Scipy (cKDTree) | 0.019 | 26,314 | 2.43x | KD-tree wins low-D |
| Euclidean | 100,000 | 8 | Sklearn (BallTree) | 0.052 | 10,200 | 0.94x | Tie |
| | | | | | | | |
| **Euclidean** | **10,000** | **64** | **PCCT** | 1.38 | **49,294** | **1.0x** | **High-D Winner** |
| Euclidean | 10,000 | 64 | Scipy (cKDTree) | 0.006 | 3,343 | 0.07x | 15x Slower |
| Euclidean | 10,000 | 64 | Sklearn (BallTree) | 0.014 | 4,718 | 0.10x | 10x Slower |
| Euclidean | 10,000 | 64 | MLPack | 6.96 | 1,656 | 0.03x | 33x Slower |

## 2. Detailed Analysis

### High Dimensionality (D=64)
This is the primary use case for Cover Trees. Standard partitioning trees (KD-trees, Ball Trees) suffer from the "curse of dimensionality," often degrading to $O(N)$ linear scans.
*   **PCCT** sustains **~50k q/s**, demonstrating effective pruning.
*   **Baselines** collapse to **3-4k q/s**.
*   **Result:** PCCT provides a **10x-15x speedup** over standard Python tools.

### Low Dimensionality (D=8)
In low dimensions, KD-trees are theoretically optimal and `scipy.spatial.cKDTree` is a highly optimized C implementation.
*   **PCCT** is surprisingly competitive, matching or beating `sklearn` BallTree.
*   **Scipy** is faster (2.4x at 100k points) due to simpler invariants and C-level optimization, but PCCT is "in the same ballpark" (10k vs 26k q/s), which is excellent for a structure generalized for complex metrics.

### Build Times
*   **PCCT:** ~1s (8k) to ~111s (100k). Python/Numba implementation of complex invariants.
*   **Baselines:** Milliseconds. Optimized C/C++ construction of simpler trees.
*   **Trade-off:** PCCT invests upfront in a high-quality index to enable faster querying in difficult (high-D/complex metric) regimes.

## 3. Reproduction

To reproduce these results, use the `cli.pcct` tool:

```bash
# 1. Install baselines
uv pip install -e ".[baseline]"

# 2. Run benchmarks
# Low-D Comparison
python -m cli.pcct query --metric euclidean --tree-points 50000 --dimension 8 --baseline scipy
python -m cli.pcct query --metric euclidean --tree-points 50000 --dimension 8 --baseline sklearn

# High-D Comparison (PCCT Dominance)
python -m cli.pcct query --metric euclidean --tree-points 10000 --dimension 64 --baseline scipy
python -m cli.pcct query --metric euclidean --tree-points 10000 --dimension 64 --baseline sklearn
```
