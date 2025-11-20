# VIF Strategy: Handling Dynamic Residual Metrics

## The Problem
In VIF (Vecchia-Inducing-Points Full-Scale) approximations, the distance metric is defined by the residual correlation:
$$ d_{res}(x, y) = f(k_\theta(x, y), Z) $$
This metric changes whenever the hyperparameters $\theta$ or the inducing points $Z$ are updated (i.e., every training step).

Since building a Cover Tree takes $O(N \log N)$ time (e.g., ~60s for 200k points in $D=64$, or ~10m in $D=8$), rebuilding the index inside the training loop is computationally infeasible.

## The Solution: Static Tree, Dynamic Query

We exploit the fact that the underlying spatial data (3D point clouds) has a **static topology**. Points that are far apart in Euclidean space are almost certainly uncorrelated, regardless of $\theta$.

### 1. Build Phase (Once)
Construct the PCCT using the **Euclidean** metric on the input coordinates $X$.
*   **Cost:** Paid once at initialization.
*   **Result:** A spatial index satisfying Euclidean covering invariants.

### 2. Update Phase (Every Step)
Update the global residual backend with the new kernel state.
*   **Action:** Update `ResidualCorrHostData` matrices (v-matrix, diagonal variances) and kernel provider.
*   **PCCT API:** Call `covertreex.metrics.residual.configure_residual_correlation` with the new state. This updates the global `_ACTIVE_BACKEND` pointer instantly.
*   **Cost:** $O(1)$ (pointer swap) or $O(N)$ (if recomputing diagonal cache).

### 3. Query Phase (Every Step)
Perform k-NN or Range Search on the **static Euclidean tree**, but using the **Dynamic Residual Metric**.
*   **Logic:** The traversal uses the Euclidean tree structure to guide the search.
*   **Pruning:** We use a **Branch-and-Bound** strategy.
    *   The Euclidean bounding ball of a tree node provides a **lower bound** on the Euclidean distance to the query.
    *   Since the stationary kernel $k(r)$ decays with distance, this lower bound on distance translates to an **upper bound on correlation** (or lower bound on residual distance).
    *   If `LowerBound(ResidualDist) > CurrentKthDist`, prune the node.
*   **Leaf Refinement:** Exact residual distances are computed only for points that survive the spatial pruning.

## Why PCCT Fits This Niche
Standard KD-trees (`scipy.spatial.cKDTree`) support only Minkowski metrics ($L_p$) and cannot handle the custom matrix math required for $d_{res}$.

PCCT is specifically architected for this:
1.  **Decoupled Backend:** The `residual` metric backend is separate from the tree structure.
2.  **Custom Kernels:** It compiles the expensive kernel math (GEMM, Cholesky solves) via Numba/BLAS, keeping the "Leaf Refinement" step fast.
3.  **High-Dimensional Robustness:** As shown in benchmarks, PCCT scales to high-D feature spaces where standard trees fail, which is critical if your kernel operates in a high-D feature space or if you use many inducing points.

## Summary
**Do not rebuild the tree.** Use the Euclidean structure as a spatial pre-filter for your dynamic residual metric.
