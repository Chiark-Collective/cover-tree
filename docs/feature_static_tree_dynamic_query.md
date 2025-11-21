# Feature: Static Tree / Dynamic Query (VIF)

**Status:** Experimental / Opt-In
**Branch:** `feature/vif-static-tree-dynamic-query`

## Overview
This feature enables the "Static Tree" strategy for Vecchia-Inducing-Points (VIF) approximations on large datasets (e.g., 3D point clouds).

### The Problem
Standard VIF requires a dynamic "Residual Correlation" metric that changes every training step. Rebuilding the Cover Tree ($O(N \log N)$) at every step is too slow for large $N$ ($>100k$) in Python.

### The Solution
We build the Cover Tree **once** using the **Euclidean Metric** (Static). We then use this static tree structure to perform k-NN searches using the **Residual Metric** (Dynamic).

This strategy assumes that **Euclidean proximity is a strong proxy for Residual Correlation**.

## Usage

### 1. Configuration
Enable the feature via `RuntimeConfig` or environment variable:

```bash
export COVERTREEX_RESIDUAL_USE_STATIC_EUCLIDEAN_TREE=1
```

Or in Python:

```python
from covertreex.runtime import configure_runtime, RuntimeConfig
import dataclasses

config = RuntimeConfig.from_env()
config = dataclasses.replace(config, residual_use_static_euclidean_tree=True)
configure_runtime(config)
```

### 2. Workflow

```python
from covertreex.api.pcct import PCCT
from covertreex.metrics.residual import ResidualCorrHostData, configure_residual_correlation

# 1. Setup Residual Backend (Dynamic State)
host_data = ResidualCorrHostData(v_matrix=..., p_diag=..., ...)
configure_residual_correlation(host_data)

# 2. Build Tree (Static Euclidean)
# Pass raw coordinates. The tree will store them.
pcct = PCCT()
tree = pcct.fit(X_coords) 

# Update PCCT wrapper with the tree AND the configured runtime
from covertreex.api.runtime import Runtime
pcct = dataclasses.replace(pcct, tree=tree, runtime=Runtime.from_config(config))

# 3. Query (Dynamic Residual)
# Pass indices matching the backend's data order.
# The query will use the Euclidean tree structure but compute Residual distances.
neighbors = pcct.knn(query_indices, k=10) 
```

## Implementation Details

*   **`residual_knn_query`**: A Python-based Best-First Search that traverses the Euclidean tree.
    *   **Priority**: Nodes are visited based on their *Euclidean* proximity to the query (or parent's residual distance heuristic).
    *   **Distance**: Exact distances are computed using the `ResidualCorrHostData` backend (Numba-accelerated kernels).
    *   **Batched Evaluation**: Candidate leaves are evaluated in batches of 32 to amortize Python/Kernel overhead.
*   **Identity Fallback**: If the tree stores Coordinates (floats) but the Residual Backend expects Indices (ints), the query engine automatically falls back to an **Identity Mapping** (assuming `tree.points[i]` corresponds to `dataset_index=i`), provided the sizes match.

## Limitations
1.  **Performance**: The current query implementation is pure Python (with Numba kernels for distance). It is significantly slower than the native Numba query engine (~40 q/s vs >2000 q/s). **Next Step:** Port `residual_knn_query` logic to Numba.
2.  **Accuracy**: The accuracy (Recall) depends entirely on how well the Euclidean structure captures the Residual correlation. If the residual process has long-range correlations not captured by Euclidean distance, the tree might prune valid neighbors.
3.  **Pruning**: Currently, the implementation performs a "Best-First Search" without strict pruning (it explores based on priority). Rigorous pruning requires a lower-bound function $LB_{res}(node) \ge f(LB_{euc}(node))$, which is not yet implemented.
