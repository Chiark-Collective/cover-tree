import time
import numpy as np
from covertreex.api.pcct import PCCT
from covertreex.runtime import configure_runtime, RuntimeConfig
from covertreex.metrics.residual import (
    ResidualCorrHostData,
    configure_residual_correlation,
    compute_residual_pairwise_matrix
)

def generate_synthetic_data(n_points, d=3, rank=16, seed=42):
    """Generate synthetic data: Coordinates X and Latent Factors V."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_points, d)).astype(np.float32)
    
    # Generate random V matrix (N, Rank) simulating L^{-1} K(X, U)
    V = rng.normal(size=(n_points, rank)).astype(np.float32)
    
    # Diagonal variances (simulated)
    # p_i = K(x,x) - ||v_i||^2. Let's just make it positive random.
    p_diag = rng.uniform(0.1, 1.0, size=n_points).astype(np.float32)
    
    # Kernel Diagonal (K(x,x))
    kernel_diag = np.sum(V**2, axis=1) + p_diag
    
    return X, V, p_diag, kernel_diag

def mock_kernel_provider(row_idx, col_idx):
    """Slow mock kernel provider for fallback/verification."""
    # In a real scenario, this computes K(X_i, X_j).
    # Here we just return identity or random for simplicity, 
    # BUT since VIF uses (K - VV')/sqrt(...), we need consistency.
    # Let's assume we don't use the kernel_provider for the V-matrix path
    # except for the 'distance_block_no_gate' check which usually needs it?
    # Actually, `compute_residual_distances` uses kernel provider for the exact residual distance.
    # d = sqrt(1 - |rho|). rho = (K - v.v') / sqrt(sig*sig)
    # So we need a consistent K.
    # Let's fake K such that K_ij = (v_i . v_j) + delta_ij * p_i + small_noise
    # This implies rho = small_noise / sqrt... ~= 0.
    # To make it interesting, let's make K_ij = (v_i . v_j) + random_correlation.
    
    # This is slow for large N, but fine for small blocks.
    # Global arrays hack
    global _GLOBAL_V, _GLOBAL_P
    v_rows = _GLOBAL_V[row_idx]
    v_cols = _GLOBAL_V[col_idx]
    dot = v_rows @ v_cols.T
    
    # Add some spatial correlation based on X?
    # For the benchmark, let's just say K = V@V.T + diag(P) + random_structure
    # The 'residual' is just the 'random_structure'.
    # Let's make the residual correlated with spatial distance to test the "Static Tree" hypothesis.
    # Residual Covariance R_ij = exp(-dist(X_i, X_j)) * scale.
    # Then K_ij = V_i V_j^T + R_ij.
    
    global _GLOBAL_X
    x_rows = _GLOBAL_X[row_idx]
    x_cols = _GLOBAL_X[col_idx]
    
    # Simple RBF on X for residual
    # Ensure 2D
    if x_rows.ndim == 1: x_rows = x_rows[None, :]
    if x_cols.ndim == 1: x_cols = x_cols[None, :]
    
    diff = x_rows[:, None, :] - x_cols[None, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    residual_cov = 0.5 * np.exp(-dist_sq) # Strong spatial correlation
    
    return dot.astype(np.float64) + residual_cov

def run_benchmark():
    N = 10_000 # Small enough for brute force comparison
    D = 3
    K = 10
    
    print(f"Generating {N} points...")
    X, V, p_diag, kernel_diag = generate_synthetic_data(N, D)
    
    # Setup Global state for mock kernel
    global _GLOBAL_V, _GLOBAL_P, _GLOBAL_X
    _GLOBAL_V = V
    _GLOBAL_P = p_diag
    _GLOBAL_X = X
    
    # 1. Configure Backend
    host_data = ResidualCorrHostData(
        v_matrix=V,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=mock_kernel_provider,
        chunk_size=512
    )
    configure_residual_correlation(host_data)
    
    # 2. Build PCCT (Euclidean)
    print("Building PCCT (Euclidean)...")
    t0 = time.perf_counter()
    pcct_base = PCCT()
    tree = pcct_base.fit(X)
    t_build = time.perf_counter() - t0
    print(f"Build time: {t_build:.4f}s")
    
    # 3. Prepare Queries
    n_queries = 100
    query_indices = np.linspace(0, N-1, n_queries, dtype=np.int64).reshape(-1, 1)
    
    # 4. Enable Static Tree Mode

    print("Configuring Static Tree Mode...")
    import dataclasses
    from covertreex.api.runtime import Runtime
    
    config = RuntimeConfig.from_env()
    config = dataclasses.replace(config, residual_use_static_euclidean_tree=True)
    configure_runtime(config)
    
    # Update PCCT wrapper with the built tree AND the new runtime config
    runtime_wrapper = Runtime.from_config(config)
    pcct = dataclasses.replace(pcct_base, tree=tree, runtime=runtime_wrapper)
    
    # 5. Run Query (Static Tree)
    print(f"Running {n_queries} queries (k={K})...")
    t0 = time.perf_counter()
    indices, dists = pcct.knn(query_indices, k=K, return_distances=True)
    t_query = time.perf_counter() - t0
    print(f"Query time: {t_query:.4f}s ({n_queries/t_query:.2f} q/s)")
    
    # 6. Gold Standard (Brute Force)
    print("Running Brute Force Verification...")
    # We can use `compute_residual_pairwise_matrix` for a block?
    # Or manual scan.
    
    recall_accum = 0
    
    for i, q_idx in enumerate(query_indices):
        # Compute all distances
        # shape (1, N)
        q_arr = np.array([q_idx.item()], dtype=np.int64)
        all_indices = np.arange(N, dtype=np.int64)
        
        # Using internal helper to get true distances
        from covertreex.metrics.residual import compute_residual_distances
        true_dists = compute_residual_distances(host_data, q_arr, all_indices).flatten()
        
        # Sort
        top_k_idx = np.argsort(true_dists)[:K]
        # true_top_k_dists = true_dists[top_k_idx]
        
        # Compare with PCCT result
        pcct_idx = indices[i]
        
        intersection = np.intersect1d(top_k_idx, pcct_idx)
        recall = len(intersection) / K
        recall_accum += recall
        
    avg_recall = recall_accum / n_queries
    print(f"Average Recall@{K}: {avg_recall:.4f}")
    
    if avg_recall < 0.5:
        print("WARNING: Recall is low! Static Tree pruning/ordering might be ineffective.")
    else:
        print("SUCCESS: Recall is acceptable.")

if __name__ == "__main__":
    run_benchmark()
