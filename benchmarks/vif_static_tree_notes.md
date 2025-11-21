# Benchmark Analysis

## Correlation
*   **Pearson:** 0.58
*   **Spearman:** 0.66
This indicates a moderate positive correlation between Euclidean distance and Residual distance.

## Performance (Static Tree + Dynamic Query)
*   **Python Baseline:** ~40 q/s (implied from previous error trace).
*   **Numba Port:** **115.53 q/s**.
    *   Speedup: ~2.8x.
    *   This is lower than expected (target >1000 q/s). The overhead might be in the "Best First Search" loop logic or the heap operations in Numba, or simply the cost of `K(q, c)` which is an exponential in Python (inside `mock_kernel_provider`)?
    *   Wait, the Numba path uses `_compute_residual_dist_rbf_batch` which is **PURE NUMBA**. It does NOT call the Python `mock_kernel_provider`.
    *   Ah, but `generate_synthetic_data` sets up the data such that `K_true = exp(-d^2)`.
    *   And I injected `rbf_lengthscale = 0.707` (which means `exp(-0.5 * d^2 / 0.5) = exp(-d^2)`).
    *   So the Numba path should be calculating the correct kernel values (mostly).

## Recall Issue
*   **Average Recall@10: 0.0010**.
*   This is catastrophic.
*   The correlation is decent (0.66), so ordering should be okay-ish.
*   Why is recall near zero?
    *   **Hypothesis 1:** The "Residual Math" in Numba (`_compute_residual_dist_rbf_batch`) is inconsistent with `compute_residual_distances` (Python/KernelProvider) used for ground truth.
    *   **Hypothesis 2:** The "Identity Mapping" or Index Mapping is wrong.
    *   **Hypothesis 3:** The Numba Heap logic is buggy (e.g. popping max instead of min?). `_pop_min_heap` looks correct.
    *   **Hypothesis 4:** `_update_knn_sorted` logic is buggy.

Let's debug Hypothesis 1.
The ground truth uses `mock_kernel_provider`.
The Numba path uses `_compute_residual_dist_rbf_batch`.
`mock_kernel_provider` returns `exp(-d^2)`.
`_compute_residual_dist_rbf_batch` calculates `k_val = var * exp(-0.5 * d^2 / ls^2)`.
I set `var=1.0`, `ls=0.707`. `ls^2 = 0.5`.
`k_val = exp(-0.5 * d^2 / 0.5) = exp(-d^2)`.
This matches.

However, `_compute_residual_dist_rbf_batch` computes:
`rho = (k_val - dot) / denom`.
`dot` is `v_q . v_c`.
`denom` is `sqrt(p_q * p_c)`.

In `mock_kernel_provider`, I implemented:
`return dot + residual_cov`.
Wait, `mock_kernel_provider` returns the **FULL Kernel** $K_{prior}$.
And `residual_cov` was defined as `0.5 * exp(-d^2)`.
So `K_prior = V.V' + 0.5 * exp(-d^2)`.

In Numba RBF path, I calculate `k_val` using RBF parameters.
If I want `k_val` to match `K_prior`, I must set RBF params such that `RBF approx V.V' + 0.5 * exp(-d^2)`.
This is impossible because `V.V'` is low rank and not translation invariant.

The Numba path implemented assumes **K_prior IS the RBF**.
The benchmark implemented **K_prior = V.V' + RBF**.

The VIF method usually assumes we have a K_prior (e.g. Matern), and we approximate it with `V.V'`.
So `K_prior approx V.V'`.
The "Residual" is `K_prior - V.V'`.
So `rho = (K_prior - V.V') / ...`

In the benchmark `mock_kernel_provider`, I returned `dot + residual_cov`.
This implies `K_prior - V.V' = residual_cov`.
And `residual_cov = 0.5 * exp(-d^2)`.

So in the benchmark, the **Residual Kernel** is `0.5 * exp(-d^2)`.
The Numba path computes `rho = (RBF - V.V') / ...`.
If I pass `var=1.0`, `ls=0.707`, then `RBF = exp(-d^2)`.
So Numba computes `rho = (exp(-d^2) - V.V') / ...`.

Mismatch:
Benchmark Truth: `rho = 0.5 * exp(-d^2) / ...`
Numba Path: `rho = (exp(-d^2) - V.V') / ...`

These are totally different.
The Numba path assumes `K_prior` is the stationary RBF.
The benchmark constructed a world where `K_prior = V.V' + Noise`.

To align them, I must set the benchmark such that `K_prior` **IS** RBF.
And `V` is an approximation of `K_prior`.
Then `K_prior - V.V'` is the residual.

So I need to fix `generate_synthetic_data` again.
1. Generate X.
2. Define K_prior = RBF(1.0, 0.707).
3. Generate V such that V.V' approx K_prior. (Nystrom or just random small V).
4. Update `mock_kernel_provider` to return `K_prior` (RBF).

Then:
Truth: `compute_residual_distances` uses `mock_kernel_provider` (RBF) - `V.V'`.
Numba: uses RBF - `V.V'`.
They should match.

Correction Plan:
1. Update `generate_synthetic_data` to make `K_prior` purely RBF.
2. Update `mock_kernel_provider` to return purely RBF.
