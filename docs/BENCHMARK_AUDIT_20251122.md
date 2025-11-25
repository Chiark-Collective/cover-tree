# Benchmark Suite Audit (2025-11-22)

## Executive Summary

The project currently relies on two "canonical" sources of truth for benchmarking and several ad-hoc developer scripts. Confusion regarding performance results often stems from the fact that the automated regression suite **does not** run the "Gold Standard" configuration used for historical claims.

1.  **Historical "Gold Standard"**: Defined in `benchmarks/run_residual_gold_standard.sh`.
2.  **Automated Regression**: Defined in `tools/run_reference_benchmarks.py` (runs different, smaller jobs).
3.  **Developer/Ad-Hoc**: Scripts like `rust_full_residual_benchmark.py` that test the Rust backend directly, often with different architectural assumptions (indices vs coordinates).

---

## 1. The "Gold Standard" (Historical Results)
**File:** `benchmarks/run_residual_gold_standard.sh`

This script is the definitive source for the **24.20s build / 0.046s query** result reported on 2025-11-17. To reproduce historical numbers, this script must be used.

*   **Entry Point:** `python -m cli.pcct query`
*   **Configuration:**
    *   **N (Points)**: 32,768
    *   **D (Dimension)**: 3
    *   **Metric**: Residual
    *   **Batch Order**: Natural
    *   **Chunking/Sparse**: DISABLED (Explicitly unsets sparse traversal and chunking).
*   **Key Insight:** The script now hard-disables the Rust backend (`COVERTREEX_ENABLE_RUST=0`) to guarantee the Python/Numba traversal for the gold run. A separate optional comparison run can be enabled via `COMP_ENGINE` (default: `rust-hilbert`).

## 2. Automated Regression Suite
**File:** `tools/run_reference_benchmarks.py`

This tool is designed for CI/CD or nightly checks to ensure feature stability. **It does NOT run the 32k Gold Standard workload.**

*   **Key Jobs:**
    *   `queries_2048_*`: Quick smoke tests (2k points).
    *   `queries_8192_*`: Medium scaling tests (8k points).
    *   `queries_32768_euclidean_hilbert_grid`: Tests Euclidean metric with advanced build options (Hilbert ordering + Grid conflict graph).
    *   `queries_32768_residual_dense_pairmerge`: Tests Residual metric with "dense pair-merge" streaming.
    *   `queries_32768_residual_perf_rust_hilbert`: Smoke check for the Rust perf preset (f32, fast paths on, rust-hilbert engine, k=50) to catch perf regressions post-parity.
*   **Purpose:** Checks for regressions in specific subsystems (diagnostics, conflict graph implementations, streaming logic) rather than tracking peak performance of the standard path.

## 3. The Rust Discrepancy (Indices vs Coordinates)
**Files:** `benchmarks/rust_full_residual_benchmark.py`, `benchmarks/rust_knn_benchmark.py`

These ad-hoc scripts reveal a critical architectural divergence between the Python and Rust backends for the Residual metric:

*   **Python/Numba Path:** Builds the tree on **D-dimensional coordinates** (floats).
*   **Rust Residual Path:** Builds the tree on **1-dimensional indices** (integers stored as floats).
    *   The Rust wrapper (`CoverTreeWrapper`) for residual mode expects the primary tree data to be `(N, 1)` arrays of indices.
    *   The actual coordinates (`X`), V-matrix, and other metric data are passed as *separate arguments* to `insert_residual` and `knn_query_residual`.

**Impact:** Toggling `enable_rust=True` in a script designed for the Python path (which passes coordinates) will result in crashes or incorrect behavior, as the Rust backend attempts to interpret coordinates as indices. The main CLI (`cli.pcct`) handles this abstraction, but raw benchmark scripts must handle this data transformation manually.

## 4. Recommendations

1.  **Use `run_residual_gold_standard.sh`** for apples-to-apples historical comparisons.
2.  **Use `tools/run_reference_benchmarks.py`** for validating stability and correctness across different features.
3.  **Treat `rust_*_benchmark.py` scripts as low-level driver tests** for backend development, not system-level benchmarks.

---

## 5. Latest runs (2025-11-24)
Commands (32,768 points, d=3, 1,024 queries, k=50):
```
# Gold run (Python/Numba enforced)
./benchmarks/run_residual_gold_standard.sh bench_residual_gold.log

# Optional comparison (default rust-hilbert); skip with COMP_ENGINE=none
COMP_ENGINE=rust-hilbert ./benchmarks/run_residual_gold_standard.sh bench_residual_gold.log
```
Latest local results (2025-11-24):
- python-numba (gold): build **7.38 s**, query **0.0250 s** (~41,0k q/s).
- rust-hilbert (comparison): build **3.34 s**, query **0.164 s** (~6.26k q/s).
- gpboost baseline (same runs): build ~1.42 s, query ~3.59 s (~278 q/s).

Note: rust-hilbert is the current default comparison engine; adjust `COMP_ENGINE` to test alternatives or set `COMP_ENGINE=none` to suppress the comparison pass.

### 2025-11-24 (evening) parity-mode reruns (best-of-5)
Command (runs executed 5×):
```
# Gold (Python/Numba enforced)
./benchmarks/run_residual_gold_standard.sh bench_residual_si_rerun_<n>.log

# Comparison (rust-hilbert with stored si_cache + parity toggle available)
COMP_ENGINE=rust-hilbert ./benchmarks/run_residual_gold_standard.sh bench_residual_si_rerun_<n>.log
```
Results (q/s):
- Gold mean **41.48k** (stdev **1.26k**), best **43.25k**.
- Rust-hilbert mean **9.15k** (stdev **2.77k**), best **13.70k**.
Artifacts: `bench_residual_si_rerun_{1..5}.log`, `bench_residual_si_rerun_{1..5}_rust-hilbert.log` in repo root.

Single-run parity toggle sanity (2025-11-24 late):
```
COVERTREEX_RESIDUAL_PARITY=1 ./benchmarks/run_residual_gold_standard.sh bench_residual_parity_rust.log
```
- Gold: **41,364 q/s** (build 7.25 s, query 0.0248 s).
- Rust-hilbert (parity mode: si_cache, no budgets/caps/reordering, stream_tile=1, f64 build): **5,127 q/s** (build 3.49 s, query 0.1997 s).
Artifacts: `bench_residual_parity_rust.log`, `bench_residual_parity_rust_rust-hilbert.log`.

### 2025-11-25 — residual_perf comparison smoke
Command (gold path Python/Numba, comparison uses perf preset on rust-hilbert):
```
COMP_PRESET=residual_perf ./benchmarks/run_residual_gold_standard.sh bench_residual_perf_baseline.log
```
Results (32,768 / d=3 / 1,024 queries / k=50):
- python-numba (gold, preset unset, Rust forced off): build **8.11 s**, query **346.01 s** → **3.0 q/s**. Telemetry sums to ~**5.6 s wall** (~182 q/s), so the summary line is misreporting; either way this is a major regression vs the 2025-11-24 gold (~41k q/s) and needs investigation.
- rust-hilbert (`COMP_PRESET=residual_perf`): build **3.33 s**, query **0.096 s** → **10,627 q/s**; gpboost baseline from the same run: **263 q/s**, so PCCT is ~40× faster.
Artifacts: `bench_residual_perf_baseline.log`, `bench_residual_perf_baseline_rust-hilbert.log`.

### 2025-11-25 — SGEMM brute-force guard + parity cap
- Added a configurable guard for the residual SGEMM brute-force path: `COVERTREEX_RESIDUAL_BRUTE_FORCE_MAX_PAIRS` (defaults to **50M** pairs, parity **30M**; set ≤0 to disable). This keeps the gold 32k workload on the SGEMM fast path while avoiding O(N·Q) blowups on larger jobs.
- Command (gold only, comparison disabled):  
  ```
  COMP_ENGINE=none ./benchmarks/run_residual_gold_standard.sh bench_residual_gold_rerun_fix2.log
  ```
- Result (python-numba enforced, 32,768 / d=3 / 1,024q / k=50): **build 8.09 s**, **query 1.17 s** → **874 q/s**. This clears the prior 3–4 q/s regression but is still ~47× below the historical ~41k q/s gold.
- Logs: `bench_residual_gold_rerun_fix2.log` (gold only), telemetry JSONL at `artifacts/benchmarks/queries_pcct-20251125-112045-2c9c68_20251125-112045.jsonl`.
