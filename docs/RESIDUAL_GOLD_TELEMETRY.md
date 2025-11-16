## Residual PCCT Gold Benchmark Telemetry (32 768 pts / 1 024 queries / k=8)

Source log: `artifacts/benchmarks/residual_dense_32768_dense_streamer_pairmerge_gold_cli_rerun3.jsonl` (run id `pcct-20251115-200110-15d295`). Config mirrors `benchmarks/run_residual_gold_standard.sh`: residual metric, dense traversal, natural batch order, diagnostics enabled, batch size 512, 64 batches, GPBoost baseline suppressed during telemetry.

### Implementation Priors

- **Traversal dominates** – `traverse_collect_scopes` (`covertreex/algo/traverse/runner.py:25`) feeds the residual streamer (`covertreex/algo/traverse/strategies/residual.py`), so residual pairwise kernels and semisort caching should consume ≥70 % of build wall clock.
- **Conflict graph still relevant** – `build_conflict_graph` (`covertreex/algo/conflict/runner.py:58`) handles scope aggregation and adjacency scattering; even with pairwise reuse, expected contribution was ~20 %.
- **MIS/persistence low impact** – `run_mis` (`covertreex/algo/mis.py:73`) plus copy-on-write persistence should be <10 % combined unless domination spikes.

### Observed Telemetry Breakdown

Per-batch telemetry exposes `*_ms` counters from `covertreex/telemetry/logs.py:126`. Summaries below aggregate 64 insert batches (values in milliseconds).

#### Top-Level Build Phases

| Stage | Total | Mean/batch | Median | P90 | Max | Build % |
| --- | --- | --- | --- | --- | --- | --- |
| Traversal total (`traversal_ms`) | 24 423.03 | 381.61 | 338.80 | 677.16 | 935.99 | 97.53 |
| Conflict graph total (`conflict_graph_ms`) | 603.49 | 9.43 | 7.92 | 15.42 | 21.00 | 2.41 |
| MIS (`mis_ms`) | 15.22 | 0.24 | 0.17 | 0.26 | 3.45 | 0.06 |
| **Build total** (`traversal+conflict+mis`) | **25 041.74** | **391.28** | **348.52** | **684.48** | **943.71** | **100.00** |

Traversal alone accounts for ~97.5 % of the build phase; the subsequent query run (not in this log) is ~0.23 s overall, so optimisation energy belongs squarely in the traversal kernel.

#### Traversal Subcomponents

| Traversal sub-stage | Total | Mean | Median | P90 | Max | Build % |
| --- | --- | --- | --- | --- | --- | --- |
| Residual pairwise kernel (`traversal_pairwise_ms`) | 20 164.71 | 315.07 | 274.46 | 586.88 | 860.39 | 80.52 |
| Semisort & scope dedupe (`traversal_semisort_ms`) | 2 887.93 | 45.12 | 46.25 | 62.29 | 67.41 | 11.53 |
| Kernel provider plumbing (`traversal_kernel_provider_ms`) | 1 667.94 | 26.06 | 26.64 | 54.83 | 86.29 | 6.66 |
| Whitened / mask / tile / gate counters | 0 | 0 | 0 | 0 | 0 | 0 |

`traversal_build_wall_ms` tracks closely with `traversal_ms` (24 400 ms total), reinforcing that the raw streamer dominates and batching overhead is negligible.

#### Conflict Graph Detail

| Conflict sub-stage | Total | Mean | Median | P90 | Max | Build % |
| --- | --- | --- | --- | --- | --- | --- |
| Adjacency scatter (`conflict_adjacency_ms`, incl. `conflict_adj_scatter_ms≈494 ms`) | 582.77 | 9.11 | 7.61 | 15.09 | 20.71 | 2.33 |
| Annulus pruning (`conflict_annulus_ms`) | 8.63 | 0.13 | 0.11 | 0.13 | 1.51 | 0.03 |
| Scope grouping (`conflict_scope_group_ms`) | 0.19 | 0.00 | 0.00 | 0.00 | 0.00 | ≈0 |
| Conflict pairwise refresh (`conflict_pairwise_ms`) | 0.31 | 0.00 | 0.00 | 0.01 | 0.01 | ≈0 |

Pairwise reuse is functioning perfectly: conflict pairwise costs are effectively zero because residual caches feed the builder. Most remaining time is the scatter kernel moving adjacency lists into arenas (`covertreex/algo/conflict/arena.py:103`).

### Interpretation vs. Priors

- **Traversal dominance confirmed and amplified** – Instead of the expected ~70 %, traversal consumes >97 % of build time; 80 % of the total goes solely to residual pairwise distances. Any regression or optimisation should focus on the streamer (e.g., smarter tile reuse, kernel fusion, gate activation).
- **Conflict graph negligible** – With pairmerge and dense streamer options on, conflict work is only 2–3 %. Further tuning there yields minimal speedup compared to streamer changes.
- **MIS/persistence are noise** – Even pathological batches top out at 3.45 ms of MIS time; the mean is sub-0.25 ms, confirming these paths are solved for the gold configuration.
- **Gate disabled** – `traversal_gate1_ms` and related counters stay at zero, meaning the gold benchmark still runs without residual gate pruning. Enabling the gate (with lookup tables from `covertreex/metrics/residual/policy.py:62`) is the most direct lever to cut candidate counts before the heavy pairwise work.

### Actionable Notes

1. **Optimise the residual kernel** – Consider reducing tile sizes, batching kernel provider calls, or precomputing whitened embeddings to drive down the 20 s spent in `traversal_pairwise_ms`.
2. **Explore gate activation** – With zero runtime allocated to gate logic, enabling the lookup (and ensuring caps/whitened coverage stay acceptable) could trim the worst-case 935 ms batches dramatically by shrinking candidate scopes.
3. **Conflict/MIS tweaks are low ROI** – Scattering is already just ~2 % of the budget; further engineering there will not materially change the gold benchmark.

Keep this file alongside `benchmarks/run_residual_gold_standard.sh` and the referenced JSONL log so regressions or improvements can be benchmarked against a shared, quantitative breakdown.
