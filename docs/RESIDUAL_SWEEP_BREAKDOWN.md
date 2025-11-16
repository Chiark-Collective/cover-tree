# Residual Gold Sweep (dims 2–3, k ∈ {15,25,50}, N ∈ {8k,16k,32k})

Script: `tools/run_residual_gold_sweep.py` (best-of-7 per configuration, seeds 42–48). Command matches the gold residual spec (`python -m cli.queries --metric residual --baseline gpboost --batch-size 512 --queries 1024`) with the natural batch order and dense traversal knobs:

```
COVERTREEX_ENABLE_NUMBA=1
COVERTREEX_SCOPE_CHUNK_TARGET=0
COVERTREEX_ENABLE_SPARSE_TRAVERSAL=0
COVERTREEX_BATCH_ORDER=natural
COVERTREEX_PREFIX_SCHEDULE=doubling
COVERTREEX_ENABLE_DIAGNOSTICS=0
```

Telemetry: JSONL logs live under `artifacts/benchmarks/residual_gold_sweep/`. The summary (`summary.json`) captures the best run (lowest total build time) per configuration, along with per-stage timing statistics aggregated over the 64 insert batches.

All numbers below use the best-of-7 run for each `(dimension, tree_points, k)` tuple; “build” is total insert time (s), percentages are relative to build time.

### Tree points = 8192

| dim | k | best build (s) | traversal % | pairwise % | semisort % | conflict % | best log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 15 | 2.421 | 93.4% | 67.8% | 16.2% | 6.33% | `artifacts/benchmarks/residual_gold_sweep/gold_dim2_k15_n8192_run5.jsonl` |
| 2 | 25 | 2.362 | 93.5% | 68.4% | 15.3% | 6.26% | `artifacts/benchmarks/residual_gold_sweep/gold_dim2_k25_n8192_run4.jsonl` |
| 2 | 50 | 2.365 | 93.8% | 68.3% | 15.5% | 5.94% | `artifacts/benchmarks/residual_gold_sweep/gold_dim2_k50_n8192_run6.jsonl` |
| 3 | 15 | 2.404 | 92.8% | 66.8% | 16.7% | 6.75% | `artifacts/benchmarks/residual_gold_sweep/gold_dim3_k15_n8192_run2.jsonl` |
| 3 | 25 | 2.616 | 94.5% | 69.0% | 16.3% | 5.22% | `artifacts/benchmarks/residual_gold_sweep/gold_dim3_k25_n8192_run6.jsonl` |
| 3 | 50 | 2.567 | 93.5% | 66.4% | 17.4% | 6.29% | `artifacts/benchmarks/residual_gold_sweep/gold_dim3_k50_n8192_run6.jsonl` |

### Tree points = 16384

| dim | k | best build (s) | traversal % | pairwise % | semisort % | conflict % | best log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 15 | 12.984 | 97.8% | 81.9% | 9.5% | 2.12% | `artifacts/benchmarks/residual_gold_sweep/gold_dim2_k15_n16384_run6.jsonl` |
| 2 | 25 | 13.009 | 97.9% | 82.0% | 9.4% | 2.05% | `artifacts/benchmarks/residual_gold_sweep/gold_dim2_k25_n16384_run2.jsonl` |
| 2 | 50 | 12.746 | 97.6% | 81.9% | 9.5% | 2.24% | `artifacts/benchmarks/residual_gold_sweep/gold_dim2_k50_n16384_run5.jsonl` |
| 3 | 15 | 13.488 | 97.5% | 81.5% | 9.6% | 2.38% | `artifacts/benchmarks/residual_gold_sweep/gold_dim3_k15_n16384_run3.jsonl` |
| 3 | 25 | 13.051 | 97.9% | 81.3% | 10.0% | 2.07% | `artifacts/benchmarks/residual_gold_sweep/gold_dim3_k25_n16384_run1.jsonl` |
| 3 | 50 | 13.399 | 97.6% | 81.5% | 9.7% | 2.34% | `artifacts/benchmarks/residual_gold_sweep/gold_dim3_k50_n16384_run5.jsonl` |

### Tree points = 32768

| dim | k | best build (s) | traversal % | pairwise % | semisort % | conflict % | best log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 15 | 14.303 | 95.9% | 74.4% | 16.1% | 3.98% | `artifacts/benchmarks/residual_gold_sweep/gold_dim2_k15_n32768_run6.jsonl` |
| 2 | 25 | 13.485 | 95.2% | 74.0% | 15.9% | 4.66% | `artifacts/benchmarks/residual_gold_sweep/gold_dim2_k25_n32768_run1.jsonl` |
| 2 | 50 | 13.849 | 94.7% | 73.9% | 15.5% | 5.21% | `artifacts/benchmarks/residual_gold_sweep/gold_dim2_k50_n32768_run2.jsonl` |
| 3 | 15 | 13.752 | 94.4% | 71.8% | 17.0% | 5.44% | `artifacts/benchmarks/residual_gold_sweep/gold_dim3_k15_n32768_run3.jsonl` |
| 3 | 25 | 13.309 | 94.7% | 72.1% | 16.9% | 5.21% | `artifacts/benchmarks/residual_gold_sweep/gold_dim3_k25_n32768_run3.jsonl` |
| 3 | 50 | 15.223 | 95.4% | 75.9% | 14.1% | 4.55% | `artifacts/benchmarks/residual_gold_sweep/gold_dim3_k50_n32768_run4.jsonl` |

### Observations

- Traversal continues to dominate every configuration (>93 % of build time at 8k, >97 % at 16k, and ≈95 % at 32k). Pairwise residual kernels alone claim 66–82 % of the build budget regardless of dimensionality.
- Increasing dimensionality from 2→3 barely shifts wall-clock at 8k/16k but does raise semisort costs slightly at 32k due to residual scope fan-out.
- Higher `k` values (25/50) are effectively cost-neutral across these ranges because batch sizes stay fixed at 512; candidate counts remain dominated by metric geometry rather than output width.
- Conflict graph work is capped at 2–7 % depending on tree size, confirming earlier findings that dense traversal is the clear target for optimisation even under lower-dimensional workloads.

### 64k follow-up (dim = 3, k = 50)

To probe larger scales we captured a single gold-spec run at 65 536 points (`artifacts/benchmarks/residual_gold_sweep/gold_dim3_k50_n65536_run1.jsonl`). It requires 128 insert batches (still 512 points each) and takes **51.13 s** to build, with traversal accounting for **97.1 %** of wall time (pairwise kernels **80.6 %**, semisort **11.9 %**).

Why the jump compared to the ~13–15 s totals at 16k/32k?

- 16k uses 32 batches, 32k uses 64, and both quickly hit the “dominated” fast path—after the first few prefixes almost every batch reduces to 3–5 survivors, so per-batch costs stay around 0.2–0.25 s and doubling the tree size barely changes total time.
- At 64k the residual streamer has to chew through a lot more novel neighbourhoods before cache hits dominate. Per-batch pairwise time jumps from ~180 ms (mean) at 32k to ~306 ms (mean) at 64k, with late-stage batches regularly exceeding 0.6 s because scope caches keep filling with new parent groups.
- Even though average survivors per batch remain tiny (≈7 selected, 509 dominated), the high point density in ℝ³ forces the streamer to process more tiles per batch, so the total build grows superlinearly once we cross ~32k.

The takeaway is that the “flat” scaling between 16k and 32k is a side effect of dominance and caching; once the cache is stressed (64k+), traversal costs per batch rise again and the residual kernel becomes the bottleneck all over. Any attempt to push beyond 32k should factor in smarter gating or batch scheduling, otherwise runtime will climb sharply.

### Low/Medium-Hanging Optimisation Ideas

These runs still leave a few obvious levers. Most revolve around shrinking candidate counts or accelerating the residual kernel, since `traversal_pairwise_ms` consistently eats 70–80 % of wall time.

1. **Re-enable the residual gate** – All gold logs show `traversal_gate1_ms=0`, meaning the gate is disabled. Use the CLI knobs (`--residual-gate lookup`, `--residual-gate-lookup-path docs/data/residual_gate_profile_32768_caps.json`, `--residual-gate-cap …`, etc.) or set `residual=ApiResidual(...)` in code so `covertreex/metrics/residual/policy.py` prunes candidates before the heavy kernel. Even modest pruning (5–10 %) translates directly into proportional savings on the 20 s spent in pairwise kernels.
2. **Try the sparse streamer** – The dense traversal path keeps entire scopes hot. Enabling sparse traversal (`COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1`, `scope_chunk_target` ~8 192, `scope_budget_schedule` tuned) trades extra bookkeeping for fewer pairwise evaluations. Combining sparse traversal with the gate is the single biggest architectural lever; prior `maskopt_v2` runs in `artifacts/benchmarks/residual_dense_32768_dense_streamer_maskappend_on_run*.jsonl` show clear reductions once chunking is enabled.
3. **Accelerate the residual kernel provider** – `covertreex/metrics/residual/host_backend.py` currently does SGEMM/Triangular solves on the CPU. Options if you can throw hardware at it:
   - Precompute/cache larger kernel blocks (more `kernel_provider` hits) by increasing memory budgets or duplicating work across threads.
   - Implement `TreeBackend.gpu` (`covertreex/core/tree.py:59`) so the kernel tiles run on GPU (CuBLAS) instead of the host.
   - Increase `residual_stream_tile` so each tile amortises more work per kernel call.
4. **Alter batch ordering/prefix schedules** – Natural order is reproducible but not optimal. Switching to Hilbert with `prefix_schedule=adaptive` (see `covertreex/algo/order/helpers.py`) can reduce early-batch variance and deliver higher dominance sooner, shaving the first few seconds off large builds.
5. **Ensure Numba kernels are warm** – Residual append routines in `covertreex/algo/_residual_scope_numba.py` and scope builders in `_scope_numba.py` only help once the JIT fires. Set `COVERTREEX_ENABLE_NUMBA=1` (already on) and call `warmup_scope_builder()` at startup so the first batches don’t pay compilation penalties. Also verify `residual_masked_scope_append` and `residual_scope_bitset` are exercising the compiled paths.
6. **Offload telemetry/persistence I/O** – `BenchmarkLogWriter` writes synchronously after each batch. Buffering writes or moving telemetry to a separate process won’t change the big picture but can claw back a few hundred milliseconds over 64 batches, especially on slower disks.

All of these focus on the traversal kernel because conflict/MIS/logging remain rounding errors (<5 %). The quickest wins should come from reducing the number of candidates the residual streamer touches (gate, sparse traversal) or speeding up the SGEMM-heavy `kernel_provider` path via better hardware utilisation.
