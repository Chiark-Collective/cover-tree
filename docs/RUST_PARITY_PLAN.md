# Rust Residual Query Parity Plan (vs. Gold Python/Numba) — 2025-11-24

Goal: Close the ~130× query throughput gap to the gold Numba path on the 32k / d=3 / k=50 / 1,024q workload **without changing the gold configuration**. The gold script now hard-disables the Rust backend (`COVERTREEX_ENABLE_RUST=0`) to keep the reference immutable; Rust parity work must compare against that fixed Python/Numba baseline.

Status 2025-11-24 (evening):
- ✅ Rust residual heap now computes and uses a stored `si_cache` (cover radii) during traversal, matching the Python separation bounds.
- ✅ Added `COVERTREEX_RESIDUAL_PARITY=1` toggle that disables budgets/caps, dynamic tiling, and child reordering to mirror the Python gold traversal (stream_tile=1, raw si radii, no cap ladder).
- ⏳ Remaining gaps tracked below.

The gold Numba run (via `benchmarks/run_residual_gold_standard.sh`) has these defaults **active**:
- Level cache reuse (`residual_level_cache_batching=True`).
- Dynamic block sizing (`residual_dynamic_query_block=True`) tied to active queries.
- Budget ladder thresholds up=0.015, down=0.002.
- Radius floor = 1e-3.
- Masked dedupe/bitset in the scope streamer.
- Query telemetry emitted every run.
- Caps, gate/prefilter, SGEMM fallback: **off** by default (not set in the gold script).

## Gaps to close (Rust)
1) **Payload / precision parity**
   - Build residual trees with coordinate payloads (or an explicit idx↔coord map) and enable a float64 build path; gold is f64 while Rust still builds f32 index payloads.

2) **Ordering & visited semantics**
   - Match Python’s insertion-order child expansion and simple visited set; optionally bypass masked dedupe for parity.

3) **Stop rule & tiling**
   - Parity toggle should short-circuit when kth bound excludes the frontier; currently uses heap exhaustion only. Keep tile=1 (done) but add explicit kth/frontier stop.

4) **Caps/budgets fully off (done) but audit side effects**
   - Verify no cap_default leakage from metric; ensure scope caps are ignored in parity.

5) **Telemetry parity**
   - Emit/query counters matching Numba JSONL when parity mode is on (frontier sizes, prunes, eval counts, yields).

6) **Optional fast paths off**
   - Ensure SGEMM/block residual fast paths remain disabled under parity.

Suggested next steps
1) Add f64 build/residual metric path and idx→coord payload support; gate via parity flag.
2) Implement explicit child-order/visited set for parity (no sort/masking), plus kth/frontier early-stop.
3) Add telemetry shim keyed by parity mode to mirror Numba counters.
4) Re-run gold vs. parity-mode Rust on 32k/d=3/k=50 (1,024q) and record in benchmark audit.
