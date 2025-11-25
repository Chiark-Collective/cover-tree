# Residual Roadmap — 2025-11-24 (WIP)

We’re at a fork: finish correctness parity, then chase performance along two axes (Numba + Rust). This doc tracks three epics to keep scope clear and benchmarkable.

---

## Epic 1 — Parity Lock & Cross-Engine Correctness
**Goal:** Full behavioral parity between Python/Numba “gold” and Rust residual traversal (parity mode), with deterministic equality tests and matching telemetry payloads.

- **Scope**
  - Add deterministic cross-engine fixture (1k pts, 64 queries, k=1..10, mixed dims/seeds) asserting identical neighbors/distances (f64 parity) and stable ordering.
  - Telemetry parity: ensure Rust `rust_query_telemetry` fields mirror Numba JSONL (frontier, prunes, evals, yields, block sizes).
  - Duplicate/visited semantics: explicit visited set in parity path; coord payload option if needed.
  - CI hook: parity fixture test gated in CI (fast) + optional long-run benchmark gate.
- **Deliverables**
  - Tests under `tests/` (python + rust) with fixed seeds.
  - Telemetry schema note and sample payload in docs.
  - Benchmark log pair showing parity run (gold vs rust-hilbert parity) with commands.
  - Preset toggle `COVERTREEX_PRESET=residual_parity` that applies the parity-safe defaults (f64, fast paths off, static tree, telemetry on) without stacking many individual env vars.
  - CLI support: `--preset residual_parity` on `cli.pcct query` and `PRESET=residual_parity ./benchmarks/run_residual_gold_standard.sh`.
- **Checkpoints**
  - ✅ Parity env toggle in place; telemetry emitted for f32/f64.
  - 🔲 Equality fixture merged & green in CI.
  - 🔲 Telemetry field-by-field match documented.
  - 🔲 Parity benchmark rerun after fixes; audit updated.

## Epic 2 — Portable Heuristics Back to Numba
**Goal:** Lift gold baseline by porting proven Rust heuristics into Numba behind flags, keeping outputs unchanged.

- **Scope**
  - Dynamic block sizing tied to frontier/active set.
  - Survivor budget ladder + low-yield early-stop (with k-safe guard).
  - Optional child reordering (stable) to tighten kth sooner.
  - Keep masked dedupe/level-cache reuse; no cap default changes.
  - Benchmark via `run_residual_gold_standard.sh` with opt-in flags; document commands/results.
- **Deliverables**
  - Flags in Numba traversal + defaults off.
  - Bench table comparing gold baseline vs each heuristic combo.
  - Notes on output invariance (neighbor sets unchanged).
- **Checkpoints**
  - 🔲 Flags implemented and unit-tested for k-safety.
  - 🔲 Benchmark deltas recorded; winners proposed for default.
- ⚠️ Regression to chase: 2025-11-25 gold run (Python/Numba) collapsed to ~3 q/s (telemetry wall ≈5.6 s → ~182 q/s) on the 32k residual workload while Rust perf preset stayed fast. Needs bisect and fix before lifting heuristics.
- ⚠️ Update 2025-11-25 (afternoon): raising the SGEMM brute-force cap (default 50M pairs, parity 30M) and keeping the fast path on by default lifts the gold run to **~874 q/s** (1.17 s query on 32k / 1,024q). Still ~47× shy of the historical ~41k q/s baseline; remaining gap is tree-walk cost vs. dense SGEMM.

## Epic 3 — Rust Perf Mode After Parity
**Goal:** Recover and surpass prior Rust perf while keeping parity mode as correctness guard.

- **Scope**
  - Re-enable fast paths (SIMD/tiled, SGEMM) and pruning bounds; tune stream tiling and cap/budget ladders.
  - Child ordering heuristics, masked dedupe, dynamic tiles, survivor budgets: measure selectively.
  - Build-time targets: match/better best-known Rust build; Query targets: approach gold QPS.
  - Telemetry-driven profiling to cut distance evals/heap pushes.
- **Deliverables**
  - “perf” preset/toggle distinct from parity.
    - `COVERTREEX_PRESET=residual_perf` (or `--preset residual_perf` in CLI) turns parity off, flips fast paths on (f32), and leaves static-tree disabled.
  - Benchmark logs (best-of-5) against gold, with commands.
  - Updated audit noting perf + correctness safeguards.
- **Checkpoints**
  - ✅ Perf preset defined; parity left untouched.
  - 🔲 Distance evals/heap pushes reduced vs current parity run.
  - 🔲 Perf benchmark beats previous Rust best; documented.

### Preset quick reference

- `residual_parity`: metric=residual_correlation, precision=float64, parity flag on, fast paths off, static Euclidean tree on, rust telemetry on.
- `residual_perf`: metric=residual_correlation, precision=float32, parity flag off, fast paths on, static tree off, Rust enabled.

---

## Operating Rules (for all epics)
- Keep `run_residual_gold_standard.sh` as the primary reproducible benchmark; include commands next to results.
- No ad-hoc benchmark scripts; add scenarios to reference suite if new coverage is needed.
- Changes that alter outputs must first be proven behind flags; only flip defaults after equality tests pass.
- Update benchmark docs when publishing new results; note command + env vars.
