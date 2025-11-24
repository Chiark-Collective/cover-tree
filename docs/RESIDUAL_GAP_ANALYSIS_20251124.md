# Residual Parity Gap (Rust Hilbert vs Python/Numba) — 2025-11-24

## Snapshot
- **python-numba (gold)**: build 7.104 s, query 0.0247 s (~41,519 q/s)  
  Command: `ENGINE=python-numba ./benchmarks/run_residual_gold_standard.sh bench_residual_python_numba_rerun.log`
- **rust-hilbert**: build 2.509 s, query 3.195 s (~320 q/s)  
  Command: `ENGINE=rust-hilbert COVERTREEX_ENABLE_RUST=1 COVERTREEX_RUST_PCCT2_SGEMM=auto COVERTREEX_BATCH_ORDER=natural ./benchmarks/run_residual_gold_standard.sh bench_residual_rust_hilbert.log`

## Key Gaps
1. **Scope streamer parity**  
   - No bitset/masked append or stream_tile batching; no level cache reuse.  
   - Budget ladder (32/64/96) not applied during streaming; scan caps not enforced per query.

2. **Pruning / traversal**  
   - Tree traversal lacks scope-driven candidate pruning and dynamic block sizing; radius_floor only partially enforced.  
   - Query still falls back to broad scans (SGEMM) rather than staged pruning.

3. **Gate & caps**  
   - Gate/profile lookup not loaded; gate telemetry absent.  
   - Scope caps table (per-level caps) not consumed in Rust path.

4. **Conflict builder efficiency**  
   - No pair-merge/buffer-reuse/max-segment controls; chunk_max_segments ignored.  
   - Degree cap optional but other heuristics missing.

5. **Telemetry completeness**  
   - Missing scope cache hits/prefetch, gate fields, cap application counters; buffer reuse stats absent.

## Next Actions (to close the gap)
1. Implement dense scope streamer in Rust (bitset/masked append, stream_tile=64, budget ladder, cache).  
2. Apply radius_floor and caps in traversal; gate/profile loading even if default-off.  
3. Add conflict buffer reuse + pair-merge respecting chunk/max-segment knobs.  
4. Re-run gold benchmarks and update BENCHMARK_AUDIT with new results/commands.
