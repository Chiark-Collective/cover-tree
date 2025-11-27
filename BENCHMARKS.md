# Benchmarks

Historical benchmark results for covertreex. Each entry includes the exact command for reproducibility.

## Gold Standard: Residual Metric (N=32k, D=3, k=50)

The primary optimization target for Vecchia-style GP workloads.

### 2025-11-26 (Rust Backend)

| Engine | Build (s) | Query (s) | Total (s) |
|--------|-----------|-----------|-----------|
| python-numba | 24.2 | 0.046 | 24.25 |
| rust-natural | 6.1 | 0.12 | 6.22 |
| rust-hilbert | 4.8 | 0.15 | 4.95 |

```bash
# Python/Numba baseline
./benchmarks/run_residual_gold_standard.sh

# Rust natural order
python -m cli.pcct query --engine rust-natural --metric residual \
  --tree-points 32768 --dimension 3 --queries 1024 --k 50

# Rust Hilbert order (fastest build)
python -m cli.pcct query --engine rust-hilbert --metric residual \
  --tree-points 32768 --dimension 3 --queries 1024 --k 50
```

## Euclidean Metric Scaling

### 2025-11-20 (N scaling, D=8, k=8)

| N | Build (s) | Query (s) | Throughput (q/s) |
|---|-----------|-----------|------------------|
| 8192 | 0.8 | 0.02 | 25600 |
| 32768 | 3.2 | 0.08 | 6400 |
| 131072 | 14.1 | 0.35 | 1463 |

```bash
python -m cli.pcct query --metric euclidean --dimension 8 \
  --tree-points 8192 --queries 512 --k 8
```

---

## Adding New Benchmarks

When recording a new benchmark:
1. Include the exact CLI command or script used
2. Note the commit hash and hardware (CPU model, RAM)
3. Run at least 3 times and report median
4. Update this file in the same commit as any performance-affecting changes
