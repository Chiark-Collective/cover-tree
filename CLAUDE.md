# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Covertreex is a high-performance cover tree library for k-NN queries, optimized for Vecchia-style Gaussian process pipelines. It features a hybrid Python/Numba + Rust implementation.

**Primary use case**: Library integration into GP pipelines (e.g., survi-v2) for efficient neighbor selection with residual correlation metric.

## Library API

The main user-facing API:

```python
from covertreex import CoverTree, Runtime

# Basic Euclidean k-NN
tree = CoverTree().fit(points)
neighbors = tree.knn(query_points, k=10)

# Fast Rust backend
runtime = Runtime(engine="rust-hilbert")
tree = CoverTree(runtime).fit(points)

# Residual correlation metric
from covertreex.metrics.residual import build_residual_backend, configure_residual_correlation

backend = build_residual_backend(coords, seed=42, inducing_count=512)
runtime = Runtime(metric="residual_correlation", engine="rust-hilbert")
ctx = runtime.activate()
configure_residual_correlation(backend, context=ctx)
tree = CoverTree(runtime).fit(indices)
neighbors = tree.knn(indices, k=50)
```

Key classes:
- `CoverTree` — Main interface for tree building and k-NN queries
- `Runtime` — Configuration for backend, metric, and engine selection
- `build_residual_backend` — Creates V-matrix backend for residual metric

## Build & Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Build Rust backend (requires Rust toolchain)
maturin develop --release

# Run all tests
pytest

# Run a single test
pytest tests/test_knn.py::test_knn_basic -v

# Run tests matching a pattern
pytest -k "residual" -v

# Lint and format
ruff check covertreex cli tests tools
ruff format covertreex cli tests tools

# Type checking
mypy covertreex
```

## CLI (Testing/Benchmarking)

The CLI is primarily for testing and benchmarking, not the main interface:

```bash
python -m cli.pcct --help
python -m cli.pcct query --engine rust-hilbert --tree-points 32768 --k 50
python -m cli.pcct doctor --profile default
```

## Canonical Benchmarks

Use these scripts for performance validation—do not create ad-hoc benchmark scripts:

1. **Gold Standard**: `./benchmarks/run_residual_gold_standard.sh [log_path]` — validates N=32k, D=3 residual performance
2. **Regression Suite**: `python tools/run_reference_benchmarks.py` — comprehensive CI checks
3. **Rust vs Python**: `python benchmarks/comprehensive_residual_benchmark.py`

## Architecture

### Module Structure

- `covertreex/` — core library
  - `api/` — public façade (`CoverTree`, `Runtime`, `Residual`)
  - `engine.py` — engine implementations (python-numba, rust-hilbert, etc.)
  - `algo/` — algorithms: batch insert, MIS, conflict graphs, traversal strategies
  - `metrics/` — distance metrics including residual correlation
  - `queries/` — k-NN query implementations
  - `runtime/` — configuration model and context management
  - `telemetry/` — JSONL logging and schema definitions
- `cli/pcct/` — Typer-based CLI (for testing)
- `src/` — Rust backend (`covertreex_backend`)
- `profiles/` — YAML runtime presets (default, residual-gold, residual-fast, etc.)

### Key Design Patterns

**Runtime Context**: All configuration flows through `Runtime.activate()` which returns a `RuntimeContext`. Low-level functions accept `context=` keyword argument.

```python
from covertreex import CoverTree, Runtime

runtime = Runtime(metric="euclidean", engine="rust-hilbert")
tree = CoverTree(runtime).fit(points)
```

**Immutable Trees**: Operations return new tree instances; trees are never mutated in-place.

**Engines**: Three execution engines:
- `python-numba` — reference implementation with full telemetry
- `rust-natural` — Rust backend, natural point order
- `rust-hilbert` — Rust backend with Hilbert curve reordering (fastest)

### Rust Backend

Located in `src/`, built via maturin. Enabled by default if compiled; disable with `COVERTREEX_ENABLE_RUST=0`.

Key files:
- `src/lib.rs` — PyO3 bindings
- `src/tree.rs` — tree data structure
- `src/algo.rs`, `src/algo/batch/` — batch insert and MIS algorithms
- `src/metric.rs` — distance computations (AVX2 SIMD optimized)

## Documentation Maintenance

### CHANGELOG.md
Update `CHANGELOG.md` for any user-visible changes following [Keep a Changelog](https://keepachangelog.com/) format.

### BENCHMARKS.md
Update `BENCHMARKS.md` when recording significant benchmark results:
- Include exact command for reproducibility
- Note commit hash and hardware specs
- Run at least 3 times and report median

## PyPI Publishing

```bash
# Build with maturin (creates wheel with Rust extension)
maturin build --release

# Upload to PyPI
source .pypi_token.env
TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_API_TOKEN" twine upload target/wheels/*.whl
```

## Agent Guidelines

- Prefer additive optimizations initially; remove superseded code only after proving superiority
- Update `BENCHMARKS.md` when benchmark results change
- Update `CHANGELOG.md` for user-visible changes
- Do not add ad-hoc benchmark scripts without consultation
