# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Covertreex is a Parallel Compressed Cover Tree (PCCT) library engineered for Vecchia-style Gaussian process pipelines. It features a hybrid Python/Numba + Rust implementation optimized for CPU workloads.

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

## CLI Usage

The primary CLI is `python -m cli.pcct`:

```bash
# List available profiles
python -m cli.pcct profile list

# Run k-NN query benchmark
python -m cli.pcct query --dimension 8 --tree-points 8192 --queries 512 --k 8

# Build tree only
python -m cli.pcct build --dimension 8 --tree-points 65536 --batch-size 1024

# Environment/dependency check
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
  - `api/` — public façade (`PCCT`, `Runtime`, `Residual`)
  - `algo/` — algorithms: batch insert, MIS, conflict graphs, traversal strategies
  - `metrics/` — distance metrics including residual correlation
  - `queries/` — k-NN query implementations
  - `runtime/` — configuration model and context management
  - `telemetry/` — JSONL logging and schema definitions
- `cli/pcct/` — Typer-based CLI application
- `src/` — Rust backend (`covertreex_backend`)
- `profiles/` — YAML runtime presets (default, residual-gold, residual-fast, etc.)

### Key Design Patterns

**Runtime Context**: All configuration flows through `Runtime.activate()` which returns a `RuntimeContext`. Low-level functions accept `context=` keyword argument and never read global state.

```python
from covertreex.api import PCCT, Runtime

runtime = Runtime(metric="euclidean", enable_numba=True)
with runtime.activate() as context:
    tree = PCCT(runtime).fit(points)
```

**Immutable Trees**: `PCCT` operations return new `PCCTree` instances; trees are never mutated in-place.

**Strategy Registries**: Traversal and conflict-graph strategies are registered via `covertreex.algo.traverse` and `covertreex.algo.conflict` modules. Third-party plugins can register via setuptools entry points.

**Engines**: Three execution engines available via `--engine`:
- `python-numba` — reference implementation with full telemetry
- `rust-natural` — Rust backend, natural point order
- `rust-hilbert` — Rust backend with Hilbert curve reordering (fastest builds)

### Rust Backend

Located in `src/`, built via maturin. Enabled by default if compiled; disable with `COVERTREEX_ENABLE_RUST=0`.

Key files:
- `src/lib.rs` — PyO3 bindings
- `src/tree.rs` — tree data structure
- `src/algo.rs`, `src/algo/batch/` — batch insert and MIS algorithms
- `src/metric.rs` — distance computations

## Documentation Maintenance

### CHANGELOG.md
Update `CHANGELOG.md` for any user-visible changes following [Keep a Changelog](https://keepachangelog.com/) format:
- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for bug fixes
- **Security** for vulnerability fixes

### BENCHMARKS.md
Update `BENCHMARKS.md` when recording significant/novel benchmark results:
- Include exact CLI command or script for reproducibility
- Note commit hash and hardware specs
- Run at least 3 times and report median
- Update in the same commit as performance-affecting changes

## Test History Tracking (pytest-chronicle)

Test results are automatically tracked via pytest-chronicle. The `.pytest-chronicle.toml` config enables auto-ingestion.

```bash
# View test timeline across commits
pytest-chronicle query timeline

# Find tests that are currently failing
pytest-chronicle query last-red

# Find when a test last passed
pytest-chronicle query last-green

# Compare test status between branches
pytest-chronicle query compare main feature-branch
```

Results are stored in `.pytest-chronicle/chronicle.db` (SQLite). The database and config are gitignored.

## PyPI Publishing

```bash
# Build with maturin (creates wheel with Rust extension)
maturin build --release

# Upload to PyPI (source .pypi_token.env for PYPI_API_TOKEN)
source .pypi_token.env
TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_API_TOKEN" twine upload target/wheels/*.whl
```

## Agent Guidelines

Per `AGENTS.md`:
- Prefer additive optimizations initially; remove superseded code only after proving superiority
- Update `BENCHMARKS.md` when benchmark results change, including reproducible commands
- Update `CHANGELOG.md` for user-visible changes
- Do not add ad-hoc benchmark scripts without consultation
- Treat repository as R&D workspace; deleting dead code is fine if not the only working implementation
