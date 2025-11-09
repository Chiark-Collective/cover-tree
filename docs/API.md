# Covertreex API Overview

The `covertreex.api` module exposes a small, typed façade that keeps experiments out of the internal
modules. Treat it as the authoritative way to configure runtimes, stage residual policies, and drive
the public PCCT interface.

## Runtime configuration

```python
from covertreex.api import Runtime, Residual

cli_runtime = Runtime(
    metric="euclidean",
    backend="numpy",
    precision="float64",
    enable_numba=True,
    diagnostics=False,
)
cli_runtime.activate()
```

`Runtime.activate()` installs the configuration globally (matching what the CLI does) without
writing to `os.environ`. Residual settings are aggregated through the `Residual` helper so callers
can declaratively override gate/look-up behaviour:

```python
residual_runtime = Runtime(
    metric="residual",
    backend="numpy",
    residual=Residual(
        gate1_enabled=True,
        lookup_path="docs/data/residual_gate_profile_diag0.json",
        scope_cap_path="docs/data/residual_scope_caps_32768.json",
    ),
)
residual_runtime.activate()
```

## PCCT façade

The `covertreex.api.PCCT` façade wraps the immutable tree plus helper operations. Typical usage:

```python
from covertreex.api import PCCT
import numpy as np

runtime = Runtime(metric="euclidean", enable_numba=True)
runtime.activate()

points = np.random.default_rng(0).normal(size=(2048, 8))
queries = np.random.default_rng(1).normal(size=(512, 8))

pcct = PCCT.empty(dimension=points.shape[1])
pcct = pcct.insert(points)
indices, distances = pcct.knn(queries, k=8)
```

Batch inserts accept both NumPy and JAX arrays depending on the runtime backend. Conflicts,
traversal, and MIS selection now run through strategy registries (`covertreex.algo.traverse` and
`covertreex.algo.conflict`) so custom strategies can be registered by calling
`register_traversal_strategy(...)` or `register_conflict_strategy(...)` during startup.

## Command-line entrypoints

All benchmark commands now live under `cli/` with compatibility shims in `benchmarks/`:

- `python -m cli.queries …` replaces `benchmarks.queries`. The module handles dataset generation,
  runtime activation, telemetry, and (optionally) baseline comparisons.
- `python -m cli.runtime_breakdown …` captures per-phase timings, CSV summaries, and plots.

You can continue invoking `python -m benchmarks.queries` while downstream tooling migrates, but the
`cli.*` modules are the supported entrypoints going forward.
