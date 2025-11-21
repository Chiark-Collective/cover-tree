# Residual Engine Refactor Plan

Goal: unify residual tree builds behind a single cover-tree abstraction with pluggable engines, defaulting to Python/Numba and offering a fast Rust residual-only engine. Avoid “PCCT” naming in user-facing surfaces.

## Naming & Config
- Introduce `engine` enum in `RuntimeConfig`/CLI: `python-numba` (default), `rust-fast` (residual-only), later `rust-hybrid`.
- Deprecate “PCCT” in help/docs; describe engines as “Python/Numba” vs “Rust fast residual”.
- Add CLI `--engine` flag; default stays `python-numba`.

## Engine Interface
- Define `TreeEngine` protocol/class with:
  - `build(points, metric, runtime) -> CoverTreeHandle`
  - `knn(handle, queries, k, runtime)`
  - Optional `meta()` for stats.
- Engines:
  - `PythonNumbaEngine`: wraps current PCCTree builder/query (Hilbert/conflict graph/telemetry).
  - `RustFastResidualEngine`: wraps `build_fast_residual_tree` (1-D indices, residual-only), dispatches `knn_query_residual` with cached mapping/backend.

## Cover Tree Wrapper
- Add `CoverTree` wrapper storing `engine` + `handle`, exposing `knn(...)` delegating to engine.
- `covertreex.queries.knn` checks for `CoverTree` and routes via engine; legacy PCCTree paths remain for compatibility.

## Builder Plumbing
- Expose `build_tree(points, *, metric, engine, runtime)` selecting engine and returning `CoverTree`.
- CLI `query` uses `build_tree` instead of PCCT-specific helpers.
- Validation: `engine=rust-fast` requires `metric == residual_correlation`, residual backend present, integer payloads; telemetry/log_writer only on `python-numba`.

## Docs & UX
- Update CLI help/examples to use `--engine rust-fast`.
- Document feature trade-offs: `python-numba` = telemetry/gating/Hilbert, slower build; `rust-fast` = residual-only, fastest build, no telemetry.
- Keep PCCT names internally for now; add deprecation note in user-facing text.

## Tests/Compat
- Add smoke test: build with `engine=rust-fast` and run residual `knn` on indices.
- Ensure existing tests keep passing; legacy entry points still accept PCCTree.

## Follow-up (optional)
- Later rename internal `pcct` modules once the wrapper/engine path is stable, or alias them behind new names to avoid churn.***
