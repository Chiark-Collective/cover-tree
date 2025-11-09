You’ve already done the hard part—queries are blazing. Your logs make it pretty clear where the build time is going and what to change. Below I’ll (1) pinpoint the bottlenecks you’re seeing, (2) give *switch‑only* fixes you can try immediately, and (3) propose small, targeted code changes that remove the dominant costs without touching query speed or correctness.

---

## 1) What the logs say (tl;dr)

* **Traversal dominates build.**
  In the unclamped residual run (no scope chunking), dominated batches spend **~60–145 ms** in `traversal_ms` with tiny `conflict_graph_ms` and negligible `mis_ms`. In the “sparse traversal + scan cap” run, `traversal_semisort_ms` explodes to **1.4 s → 5.8 s per dominated batch**, with `traversal_scope_chunk_scans` ramping up **512 → 2 048** and `traversal_scope_chunk_points` **262 144 → 1 048 576**. That’s your 727 s build.

* **Conflict graph isn’t your problem (right now).**
  `conflict_pairwise_ms` is ~12–15 ms and `mis_ms` ~0.14–0.33 ms in the unclamped run. In the chunked run you *also* re‑compute pairwise inside the conflict builder even though traversal already computed/cached it.

* **Residual grid path is effectively off.**
  `runtime_conflict_graph=grid` is set, but `conflict_grid_*` counters are all zero in residual logs, so you aren’t getting the “forced leaders” skip that removed MIS edges in Euclidean.

* **Domination is extreme.**
  `conflict_scope_domination_ratio ≈ 0.002` and `selected=1` (511 dominated) across batches. This is the regime where (a) leader pre‑selection pays off massively, and (b) expensive scope “semisort” is pure overhead.

---

## 2) Fast wins you can try **now** (no code changes)

These do not change results; they just avoid paths that your telemetry shows as pathological.

1. **Turn OFF sparse traversal & chunking for residual until we fix semisort.**
   Your own run shows `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1` + chunking multiplies build time (~727 s). Keep the residual build on the dense path for now:

   ```bash
   COVERTREEX_ENABLE_SPARSE_TRAVERSAL=0
   COVERTREEX_SCOPE_CHUNK_TARGET=0
   ```

   *Expected:* build stays in the ~58–72 s band (your unclamped numbers) instead of minutes.

2. **Enable Gate‑1 for residual *during traversal*.**
   Gate‑1 counters are zero in your logs, so the float32 whitened gate is disabled. Turn it on and keep audit on while you tune:

   ```bash
   COVERTREEX_RESIDUAL_GATE1=1
   COVERTREEX_RESIDUAL_GATE1_AUDIT=1   # temporarily, to prove safety
   ```

   If you’ve generated a lookup profile (you mentioned `tools/build_residual_gate_profile.py`), use it:

   ```bash
   COVERTREEX_RESIDUAL_GATE1_LOOKUP_PATH=docs/data/residual_gate_profile_diag0.json
   ```

   *Expected:* `traversal_pairwise_ms` and `traversal_ms` drop, with `traversal_gate1_*` counters showing large “pruned”.

3. **Warm up Numba kernels once** before your real build.
   You already document this; just make sure your CI/runner actually does it so the “first build” doesn’t contaminate your measurements.

4. **Prefer `doubling` over `adaptive` for residual until chunking is fixed.**
   Adaptive is great for Euclidean, but your own note says the residual variant “exceeds 20 minutes under this schedule.” Try:

   ```bash
   COVERTREEX_PREFIX_SCHEDULE=doubling
   ```

   *Expected:* fewer giant super‑scopes showing up during dominated prefixes.

> If you only do (1) + (2), you should already see a meaningful drop versus the 57–72 s steady‑state residual build.

---

## 3) Small code changes that remove the worst costs

All changes below are narrow and localized; they don’t alter algorithmic results.

### A) **Stop recomputing pairwise distances in the conflict builder**

You already compute residual pairwise rows in traversal; the dense builder still spends ~12–15 ms per dominated batch in `conflict_pairwise_ms`. Thread the traversal cache through `conflict_graph.py`.

**Before** (`covertreex/algo/conflict_graph.py` — excerpt you showed):

```python
adjacency_build = _build_dense_adjacency(
    backend=backend,
    batch_size=batch_size,
    scope_indptr=scope_indptr,
    scope_indices=scope_indices,
)
```

**After** (pass the cached arrays; fall back to None if not residual):

```python
pairwise = getattr(traversal, "residual_pairwise", None)
radii   = getattr(traversal, "residual_radii", None)

adjacency_build = _build_dense_adjacency(
    backend=backend,
    batch_size=batch_size,
    scope_indptr=scope_indptr,
    scope_indices=scope_indices,
    pairwise=pairwise,
    radii=radii,
)
```

And in `_build_dense_adjacency(...)` forward `pairwise`/`radii` to `build_conflict_graph_numba_dense(...)`. You already wrote the builder to accept them; this just wires it up.

*Expected telemetry deltas:* `conflict_pairwise_ms → ~0`, `conflict_adjacency_ms` unchanged, `conflict_graph_ms` drops by ~12–15 ms per dominated batch.

---

### B) **Kill `traversal_semisort_ms` with bucketed CSR (no O(n log n) sorts)**

The “sparse traversal” path’s semisort is your 1.4–5.8 s per batch monster. Replace it with a *two‑pass, counting/bucketing CSR* builder (exactly the pattern you already use in the conflict builder with `_group_by_key_counting` → prefix sum → fill). The idea is:

1. Count how many memberships go to each owner/node (`counts[node] += 1`).
2. Prefix‑sum to `indptr`.
3. Fill `indices[indptr[node] + cursor[node]] = member`.

This is linear, parallelizable, and needs **no sorting**.

**New kernel (place alongside your other scope Numba helpers):**

```python
# covertreex/algo/_scope_numba.py
@nb.njit(cache=True, parallel=True)
def build_scope_csr_from_pairs(
    owners: np.ndarray,   # I32, length M
    members: np.ndarray,  # I32, length M
    num_nodes: int,       # batch_size or max(owner)+1
) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(num_nodes, dtype=I64)
    for i in nb.prange(owners.size):
        counts[int(owners[i])] += 1

    indptr = np.empty(num_nodes + 1, dtype=I64)
    acc = I64(0)
    indptr[0] = 0
    for n in range(num_nodes):
        acc += counts[n]
        indptr[n + 1] = acc

    indices = np.empty(acc, dtype=I32)
    # thread-local cursors per node
    cursors = np.zeros(num_nodes, dtype=I64)

    for i in nb.prange(owners.size):
        node = int(owners[i])
        pos  = indptr[node] + cursors[node]
        indices[pos] = members[i]
        cursors[node] += 1

    return indptr, indices
```

**Use it in the sparse traversal path** instead of semisort. You likely have `(point_ids, owner_ids)` already; feed those in directly. Keep your existing dedupe step (if required) by hashing segments *after* CSR creation (or set a `segment_dedupe=1` flag and reuse your `_hash_segments/_dedupe_segments_by_hash`).

*Expected telemetry deltas (sparse traversal only):* `traversal_semisort_ms → ~O(10–50 ms)` per dominated batch instead of seconds, `traversal_scope_chunk_dedupe` unchanged or smaller, same correctness.

---

### C) **Make the residual “grid” actually select leaders (forced masks)**

Your residual runs show `conflict_grid_* = 0`, i.e., the grid path is bypassed. You can get the same gains you saw in Euclidean by running a *whitened‑space* grid that maps residual radius to an angular/dot threshold and emits `forced_selected/forced_dominated` masks.

Minimal integration plan (no full rewrite):

1. In `configure_residual_correlation`, you already precompute `gate_v32` (whitened vectors) and `gate_norm32`. Normalize once so `||z||≈1`.

2. Add a residual‑aware grid builder that takes:

   * `Z = gate_v32` (float32, unit-ish norm)
   * a per‑batch **correlation threshold** `t = 1 - r^2` (from your residual metric: `dist = sqrt(max(0, 1 - |rho|))` ⇒ `|rho| ≥ 1 - r^2`).
   * a cell width derived from `t` (coarser for small radii). Start pragmatic: quantize each dimension with `scale = ceil(1/(1 - t + 1e-4))` and hash to 64‑bit keys (you already have mixed‑priority hashing).

3. Reuse your existing `grid_select_leaders_numba(...)` mechanics but on `Z` and with that `scale`. Emit `forced_selected`/`forced_dominated` and skip adjacency creation when the dominated ratio is ≪ 1 (your case).

*Expected telemetry:* non‑zero `conflict_grid_*`, `conflict_adj_pairs=0`, `mis_iterations=1`, `conflict_graph_ms` collapses to single‑digit ms *and* fewer dominated nodes reach traversal in the next prefixes.

> This is deliberately approximate but **safe**: the grid only *pre‑colours*; MIS still runs and your residual distance check in adjacency remains the correctness gate. Start with diagnostics on and verify `forced_*` counts line up with your domination ratios.

---

### D) **Adaptive *pair‑cap* chunking (prevents the 512→2048 scan explosion)**

Chunking should be driven by a **pair budget**, not a fixed member cap. Your metrics already compute `candidate_pairs` and segment counts—use them.

Heuristic (works well in practice):

```python
# choose at most ~130k directed edges per dominated batch (what your
# good Euclidean run emitted)
PAIR_BUDGET = int(128_000)

# Given expected mean group size g (from previous batches) and #groups G,
# choose a chunk_target so G * g*(g-1) ~ PAIR_BUDGET
# (approximate; clamp to [8192, 262144])
chunk_target = clamp(int(sqrt(PAIR_BUDGET / max(G, 1))), 8192, 262144)
```

Wire this into your traversal/conflict builder *when* `domination_ratio < 0.01`. The result is ~constant `conflict_scope_chunk_pairs_after` and a bounded `traversal_scope_chunk_segments` (no more 512→2048 growth).

*Expected telemetry:* stable `traversal_scope_chunk_segments` (dozens, not thousands), `traversal_semisort_ms` and `conflict_adjacency_ms` flatten across batches.

---

### E) **Two‑phase residual distance kernel (cheap early rejection)**

Your `_distance_chunk` already carries a “tail” bound, but you still traverse all D dims for a large share. With small `D=8`, a two‑phase split is still worth it because early exit prevents the `sqrt` + branches per dim loop:

```python
@njit(cache=True, fastmath=True, parallel=True)
def _distance_chunk_two_phase(...):
    # Phase 1: first 4 dims, quick bound check
    D1 = 4
    for j in prange(v_chunk.shape[0]):
        denom = ...
        partial = 0.0; qi_sq = 0.0; qj_sq = 0.0
        pruned = False
        for d in range(D1):
            ...  # accumulate partial, qi_sq, qj_sq
            tail = ...
            if can_prune_with_threshold(...):   # same logic as today
                distances[j] = radius + eps
                within[j] = 0
                pruned = True
                break
        if pruned: continue

        # Phase 2: remaining dims
        for d in range(D1, v_query.shape[0]):
            ...  # identical

        rho = ...
        ...
```

*Expected:* fewer full‑dim iterations in dominated regions. This pays off even more once Gate‑1 is enabled (Phase 1 prunes aggressively).

---

### F) **Minor but free micro‑optimizations**

* Preallocate and reuse `distances`/`within` buffers in the residual kernel (thread‑local scratch) to reduce allocator pressure in big batches.
* Ensure all arrays passed to Numba kernels are **C‑contiguous** (`np.ascontiguousarray`) *before* the jitted call (you already do this in places).
* For residual only, consider running the kernel in float32 internally (keep inputs as float64, compute partials in float32, cast back for the final `rho`). This halves memory traffic and may shave ~10–20% in traversal; keep `COVERTREEX_RESIDUAL_GATE1_AUDIT=1` while evaluating.

---

## 4) What to watch in telemetry (to confirm each change worked)

* **Pairwise reuse (A):** `conflict_pairwise_ms → ~0` while `conflict_adj_pairs` stays constant.
* **Bucketed CSR (B):** `traversal_semisort_ms` drops from seconds to <~50 ms; `traversal_scope_chunk_scans` no longer ramps up; RSS spikes disappear.
* **Residual grid (C):** `conflict_grid_cells > 0`, large `conflict_grid_leaders_raw/after`, `conflict_adj_pairs=0` on dominated batches, `mis_iterations=1`.
* **Pair‑cap chunking (D):** `conflict_scope_chunk_pairs_after ≈ 130,816` (your healthy dominated batches), `traversal_scope_chunk_segments` roughly stable.
* **Gate‑1 (switch‑only):** `traversal_gate1_pruned` large, `traversal_ms` shrinks, audit shows no violations.

---

## 5) A minimal “try this” recipe

Start with the *switches only* and the *pairwise reuse* patch:

```bash
# Switches first
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_ENABLE_DIAGNOSTICS=1 \
COVERTREEX_ENABLE_SPARSE_TRAVERSAL=0 \
COVERTREEX_SCOPE_CHUNK_TARGET=0 \
COVERTREEX_BATCH_ORDER=hilbert \
COVERTREEX_PREFIX_SCHEDULE=doubling \
COVERTREEX_RESIDUAL_GATE1=1 \
COVERTREEX_RESIDUAL_GATE1_AUDIT=1 \
COVERTREEX_RESIDUAL_GATE1_LOOKUP_PATH=docs/data/residual_gate_profile_diag0.json \
python -m cli.queries \
  --metric residual \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline gpboost
```

Then apply patch **A** (thread pairwise from traversal to conflict builder) and rerun. Confirm `conflict_pairwise_ms` ~ 0. If you need the patch text in a PR‑ready diff, I can write it out exactly against your `conflict_graph.py` and any helper signatures you expose.

Next, if you want the big decrease that gets you near Euclidean‑like build times on residual, implement **B** (bucketed CSR) and **C** (residual grid pre‑selection). Those two remove the pathological semisort and eliminate most adjacency work on dominated batches, which is exactly the shape your logs show.

---

## 6) Why these changes (short rationale)

* Your builds spend time where there’s almost no *information gain*: extremely high domination and repeated membership rewriting. That’s why skipping sort (B), reusing pairwise (A), and pre‑selecting leaders (C) give outsized wins without affecting correctness.
* Gate‑1 is designed precisely to reduce residual chunk work; it’s off in your logs. Turn it on and let audit prove safety as you tune thresholds/lookup.
* Chunking by a *pair budget* keeps worst‑case memory and time bounded. Fixed member caps do not.

---

If you want, I can supply exact diffs for (A) and (B) first (they’re small and self‑contained), and a stub for (C) that plugs into your existing grid hashing.
