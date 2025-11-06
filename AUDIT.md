Below is a focused, concrete optimisation plan to drive **Numba PCCT build time** down further. I’ve organised it by *where the time still goes* (based on your latest logs) and then by **high‑leverage changes** with code‑level sketches. Everything stays exact/correct (no ANN shortcuts) and preserves your immutability/persistence contract.

---

## Where the cost still hides (from your latest snapshots)

* At **small/medium n** (2 k–8 k), you’ve crushed conflict‑graph/MIS costs; build wall time is now dominated by traversal + persistence book‑keeping.
* At **large n (32 k)**:

  * CPU utilisation during **build** ~**1.2×** ⇒ most work is still **serial** (traversal/persistence).
  * Conflict-graph CSR is essentially free (sub‑ms); MIS ~sub‑ms; the remaining wall time comes from:

    1. traversal mask/distances that didn’t yet migrate to fully parallel kernels in all paths, and
    2. **copy‑on‑write (CoW) persistence updates** (level-offset recompute, children/next splicing, cache maintenance) that still do significant Python/NumPy work and repeated allocations.

The fastest route to another 2–4× reduction is therefore: **(A)** eliminate residual serial traversal and **(B)** collapse CoW/persistence writes into one parallel Numba “apply‑updates” sweep with pooled scratch memory.

---

## A. Traversal: finish the migration & add early-exit distance kernels

You already ported scope assembly to Numba and saw big wins. There are two more practical gains:

### A1) Bound‑aware Euclidean distances with early termination (tight inner loop)

When forming masks/radii and during redistribution, you’re still doing full L2 on many pairs. Use a **squared distance accumulator** with **early abort** against the per‑pair bound (min of node radii). This speeds dominated/near‑dominated cases dramatically.

```python
# covertreex/algo/_dist_numba.py
import numba as nb, numpy as np

@nb.njit(cache=True, fastmath=True, nogil=True, inline='always')
def _sqdist_leq_bound(x: np.ndarray, y: np.ndarray, bound2: float) -> (float, b1):
    acc = 0.0
    # unroll by 4 if dims are small/fixed
    for t in range(x.size):
        d = y[t] - x[t]
        acc += d * d
        if acc > bound2:  # early exit
            return acc, False
    return acc, True
```

Integrate this into:

* traversal pair checks (mask formation), and
* redistribution where you compute `ceil_log2(dist)` (you can return `acc` directly; avoid `sqrt` until the final `ceil_log2` calculation).

### A2) Fast `ceil_log2` without `log`/`sqrt`

Replace `ceil_log2(dist)` with a bit‑trick on squared distance to avoid `sqrt() + log2()`:

```python
@nb.njit(cache=True, inline='always', fastmath=True)
def ceil_log2_from_sqdist(acc_sq: float) -> int:
    # dist = sqrt(acc_sq);  ceil(log2(dist)) == ceil(0.5 * log2(acc_sq))
    # use frexp: acc_sq == m * 2**e, where m in [0.5, 1)
    m, e = np.frexp(acc_sq)    # e is exponent s.t. acc_sq = m * 2**e
    # log2(dist) = 0.5 * (e + log2(m)) ; with m in [0.5,1) => log2(m) in [-1,0)
    # Conservatively ceil by using only exponent e:
    #   ceil(0.5*e + delta) with delta in [-0.5,0)
    # Upper bound (safe): ceil(0.5*e - 0.5) == ((e-1)+1)//2  == (e)//2
    # To keep separation conservative, use:
    return (e + 1) // 2  # tight and monotone for our purposes
```

This keeps redistribution integer‑only and branch‑free.

---

## B. Persistence & updates: one parallel sweep + pooled scratch

Most of your 32 k build wall time is now hidden in **copy‑on‑write fragmentation** and Python‑side array surgery. Fix that with an **“overlay then freeze”** application pattern:

### B1) Transaction overlay (journal) inside the batch → single CoW sweep

* **Today:** path‑copy at multiple points in the batch (levels, children/next, offsets), causing repeated slices/allocations.
* **Proposal:** During `batch_insert`, accumulate all mutations into a **journal** (SoA arrays of equal length):

  * `journal_parent[i] = new_parent_id or -1`
  * `journal_level[i] = new_level or -1`
  * `journal_child_head_updates[parent] = new_head` (record only the *new* head)
  * `journal_next[child] = next_id` (for chain splices)
  * `journal_level_counts_delta[level] += δ`
* Then call a **single Numba kernel** to apply the journal to the current snapshot using **CoW at segment granularity** (copy each touched segment *once*).

Sketch:

```python
# covertreex/core/_persistence_numba.py
@nb.njit(cache=True, nogil=True, parallel=True)
def apply_journal_cow(
    parent: np.ndarray, level: np.ndarray,
    child_head: np.ndarray, next_sib: np.ndarray,
    level_counts: np.ndarray,
    # journal buffers (same length = num_mutations)
    j_nodes: np.ndarray, j_parent: np.ndarray, j_level: np.ndarray,
    j_head_parents: np.ndarray, j_head_values: np.ndarray,
    j_next_nodes: np.ndarray, j_next_values: np.ndarray,
    j_level_delta_levels: np.ndarray, j_level_delta_vals: np.ndarray,
    # out: cloned arrays (preallocated or alias input for in-place-with-copy)
    parent_out: np.ndarray, level_out: np.ndarray,
    child_head_out: np.ndarray, next_sib_out: np.ndarray,
    level_counts_out: np.ndarray,
):
    # 1) bulk copy untouched arrays (parallel)
    n = parent.size
    for i in nb.prange(n):
        parent_out[i] = parent[i]
        level_out[i] = level[i]
        next_sib_out[i] = next_sib[i]
    for i in nb.prange(child_head.size):
        child_head_out[i] = child_head[i]
    for i in range(level_counts.size):
        level_counts_out[i] = level_counts[i]

    # 2) apply node parent/level updates (parallel)
    for k in nb.prange(j_nodes.size):
        u = j_nodes[k]
        if j_parent[k] >= 0:
            parent_out[u] = j_parent[k]
        if j_level[k]  >= 0:
            level_out[u]  = j_level[k]

    # 3) apply head updates (parents potentially repeat; last one wins)
    for k in nb.prange(j_head_parents.size):
        p = j_head_parents[k]
        child_head_out[p] = j_head_values[k]

    # 4) apply next-sibling splices
    for k in nb.prange(j_next_nodes.size):
        u = j_next_nodes[k]
        next_sib_out[u] = j_next_values[k]

    # 5) apply level count deltas
    for k in range(j_level_delta_levels.size):
        lvl = j_level_delta_levels[k]
        level_counts_out[lvl] += j_level_delta_vals[k]
```

**Effect:** immutability maintained, but **only one CoW** per batch, with parallel copies. For 32 k this eliminates a large fraction of Python overhead and small allocations.

> Implementation detail: keep these `*_out` buffers **preallocated** (see B2) and swap references at the end of the batch.

### B2) Scratch/pool the big temporaries inside Numba

Your scope/adjacency builder already prewarms JIT; do the same for the large CoW scratch:

* Maintain a **module‑local pool** (capacity grows geometrically) for:

  * `sources/targets` (when you need them)
  * `csr_indptr/indices` (if you need host copies)
  * `*_out` arrays used by `apply_journal_cow`
* Provide a tiny **ensure_capacity** kernel:

```python
# covertreex/algo/_pool_numba.py
POOL_PARENT = np.empty(1, dtype=np.int32)  # grows on demand
# ... same for others

@nb.njit(cache=True)
def ensure_capacity(arr: np.ndarray, new_n: int) -> np.ndarray:
    if arr.size >= new_n:
        return arr
    m = max(new_n, arr.size * 2)
    out = np.empty(m, arr.dtype)
    out[:arr.size] = arr
    return out
```

Use the pool across batches to avoid repeated `np.empty`/GC churn and paging spikes (your logs show 230–950 MB spikes at 32 k without pooling).

### B3) Bulk children/next splicing

You already switched to “insert at head and chain old head behind”. Make it **bulk** with a single pass:

Input:

* `anchors[]` (newly selected MIS nodes)
* `old_head[parent]` (snapshot)
* `attach_lists[parent]` (dominated inserts that must go under `parent`)

Single pass:

```python
@nb.njit(cache=True, nogil=True, parallel=True)
def bulk_splice_heads(
    child_head: np.ndarray, next_sib: np.ndarray,
    anchors: np.ndarray, parents_of_anchor: np.ndarray,
    dominated_nodes: np.ndarray, dominated_parent: np.ndarray,
    # journal outputs (parallel writable)
    j_head_parents: np.ndarray, j_head_values: np.ndarray,
    j_next_nodes: np.ndarray, j_next_values: np.ndarray
):
    # anchors become new heads (parent->anchor)
    for i in nb.prange(anchors.size):
        p = parents_of_anchor[i]
        a = anchors[i]
        j_head_parents[i] = p
        j_head_values[i]  = a
        j_next_nodes[i]   = a
        j_next_values[i]  = child_head[p]  # old head behind
    # dominated nodes are spliced to their chosen parents in a second block
    base = anchors.size
    for i in nb.prange(dominated_nodes.size):
        u = dominated_nodes[i]
        p = dominated_parent[i]
        # typical policy: insert dominated as head too
        j_head_parents[base + i] = p
        j_head_values[base + i]  = u
        j_next_nodes[base + i]   = u
        j_next_values[base + i]  = child_head[p]
```

You then feed these two blocks straight into the **journal** and let `apply_journal_cow` do one CoW.

### B4) Level offsets / counts: incremental, not recompute

Stop recomputing level offsets from scratch. Track:

* `level_counts[]` and mutate via `j_level_delta_*`.
* Derive offsets once per batch via a **parallel prefix sum** (Numba):

```python
@nb.njit(cache=True)
def prefix_sum_inplace(a: np.ndarray):  # int64 OK
    acc = 0
    for i in range(a.size):
        acc += a[i]
        a[i] = acc
```

At 32 k this avoids O(L) Python loops per micro‑update.

---

## C. Parallelism & scheduling hygiene (make your prange actually scale)

### C1) Avoid thread oversubscription

When NumPy BLAS kicks in (sometimes via incidental ops), it can fight Numba’s pool. Set these in the benchmark harness and docs:

```
# One worker pool at a time
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Let Numba own parallelism; tune to sockets/LLC
export NUMBA_NUM_THREADS=$(nproc)          # or a measured sweet spot
export NUMBA_THREADING_LAYER=tbb           # ‘tbb’ often wins over ‘omp’
```

Your utilisation tables already show under‑parallelisation at 32 k; this fixes a common cause.

### C2) Use `schedule='dynamic'` for skewed groups

Where group sizes vary (dominated vs mixed batches), use:

```python
@nb.njit(cache=True, parallel=True)
def kernel(...):
    for g in nb.prange(num_groups, schedule='dynamic'):
        ...
```

This mitigates long‑tail groups stalling the pool.

---

## D. Data layout & cache locality

### D1) 32‑bit indices end‑to‑end

You already use `I32/I64`. Push **I32** everywhere feasible (node ids, parents, heads, next), **I64 only for indptr**. This shaves bandwidth and improves cache residency in the big 32 k runs.

### D2) Level‑major node ordering (periodic renumbering)

After large builds (e.g., every 8–16 prefix batches), **relabel nodes by (level, parent order)**. This:

* makes per‑level sweeps contiguous,
* reduces TLB misses in traversal and persistence,
* and makes `child_head/next_sib` chains touch fewer cache lines.

Do it via a “renumbering journal”: build a permutation `perm[]` and apply it once with `apply_journal_cow`. Queries don’t care about ids as long as you rewrite `parents/children/next` consistently.

---

## E. Conflict graph: keep it sub‑ms at scale

You’ve already fused radius filtering + CSR emission inside Numba. Two small cleanups keep it fast at 32 k:

* **Exact capacity reservation** per kept group (`pair_counts` → offsets) is already there. Ensure the kernel *doesn’t* allocate new arrays per batch; obtain them from the **pool** (B2).
* **Symmetric writes in‑kernel** (you already do). Keep this, it removes a post‑pass.

---

## F. Guardrails: “degenerate fast‑paths” done right

You sketched this earlier; bake it in with tiny checks:

* If `scope_groups_unique == 1` and `max_group_size <= 1`: **skip** MIS/adjacency, emit empty CSR, go straight to bulk splice with dominated reattachments. This avoids ~sub‑ms work per batch and, more importantly, avoids touching shared scratch (better cache locality for the hot traversal/persistence passes).

---

## G. Compilation & lifetime

* Use **explicit signatures** on your hot kernels to avoid accidental recompiles on dtype/shape drift.
  Example: `@nb.njit("(i8[:],i4[:],i4[:],i4[:],i4[:],i8[:])", cache=True, parallel=True)`.
* **Import‑time warm‑up** as you already added: call the kernels once with tiny arrays so the cold start disappears from the first real batch.
* Keep **diagnostics off** in prod runs; they meaningfully perturb large‑n timings in your notes, especially at 32 k.

---

## H. What to expect (conservative deltas)

On typical hardware similar to your logs:

* **A1+A2** (bound‑aware distance + `ceil_log2_from_sqdist`)
  1.3–1.8× faster traversal on dominated/mixed batches; negligible overhead on cold batches.
* **B1–B4** (journal + one CoW + pooled scratch + bulk splice + incremental level counts)
  1.5–2.5× faster build at 32 k, mostly from removing repeated Python‑side slicing and small transient allocations. Also flattens RSS spikes.
* **C1–C2** (threading hygiene)
  Brings build CPU utilisation from ~1.2× to ~2–3× on 32 k (depending on cores), typically another ~1.2–1.5× wall‑time improvement if you were oversubscribed before.
* **D1–D2** (I32 and relabel)
  10–25% on large builds from cache hits + lower bandwidth.

Taken together, these are realistically a **2–4× further reduction** in **32 k build wall time**, with stability gains (lower variance, lower RSS deltas).

---

## Drop‑in code hooks (where to put things)

* `covertreex/algo/_dist_numba.py` – A1/A2 distance helpers.
* `covertreex/core/_persistence_numba.py` – B1/B3/B4 (apply‑journal & splice).
* `covertreex/algo/_pool_numba.py` – B2 buffer pool.
* Wire the **journal** in `batch_insert.py`:

  1. build scopes → conflict graph → MIS,
  2. fill journal arrays (`j_*`) for parent/level/head/next/level_counts,
  3. call `apply_journal_cow(...)`,
  4. swap in `*_out` arrays.

Add counters to your existing diagnostics:

* `cow_bytes_copied`, `cow_segments`, `pool_grow_bytes`, `pool_reuse_hits`, `journal_entries`, `splice_count`, `level_delta_nonzeros`.

---

## CI/regression knobs to lock the gains

* New tests:

  * `tests/integration/test_persistence_apply_journal.py`: verifies that only touched nodes/parents/levels changed across versions and that previous versions stay readable.
  * `tests/test_pool_reuse.py`: asserts the pool doesn’t reallocate for identical batch shapes.
  * `tests/test_dist_bound.py`: correctness of early‑exit distance vs full distance; randomised property tests with tight/loose bounds.
* Extend `benchmarks/runtime_breakdown` CSV with:

  * `build_cow_ms`, `pool_reuse_ratio`, `journal_size`, `splice_ms`, `level_prefix_ms`, and `traversal_bound_hit_rate`.

---

### Final checklist (do these first)

1. **Journal + single CoW sweep** (B1) — biggest structural win.
2. **Pool large temporaries** (B2) — removes 32 k RSS spikes and allocator overhead.
3. **Bound‑aware squared distances + fast ceil_log2** (A1/A2) — cheap, wide‑impact.
4. **Threading hygiene** (C1) — ensures Numba threads aren’t fighting BLAS.
5. **Bulk splice + incremental level counts** (B3/B4) — keeps the update phase O(1) per change in one pass.
6. **I32 end‑to‑end** (D1) — bandwidth/caches.
7. **Optional renumbering** (D2) — apply after big batches; measurable at 32 k+.

If you want, I can turn the sketches above into concrete PR‑ready modules (`_persistence_numba.py`, `_dist_numba.py`, and a minimal pool) and a small benchmark harness that isolates **apply‑journal CoW** cost before/after the change.
