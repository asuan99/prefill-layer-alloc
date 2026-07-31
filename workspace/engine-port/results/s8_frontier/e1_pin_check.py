#!/usr/bin/env python3
"""Realized-partition gate for the E1 frontier campaign.

★HISTORY (SIX distinct bugs found and fixed across 2026-07-28/07-30, kept
here so the same mistake is never rediscovered from scratch):
  #1 (2026-07-28, smoke 865832): gate population was `engaged =
     prefill_active OR decode_active`, wrongly counting decode-only windows
     (which are SUPPOSED to show the unpartitioned stream by design,
     `multiplexing_mixin.py`'s stream-selection) as pin failures. Fixed by
     conditioning on `prefill_active_batch_size > 0` only.
  #2 (2026-07-29, smoke 866066): the MANDATORY CONCURRENCY DIAGNOSTIC
     (`concurrent_frac_of_bench`) was snapshot-COUNT-based, silently
     reweighting by inverse iteration-duration (a 235ms prefill iteration
     and an 11ms decode iteration each count as "1"). Fixed by time-
     weighting (`concurrent_time_frac` etc.) -- this one was RIGHT the
     first time and stays time-weighted through every later revision.
  #3 (2026-07-29, smoke 866868/867034): `achieved_rps`/arrival timing had
     three sub-issues (tail dilution, no warmup discard, NP too small at low
     rate) -- fixed in `e1_capacity_scan.sbatch`, not this file; see
     DESIGN.md sec 4.2.1-4.2.3. Its seed-policy corollary (867034 diagnosis:
     --seed never overridden, so every rep/cell/arm shared one bit-identical
     workload realization) is fixed in `e1_sweep.sbatch`/DESIGN.md sec 4.5,
     also not this file.
  #4 (2026-07-29): bug #1's fix conditioned on `prefill_active_batch_size >
     0` SNAPSHOTS, but `--chunked-prefill-size -1` means one prompt = one
     forward pass = ONE event-loop iteration, so a prefill EPISODE produces
     AT MOST one snapshot before `dual_worker_trace_every` (32x) subsampling
     discards most of even that -- counting snapshots undercounts EPISODES
     regardless of duration (a different problem from bug #2's duration-
     weighting: this is about the wrong OBJECT being counted).
  #5 (2026-07-29): the bug #4 fix (episode bracketing, ported from
     `s8p_prefill/s8p_analyze.py`) read the realized partition at the
     bracket ENDPOINTS, which are outside the request's own prefill window
     BY CONSTRUCTION -- so a fast cell's endpoints only ever see the
     runtime's idle auto-partition fallback, never the target, no matter
     how much sampling density is added. Fixed by reading the realized
     partition from snapshots STRICTLY INSIDE the bracket instead
     (`compute_episode_gate` below) -- endpoints now used ONLY for interval-
     boundary proximity sanity, never for the partition read.
  #6 (2026-07-30, coordinator's own hand-aggregation of 866066's raw
     telemetry): bugs #4/#5's fix was itself answering the WRONG QUESTION
     for a gate. `compute_episode_gate` (kept, see below) answers "which
     realized partition served THIS SPECIFIC request's TTFT" -- a
     per-request LATENCY-ATTRIBUTION question that legitimately throws away
     most prefill-active samples to preserve per-request granularity. The
     GATE's actual question is different: "what fraction of prefill WORK
     (time) ran at the target partition," which should use ALL available
     prefill-active samples, TIME-weighted (bug #2's insight, re-applied
     here to a different quantity). Coordinator's manual count on 866066
     D16: only 12 of 32,145 total snapshots were prefill-active (7 at the
     P92 target, 5 at the P108 auto-partition fallback) -- COUNT fraction
     7/12=0.583 (matches this file's pre-bug-#4 snapshot-count output
     exactly), but TIME-weighting those SAME 12 snapshots gives 0.847 (the
     5 fallback snapshots occupy disproportionately SHORT durations) --
     independently reproduced by this implementation. `compute_episode_gate`
     is RESCOPED to latency-attribution only (its own bug #5 fix remains
     correct for that purpose); `compute_time_weighted_pin_gate` is now the
     PRIMARY, cite-blocking gate.

**Corrected design (single source of truth: the `compute_*`/`*_verdict`
functions here, imported by `e1_analyze.py` so the sbatch's own stdout and
the analyzer's citeability filter never drift apart):**

1. **PRIMARY, cite-blocking gate (2026-07-30) = TIME-WEIGHTED realized-
   partition pin.** `compute_time_weighted_pin_gate()`: among ALL prefill-
   active WALL-CLOCK TIME in the window (every snapshot with
   `prefill_active_batch_size>0`, weighted by the interval to the next
   snapshot -- same convention as `compute_concurrency_diagnostic`), the
   fraction whose realized `prefill_sms` equals the cell's target must have
   an episode-level (cluster) BOOTSTRAP 95% lower bound `>= 0.80`. Prefill-
   active time is partitioned into EPISODES (maximal contiguous runs of
   `prefill_active_batch_size>0`) for the bootstrap resampling, since
   consecutive snapshots within one run are not independent observations
   and Clopper-Pearson (bug #4/#5's criterion, assumes iid Bernoulli
   trials) does not apply to a time-weighted, serially correlated
   quantity. `n_episodes` (NOT the raw snapshot count, NOT a bracketed-
   request count) is the effective sample size to read for statistical
   power, and MUST be reported alongside every verdict.
2. **`compute_episode_gate()` is RETAINED but RESCOPED to latency
   attribution only, NOT used for gating.** Its own bug #5 fix (read the
   realized partition from INSIDE the bracket, not the endpoints) remains
   correct for its (now narrower) purpose: if a future analysis needs "what
   partition served THIS request's TTFT" on a per-request basis (e.g. to
   condition latency percentiles on realized SM), this is still the right
   tool -- it is simply not equipped to answer the GATE's aggregate
   question, because getting per-request granularity requires discarding
   most of the available prefill-active evidence (866066 D16: only 1 of
   150 requests could be bracketed to a single inside-bracket sample at
   all, vs all 12 prefill-active snapshots being usable for the time-
   weighted gate above).
3. **MANDATORY DIAGNOSTIC (unchanged since bug #2) =
   `compute_concurrency_diagnostic()`**: the fraction of the window's
   WALL-CLOCK where prefill and decode are both simultaneously active.
   Time-weighted for the same reason the pin gate now is, but answers a
   genuinely different question (concurrent OCCUPANCY, not partition
   CORRECTNESS) and is computed independently.

`prefill_admission_blocked` is reported over ALL benchmark-phase snapshots
with `trace_forced != True` (2026-07-29 fix for engine-porter's
`PDMUX_TRACE_FORCE_PREFILL` patch, see `compute_concurrency_diagnostic`'s
own docstring) -- by construction a blocked sample has
`prefill_active_batch_size==0` (`dual_worker.py:590-599`), so conditioning
on prefill-active would always show 0% and hide the signal.

Usage (CLI):
  e1_pin_check.py <telemetry.jsonl> <expected_decode_sm> [min_lower95] \
      [raw_jsonl_path seed rate] --t0 T0_MONOTONIC_S --t1 T1_MONOTONIC_S

The PRIMARY time-weighted gate only needs `<telemetry.jsonl>`,
`<expected_decode_sm>`, and the `--t0`/`--t1` window -- `raw_jsonl_path`/
`seed`/`rate` are OPTIONAL and, if given, additionally run the (non-
gating, latency-attribution-only) episode bracket for informational
comparison.

Exit status is always 0; the printed verdict lines are what the campaign log
records.
"""
import bisect
import json
import sys

from scipy.stats import beta as _beta

D_TO_PREFILL = {16: 92, 24: 84, 44: 64, 54: 54, 92: 16}
DEFAULT_GUARD_S = 3.0


# ---------------------------------------------------------------------------
# Time-weighted concurrency diagnostic (bug #2 fix, largely UNCHANGED by
# this revision -- see module docstring point 4 -- EXCEPT admission_blocked_frac,
# fixed here for engine-porter's PDMUX_TRACE_FORCE_PREFILL patch, see below.
# ---------------------------------------------------------------------------
def compute_concurrency_diagnostic(path, t0=None, t1=None):
    """Returns time-weighted prefill/decode/concurrent occupancy fractions
    over [t0, t1] (or the whole `phase=="benchmark"` telemetry if both are
    None), plus admission-blocked stats. This is the MANDATORY DIAGNOSTIC,
    not gate-blocking on its own.

    `admission_blocked_frac` (2026-07-29, coordinator + engine-porter
    diagnosis): computed only over samples with `trace_forced != True`.
    Engine-porter's `PDMUX_TRACE_FORCE_PREFILL` patch (default OFF, byte-
    identical output when off -- so this filter is a no-op on all telemetry
    predating the patch or with it disabled) forces an extra snapshot
    whenever prefill is in flight -- by construction every such forced
    record has `prefill_active_batch_size>0`, which is mutually exclusive
    with `prefill_admission_blocked` (`dual_worker.py:590-599`:
    `admission_blocked = queue_depth>0 AND active_batch_size==0`). Counting
    forced records in the denominator (they can NEVER be numerator hits)
    mechanically dilutes `admission_blocked_frac` toward 0, and does so
    MORE for cells where forcing adds proportionally more records (slower
    prefill, e.g. D92 -- see DESIGN.md sec 4.7's emission-count asymmetry)
    -- a cell-dependent bias that would corrupt any cross-cell comparison
    of this fraction once the flag is on. The time-weighted quantities
    above (`*_time_frac`) are NOT filtered this way -- they benefit from
    the extra resolution forcing provides and are not mechanically biased
    by it (more snapshots inside a truly-active window only refine the
    time-weighting, it does not shift which fraction of TIME was active)."""
    n_bench = 0
    n_bench_natural = 0
    admission_blocked_natural = 0
    block_reasons = {}
    rows = []  # (ts, p_active, d_active)

    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
            continue
        ts = e.get("timestamp_monotonic_s")
        if t0 is not None and (ts is None or ts < t0):
            continue
        if t1 is not None and (ts is None or ts > t1):
            continue
        n_bench += 1
        p_active = e.get("prefill_active_batch_size", 0) > 0
        d_active = e.get("decode_running_batch_size", 0) > 0
        rows.append((ts, p_active, d_active))
        if not e.get("trace_forced"):
            n_bench_natural += 1
            if e.get("prefill_admission_blocked"):
                admission_blocked_natural += 1
                reason = e.get("prefill_admission_block_reason") or "(unset)"
                block_reasons[reason] = block_reasons.get(reason, 0) + 1

    rows.sort(key=lambda r: r[0])
    t_total = t_prefill = t_decode = t_both = 0.0
    for i, (ts, p, d) in enumerate(rows):
        nxt = rows[i + 1][0] if i + 1 < len(rows) else t1
        if nxt is None:
            continue
        dt = max(0.0, nxt - ts)
        t_total += dt
        if p:
            t_prefill += dt
        if d:
            t_decode += dt
        if p and d:
            t_both += dt

    return dict(
        n_bench=n_bench,
        t_total_s=t_total,
        prefill_active_time_frac=(t_prefill / t_total) if t_total else float("nan"),
        decode_active_time_frac=(t_decode / t_total) if t_total else float("nan"),
        concurrent_time_frac=(t_both / t_total) if t_total else float("nan"),
        admission_blocked_frac=(admission_blocked_natural / n_bench_natural) if n_bench_natural else float("nan"),
        n_bench_natural=n_bench_natural,
        block_reasons=block_reasons,
    )


# ---------------------------------------------------------------------------
# ★★★★★★ Time-weighted PIN GATE (bug #6 fix, 2026-07-30 -- coordinator's own
# raw-data re-derivation). THIS IS NOW THE PRIMARY, CITE-BLOCKING GATE. See
# module docstring bug #6 for the full history: bug #4/#5's episode-bracket
# approach answers a DIFFERENT question (per-request latency attribution)
# than the gate needs (aggregate "what fraction of prefill WORK ran at
# target"), and answering the gate's question by throwing away most of the
# data to get per-request granularity was itself the mistake -- not bug #5's
# inside-vs-endpoint fix, which stays correct for its own (now narrower)
# purpose. See `compute_episode_gate` below, retained for latency
# attribution only, NOT used for gating any more.
# ---------------------------------------------------------------------------
def compute_time_weighted_pin_gate(telemetry_path, expect_d, t0=None, t1=None,
                                    n_boot=10000, seed=20260730):
    """PRIMARY pin gate (2026-07-30 revision): among ALL prefill-active
    WALL-CLOCK TIME in the window (every `runtime_snapshot` row with
    `prefill_active_batch_size>0`, weighted by the interval to the next
    row -- same time-weighting convention as `compute_concurrency_diagnostic`,
    NOT a per-request bracket subsample), what fraction ran at the cell's
    target `prefill_sms`? This directly answers the gate's actual question
    ("did prefill work run on the target partition") using every available
    prefill-active sample, rather than discarding most of them to get
    per-request attribution the gate does not need (bug #6).

    Uncertainty: Clopper-Pearson (used by the retired episode-based gate,
    bug #4/#5) assumes iid Bernoulli trials, which time-weighted, serially
    correlated snapshots are not -- consecutive snapshots within one
    contiguous prefill-active run are not independent observations. Instead,
    prefill-active time is partitioned into EPISODES (maximal contiguous
    runs of `prefill_active_batch_size>0`, regardless of which `prefill_sms`
    value they show -- a policy switch mid-episode would be an anomaly, not
    assumed away), and the reported CI is a CLUSTER (episode-level)
    BOOTSTRAP: resample episodes with replacement, and for each resample
    recompute the pooled ratio `sum(target-time)/sum(total-time)` over the
    resampled episodes. `n_episodes` is reported as the effective sample
    size for this estimator -- it is what should be read as "how much
    independent evidence supports this fraction," not the raw snapshot
    count (bug #4) nor the bracketed-request count (bug #5's function).

    Verified against the coordinator's own independent hand-aggregation of
    866066's D16 telemetry (32,145 total snapshots, only 12 prefill-active:
    7 at the P92 target, 5 at the P108 auto-partition fallback) -- COUNT-based
    fraction 7/12=0.583 (matches this file's pre-bug#4 output exactly); this
    function's TIME-weighted fraction over the SAME 12 snapshots is 0.847
    (1.719s of 2.030s total prefill-active time), because the 5 fallback
    snapshots occupy disproportionately SHORT durations relative to the 7
    target snapshots -- reproduced independently by this implementation."""
    expect_p = D_TO_PREFILL.get(expect_d)
    if expect_p is None:
        raise ValueError(f"unknown D={expect_d}, not in {sorted(D_TO_PREFILL)}")

    rows = []
    for line in open(telemetry_path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
            continue
        ts = e.get("timestamp_monotonic_s")
        if t0 is not None and (ts is None or ts < t0):
            continue
        if t1 is not None and (ts is None or ts > t1):
            continue
        rows.append((ts, e.get("prefill_sms"), e.get("prefill_active_batch_size", 0)))
    rows.sort(key=lambda r: r[0])

    # Build episodes: maximal contiguous runs of prefill_active_batch_size>0.
    # Each episode accumulates (t_total, t_target, realized_hist) using the
    # SAME weight-to-next-row convention as compute_concurrency_diagnostic.
    episodes = []          # list of (t_total, t_target)
    realized_hist = {}     # {prefill_sms: total time (s), across ALL episodes}
    cur_total = cur_target = 0.0
    in_episode = False
    for i, (ts, psm, pab) in enumerate(rows):
        nxt = rows[i + 1][0] if i + 1 < len(rows) else t1
        if nxt is None:
            continue
        dt = max(0.0, nxt - ts)
        if pab > 0:
            in_episode = True
            cur_total += dt
            if psm == expect_p:
                cur_target += dt
            realized_hist[psm] = realized_hist.get(psm, 0.0) + dt
        else:
            if in_episode:
                episodes.append((cur_total, cur_target))
                cur_total = cur_target = 0.0
                in_episode = False
    if in_episode:
        episodes.append((cur_total, cur_target))

    n_episodes = len(episodes)
    t_prefill_active_total = sum(e[0] for e in episodes)
    t_target = sum(e[1] for e in episodes)
    pin_frac = (t_target / t_prefill_active_total) if t_prefill_active_total else float("nan")

    # Episode-level (cluster) bootstrap for the pooled ratio estimator.
    if n_episodes >= 2:
        import random as _random
        rng = _random.Random(seed)
        boots = []
        for _ in range(n_boot):
            sample = [episodes[rng.randrange(n_episodes)] for _ in range(n_episodes)]
            st_, sg_ = sum(e[0] for e in sample), sum(e[1] for e in sample)
            if st_ > 0:
                boots.append(sg_ / st_)
        boots.sort()
        pin_frac_lower95 = boots[int(0.025 * len(boots))] if boots else 0.0
        bootstrap_reliable = True
    elif n_episodes == 1:
        # A single cluster cannot be resampled meaningfully (every bootstrap
        # draw reproduces the same one episode) -- report the point estimate
        # as its own bound but flag it as NOT a real CI, per the coordinator's
        # instruction to report uncertainty honestly rather than fabricate a
        # confident-looking number from n=1.
        pin_frac_lower95 = pin_frac if pin_frac == pin_frac else 0.0
        bootstrap_reliable = False
    else:
        pin_frac_lower95 = 0.0
        bootstrap_reliable = False

    return dict(
        expect_p=expect_p, expect_d=expect_d,
        n_episodes=n_episodes, t_prefill_active_total_s=t_prefill_active_total,
        t_target_s=t_target, pin_frac=pin_frac, pin_frac_lower95=pin_frac_lower95,
        bootstrap_reliable=bootstrap_reliable, realized_hist=realized_hist,
    )


def time_weighted_gate_verdict(g, min_lower95=0.80):
    """PRIMARY, cite-blocking criterion (2026-07-30 revision): the
    episode-bootstrap 95% lower bound of the TIME-WEIGHTED pin fraction
    must clear min_lower95. A cell with 0 prefill-active time fails with an
    explicit reason; a cell with exactly 1 episode passes/fails on its
    (unreliable, `bootstrap_reliable=False`) point estimate but the report
    MUST surface that unreliability, never silently treat n_episodes=1 as
    equivalent to a real bootstrap CI."""
    if g["n_episodes"] == 0 or g["t_prefill_active_total_s"] <= 0:
        return dict(pin_pass=False, reason="no prefill-active time in this window "
                                            "(n_episodes=0) -- cannot judge the pin gate")
    pin_pass = g["pin_frac_lower95"] >= min_lower95
    reliability_note = "" if g["bootstrap_reliable"] else \
        f"  [UNRELIABLE: only n_episodes={g['n_episodes']}, bootstrap CI not meaningful]"
    return dict(pin_pass=pin_pass,
                 reason=("" if pin_pass else
                         f"pin_frac_lower95={g['pin_frac_lower95']:.3f} < {min_lower95} "
                         f"(point estimate {g['pin_frac']:.3f}, n_episodes={g['n_episodes']})")
                 + reliability_note)


# ---------------------------------------------------------------------------
# Episode-based per-request bracket (bug #4/#5 fix, 2026-07-29 -- ported
# bracketing technique from s8p_prefill/s8p_analyze.py). ★★★RESCOPED
# 2026-07-30 (bug #6): this answers "which realized partition served THIS
# SPECIFIC request's TTFT" -- a per-request LATENCY-ATTRIBUTION question --
# and is retained for that purpose only. It is NO LONGER the pin gate;
# `compute_time_weighted_pin_gate` above is. Nothing in this section's own
# bug #4/#5 fix (inside-bracket vs endpoint reading) was wrong; only its use
# AS THE GATE was.
# ---------------------------------------------------------------------------
def reconstruct_arrival_and_completion(seed, rate, ttfts, itls):
    """RNG-replay reconstruction of each request's relative dispatch
    (arrival) time and relative completion time, for `--dataset-name
    sharegpt` (dataset prep only touches Python's stdlib `random`, never
    `np.random` -- `sglang/benchmark/datasets/sharegpt.py:98` -- so the
    `np.random` stream is untouched between `np.random.seed()`
    (`bench_serving.py:1706`) and `get_request()`'s first draw
    (`bench_serving.py:948`)). VALIDATED (2026-07-29, CPU-only, no server
    needed): replayed against `sglang.bench_serving.get_request()`'s own
    actual asyncio-scheduled dispatch times for seed in {1,2,7} x rate in
    {2,8} -- max abs error 2-70ms out of several seconds (event-loop
    jitter), confirming this generalizes to ANY seed. FRAGILE: re-verify if
    the dataset/backend changes to one that DOES consume `np.random` during
    prep (e.g. `--dataset-name random-ids`,
    `sglang/benchmark/datasets/random.py:147`)."""
    import numpy as np
    npc = len(ttfts)
    np.random.seed(seed)
    intervals = [np.random.exponential(1.0 / rate) for _ in range(npc)]
    arrival_rel = np.cumsum([0.0] + intervals[:-1])
    completion_rel = []
    for i in range(npc):
        lat = ttfts[i] + (sum(itls[i]) if itls[i] else 0.0)
        completion_rel.append(arrival_rel[i] + lat)
    return arrival_rel.tolist(), completion_rel


def compute_episode_gate(telemetry_path, expect_d, seed, rate, ttfts, itls, errors,
                          t1_anchor, guard=DEFAULT_GUARD_S):
    """Episode-based pin gate. One episode = one request's [admission,
    first-token] interval, reconstructed via `reconstruct_arrival_and_completion`
    and anchored so the LAST request's reconstructed completion time lands
    exactly at `t1_anchor` (the phase's own measured end -- e.g.
    `e1_sweep.sbatch`'s `T1_LO`/`T1_HI`).

    ★★★RE-REVISED 2026-07-29 (bug #5, coordinator + engine-porter
    diagnosis): the PREVIOUS version of this function read the realized
    partition at the bracket ENDPOINTS (`ii` = last snapshot at or before
    admission, `jj` = first snapshot at or after first-token). Both
    endpoints are, BY CONSTRUCTION, outside the request's own prefill
    window -- so at a fast-prefill cell they only ever see the runtime's
    idle/auto-partition fallback (866066/D16: endpoint histogram over
    "accepted" brackets was P0:69/P108:28, NEVER the P92 target), and
    D16 was informative-0 not because prefill never ran at target, but
    because the gate was looking at the wrong two instants. A read-only
    diagnostic (`results/e1_traceforce/tfgate_inside_bracket.py`, built
    while validating engine-porter's `PDMUX_TRACE_FORCE_PREFILL` patch)
    confirmed real evidence exists INSIDE the bracket even pre-patch: D16
    had 2 prefill-active snapshots strictly between admission and
    first-token, both at the P92 target (on_target_frac=1.000); D92
    (control) had 44, all at P16 (on_target_frac=1.000, matching the
    already-established result).

    **Fix**: the realized partition is now read from snapshots STRICTLY
    INSIDE `(a, b)` with `prefill_active_batch_size > 0` -- `lo =
    bisect.bisect_right(ts, a)`, `hi = bisect.bisect_left(ts, b)`,
    `range(lo, hi)` (same indexing as the diagnostic script). The bracket
    ENDPOINTS (`ii`, `jj`, ported verbatim from `s8p_prefill/s8p_analyze.py`)
    are used ONLY to confirm the interval's own boundaries are close to real
    telemetry (`guard` proximity) -- a timing sanity check, not a partition
    read -- per explicit instruction not to use them for partition
    determination any more. The old `psm[ii]==psm[jj]` consistency
    requirement is DROPPED (it was a proxy for "nothing changed during the
    episode" that made sense only when partition reads came from the
    endpoints themselves; it no longer applies once reads come from inside).
    A bracket with no snapshot inside `(a,b)`, or none of them
    prefill-active, is `n_vacuous_idle` (no positive evidence, same
    semantics as before, just re-scoped to "inside" rather than "endpoints
    agree and are idle"). If multiple inside-active snapshots disagree on
    `prefill_sms` (should not happen under a static `FixedPolicy` config;
    tracked as `n_inconsistent_partition`), the chronologically FIRST one is
    used and the disagreement is still counted/reported, never silently
    resolved by majority vote."""
    expect_p = D_TO_PREFILL.get(expect_d)
    if expect_p is None:
        raise ValueError(f"unknown D={expect_d}, not in {sorted(D_TO_PREFILL)}")

    ts, psm, pab = [], [], []
    for line in open(telemetry_path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
            continue
        ts.append(e["timestamp_monotonic_s"])
        psm.append(e.get("prefill_sms"))
        pab.append(e.get("prefill_active_batch_size", 0))

    npc = len(ttfts)
    if npc == 0 or not ts:
        return dict(expect_p=expect_p, expect_d=expect_d, n_total=npc, n_accepted=0,
                     n_informative=0, acceptance_rate=float("nan"),
                     informative_rate=float("nan"), pin_frac=float("nan"),
                     pin_frac_lower95=0.0, n_vacuous_idle=0, n_inconsistent_partition=0,
                     realized_hist_endpoints={}, realized_hist_informative={})

    arrival_rel, completion_rel = reconstruct_arrival_and_completion(seed, rate, ttfts, itls)
    t0_abs = t1_anchor - max(completion_rel)

    n_accepted = 0
    n_vacuous_idle = 0
    n_inconsistent_partition = 0
    n_target_informative = 0
    realized_hist_endpoints = {}       # AUDIT-TRAIL ONLY (pre-fix method) -- not used for gating
    realized_hist_informative = {}     # PRIMARY (bug #5 fix): inside-bracket, gate-determining
    for i in range(npc):
        if errors and i < len(errors) and errors[i]:
            continue  # failed requests carry no valid ttft/itl, exclude from episode accounting
        a = t0_abs + arrival_rel[i]
        b = a + ttfts[i]
        ii = bisect.bisect_right(ts, a) - 1
        jj = bisect.bisect_left(ts, b)
        if ii < 0 or jj >= len(ts):
            continue
        if (a - ts[ii]) > guard or (ts[jj] - b) > guard:
            continue
        # Boundary sanity passes -- this is "accepted" (matches the ported
        # algorithm's population, but the psm[ii]==psm[jj] consistency
        # check is DROPPED, see docstring: endpoints no longer determine
        # partition, only confirm the interval is near real telemetry).
        n_accepted += 1
        realized_hist_endpoints[psm[ii]] = realized_hist_endpoints.get(psm[ii], 0) + 1

        lo = bisect.bisect_right(ts, a)   # first index with ts strictly > a
        hi = bisect.bisect_left(ts, b)    # first index with ts >= b
        inside_active = [k for k in range(lo, hi) if pab[k] > 0]
        if not inside_active:
            n_vacuous_idle += 1
            continue
        inside_psm_values = {psm[k] for k in inside_active}
        if len(inside_psm_values) > 1:
            n_inconsistent_partition += 1
        realized_psm = psm[inside_active[0]]  # chronologically first inside-active sample
        realized_hist_informative[realized_psm] = realized_hist_informative.get(realized_psm, 0) + 1
        if realized_psm == expect_p:
            n_target_informative += 1

    n_informative = n_accepted - n_vacuous_idle
    pin_frac = (n_target_informative / n_informative) if n_informative else float("nan")
    if n_informative:
        pin_frac_lower95 = 0.0 if n_target_informative == 0 else float(
            _beta.ppf(0.05, n_target_informative, n_informative - n_target_informative + 1))
    else:
        pin_frac_lower95 = 0.0

    return dict(
        expect_p=expect_p, expect_d=expect_d, n_total=npc,
        n_accepted=n_accepted, n_informative=n_informative,
        acceptance_rate=(n_accepted / npc) if npc else float("nan"),
        informative_rate=(n_informative / npc) if npc else float("nan"),
        pin_frac=pin_frac, pin_frac_lower95=pin_frac_lower95,
        n_vacuous_idle=n_vacuous_idle, n_inconsistent_partition=n_inconsistent_partition,
        realized_hist_endpoints=realized_hist_endpoints,
        realized_hist_informative=realized_hist_informative,
    )


def episode_gate_verdict(g, min_lower95=0.80):
    """Single, honest pass/fail criterion (2026-07-29 revision): the
    Clopper-Pearson 95% lower bound of pin_frac, computed over INFORMATIVE
    episodes (accepted brackets minus vacuous-idle ones, see
    `compute_episode_gate`'s docstring), must clear min_lower95. Replaces
    the old two-part `pin_frac >= threshold AND n >= min_n` check -- a small
    n now widens the CI and fails this test on its own, so there is no
    separate `UNDERPOWERED` reason code any more; low n and wrong SM both
    show up as "the lower bound isn't high enough," which is the honest
    description of what a small, noisy, or genuinely-off-target sample all
    look like. A cell with n_informative=0 (e.g. D16 in this campaign, whose
    92-SM prefill windows are apparently too brief to ever be caught with
    positive evidence even across n=150 requests) fails here with an
    explicit "no informative episodes" reason -- a transparent, honest
    non-verdict, not a false "wrong SM" the way pin_frac=0/0 would read if
    vacuous brackets were not excluded."""
    if g["n_informative"] == 0:
        return dict(pin_pass=False,
                     reason=f"no INFORMATIVE episodes (n_accepted={g['n_accepted']}, "
                            f"all {g['n_vacuous_idle']} were vacuous-idle brackets with "
                            "no positive evidence of prefill activity -- see module docstring)")
    pin_pass = g["pin_frac_lower95"] >= min_lower95
    return dict(pin_pass=pin_pass,
                 reason="" if pin_pass else
                 f"pin_frac_lower95={g['pin_frac_lower95']:.3f} < {min_lower95} "
                 f"(point estimate {g['pin_frac']:.3f}, n_informative={g['n_informative']})")


def main():
    argv = list(sys.argv[1:])
    t0 = t1 = None
    if "--t0" in argv:
        i = argv.index("--t0")
        t0 = float(argv[i + 1])
        del argv[i:i + 2]
    if "--t1" in argv:
        i = argv.index("--t1")
        t1 = float(argv[i + 1])
        del argv[i:i + 2]

    telemetry_path = argv[0]
    expect_d = int(argv[1])
    min_lower95 = float(argv[2]) if len(argv) > 2 else 0.80

    conc = compute_concurrency_diagnostic(telemetry_path, t0, t1)
    window = f" window=[{t0},{t1}]" if (t0 is not None or t1 is not None) else ""
    print(f"E1_PIN_CHECK D={expect_d}{window}: n_bench_snapshots={conc['n_bench']}")
    print(f"E1_CONCURRENCY_DIAGNOSTIC(time-weighted, window={conc['t_total_s']:.2f}s): "
          f"concurrent_time_frac={conc['concurrent_time_frac']:.4f} "
          f"prefill_active_time_frac={conc['prefill_active_time_frac']:.4f} "
          f"decode_active_time_frac={conc['decode_active_time_frac']:.4f}")
    print(f"E1_ADMISSION_BLOCKED(trace_forced-excluded bench snapshots, n={conc['n_bench_natural']} "
          f"of {conc['n_bench']} total): frac={conc['admission_blocked_frac']:.3f} "
          f"reasons={conc['block_reasons']}")

    # --- PRIMARY, cite-blocking gate (2026-07-30 revision, bug #6): TIME-WEIGHTED. ---
    g = compute_time_weighted_pin_gate(telemetry_path, expect_d, t0, t1)
    v = time_weighted_gate_verdict(g, min_lower95)
    dist = ", ".join(f"P{p}:{t_:.3f}s({t_/g['t_prefill_active_total_s']*100:.0f}%)"
                      for p, t_ in sorted(g["realized_hist"].items(), key=lambda kv: -kv[1])) \
        if g["t_prefill_active_total_s"] else "(none)"
    print(f"E1_REALIZED_hist(TIME-WEIGHTED over prefill-active time, PRIMARY/gate-determining, "
          f"total={g['t_prefill_active_total_s']:.3f}s): {dist}")
    print(f"E1_PIN_GATE D={expect_d}(P{g['expect_p']}) [TIME-WEIGHTED, bug #6]: "
          f"{'PASS' if v['pin_pass'] else 'FAIL -> ' + v['reason']} "
          f"(pin_frac={g['pin_frac']:.3f}, lower95={g['pin_frac_lower95']:.3f}, "
          f"n_episodes={g['n_episodes']}, min_lower95={min_lower95})")

    # --- OPTIONAL, non-gating: per-request latency-attribution diagnostic
    # (compute_episode_gate, rescoped 2026-07-30 -- NOT the pin gate any
    # more). Only runs if raw_jsonl/seed/rate are additionally given. ---
    if len(argv) < 6:
        return
    raw_jsonl, seed_s, rate_s = argv[3], argv[4], argv[5]
    seed, rate = int(seed_s), float(rate_s)
    if t1 is None:
        print("E1_EPISODE_ATTRIBUTION: skipped (--t1 required as the reconstruction anchor)")
        return

    ttfts, itls, errors = [], [], []
    for line in open(raw_jsonl):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        ttfts = o.get("ttfts") or []
        itls = o.get("itls") or []
        errors = o.get("errors") or []
        break  # one row per phase file

    ge = compute_episode_gate(telemetry_path, expect_d, seed, rate, ttfts, itls, errors, t1)
    dist_endpoints = ", ".join(f"P{p}:{c}({c/ge['n_accepted']*100:.0f}%)" for p, c in
                                sorted(ge["realized_hist_endpoints"].items(), key=lambda kv: -kv[1])) \
        if ge["n_accepted"] else "(none)"
    dist_inf = ", ".join(f"P{p}:{c}({c/ge['n_informative']*100:.0f}%)" for p, c in
                          sorted(ge["realized_hist_informative"].items(), key=lambda kv: -kv[1])) \
        if ge["n_informative"] else "(none)"
    print(f"E1_EPISODE_ATTRIBUTION (informational ONLY since 2026-07-30 bug #6 -- "
          f"per-request latency attribution, NOT the pin gate): n_total={ge['n_total']} "
          f"n_accepted={ge['n_accepted']} (rate={ge['acceptance_rate']:.3f}) "
          f"n_vacuous_idle={ge['n_vacuous_idle']} n_informative={ge['n_informative']} "
          f"(rate={ge['informative_rate']:.3f}) n_inconsistent_partition={ge['n_inconsistent_partition']} "
          f"pin_frac(over informative brackets)={ge['pin_frac']:.3f}")
    print(f"E1_EPISODE_ATTRIBUTION_hist(bracket ENDPOINTS, audit-trail, n={ge['n_accepted']}): {dist_endpoints}")
    print(f"E1_EPISODE_ATTRIBUTION_hist(INSIDE-BRACKET informative, n={ge['n_informative']}): {dist_inf}")


if __name__ == "__main__":
    main()
