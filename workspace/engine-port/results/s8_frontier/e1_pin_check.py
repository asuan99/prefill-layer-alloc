#!/usr/bin/env python3
"""Realized-partition gate for the E1 frontier campaign.

★★REVISED 2026-07-28 (coordinator directive, smoke job 865832) then
★★★RE-REVISED 2026-07-29 (coordinator directive, smoke job 866066 exposed a
SECOND, DIFFERENT bug in the mandatory concurrency diagnostic -- read this
whole docstring, the two bugs are distinct and both matter).

**Bug #1 (fixed 2026-07-28): the PRIMARY pin gate's population was wrong.**
The original version conditioned on `engaged = prefill_active_batch_size>0
OR decode_running_batch_size>0`. That is WRONG for judging whether the
target partition was realized: `multiplexing_mixin.py`'s stream-selection
logic (`elif not self.running_batch.is_empty(): set_current_stream_idx(last
group)`, i.e. decode-only windows get the FULL unpartitioned (0,108) stream
BY DESIGN, not our [108-D,D] target) means decode-only samples are SUPPOSED
to show (0,108) -- counting them as "should have been at target, wasn't"
is not measuring a failure. Fixed by conditioning the PRIMARY gate on
`prefill_active_batch_size > 0` ONLY (mirrors
`s8p_prefill/prefill_pin_check.py:51,56-58,67` exactly). This fix is CORRECT
and UNCHANGED by the 2026-07-29 finding below -- `pin_frac` is a
request/event-conditional proportion (see bug #2's discussion of why this
matters), and NP=150 re-run (866066) confirms it behaves sensibly: d92
n=118-242, pin 0.968-1.000 PASS; d16 n=12, correctly flagged UNDERPOWERED
(not falsely "wrong SM").

**Bug #2 (fixed 2026-07-29): the MANDATORY CONCURRENCY DIAGNOSTIC
(`concurrent_frac_of_bench`/`prefill_active_frac_of_bench`) was ALSO
snapshot-COUNT-based, and for an ABSOLUTE WALL-CLOCK-SHARE claim (unlike
`pin_frac`, which is a conditional proportion) that is the SAME KIND of
error as bug #1, applied to a DIFFERENT quantity.** `runtime_snapshot` is
emitted once per EVENT-LOOP ITERATION (subsampled every
`dual_worker_trace_every`=32nd, see `multiplexing_mixin.py:90-95`), not
once per fixed wall-clock tick. A single prefill forward pass = ~1 event-
loop iteration = ~1 trace tick, REGARDLESS of how long that forward pass
takes (16 SM: ~235ms; 92 SM: ~a few ms); a single decode step is also ~1
iteration/tick but takes only ~1 ITL (~11-12ms here). So counting
"snapshots with property X" and dividing by "total snapshots" answers "what
fraction of EVENT-LOOP ITERATIONS have X", which silently reweights
everything by inverse-duration: a cell whose prefill windows are long (D92,
prefill=16 SM) gets the SAME ~1-tick representation per prefill event as a
cell whose prefill windows are short (D16, prefill=92 SM), while decode's
many short steps rack up many ticks in the same wall-clock time. The
result: `concurrent_frac_of_bench` computed this way UNDERSTATES true
wall-clock concurrency by ~15-30x (866066, T8: time-weighted recomputation
from the SAME telemetry gives d92 LO/HI concurrent_time_frac
0.248/0.398 vs the count-based 0.0147/0.0154 it had reported; d16 LO
0.0102 vs 0.0003 -- independently cross-checked against a client-side sum-
of-TTFT/duration upper bound, `d92 LO 63%/HI 255%(loose)` vs time-weighted
25%/40%, same ballpark, confirming direction and rough magnitude). This is
NOT the same bug as #1 -- `pin_frac` is a conditional PROPORTION (an
event-count quantity is the semantically RIGHT thing to compute for "was
the SM correct for the requests we measured"), while
`concurrent_frac_of_bench` is an ABSOLUTE SHARE-OF-TIME claim (a wall-clock
quantity), and only the latter needed the fix below. See
`DESIGN.md` sec 5.2/9.7 (2026-07-29 revision) for the full retraction of the
"D-axis barely exercised / possible ill-posed frontier" conclusion this bug
had produced -- that conclusion was the artifact, not a real finding.

**Fix: time-weighted occupancy.** Each `runtime_snapshot` row's state is
assumed to hold from its own `timestamp_monotonic_s` until the NEXT row's
timestamp (or, for the final row in a bounded window, until `t1`; for an
unbounded whole-file check, the final row's trailing duration is dropped --
negligible, one row's worth out of thousands). `prefill_active_time_frac`,
`decode_active_time_frac`, and `concurrent_time_frac` are the resulting
wall-clock-weighted fractions -- these are now the MANDATORY DIAGNOSTIC
(coordinator: report always, still not a hard pass/fail gate since no
principled threshold has been established). The OLD count-based numbers
are RETAINED under `*_count_based` keys, for audit-trail/comparison only --
**do not use them to argue "the D-axis was barely exercised"; that
inference was the bug.**

**Corrected design (single source of truth: `compute_gates`/`gate_verdict`,
imported by `e1_analyze.py` so the sbatch's own stdout and the analyzer's
citeability filter never drift apart):**

1. **PRIMARY, cite-blocking gate = REALIZED PARTITION PIN** -- unchanged by
   this revision, see bug #1 above. `pin_frac >= min_pin_frac` (0.80) among
   `n_prefill_active >= min_n_prefill_active` (20) prefill-active samples.
2. **MANDATORY DIAGNOSTIC (not gating) = TIME-WEIGHTED concurrency**
   (`concurrent_time_frac`, `prefill_active_time_frac`,
   `decode_active_time_frac`) -- see bug #2 above.

`prefill_admission_blocked` is reported over ALL benchmark-phase snapshots,
not the prefill-active population -- by construction
(`dual_worker.py:590-599`, `admission_blocked = queue_depth>0 and
active_batch_size==0`) a blocked sample has `prefill_active_batch_size==0`,
so conditioning on prefill-active would always show 0% and hide the signal.

Usage:
  e1_pin_check.py <telemetry.jsonl> <expected_decode_sm> [min_pin_frac] \
      [min_n_prefill_active] [--t0 T0_MONOTONIC_S --t1 T1_MONOTONIC_S]

Without --t0/--t1 the whole telemetry file (phase=benchmark rows) is used --
this is the cell-level check. With --t0/--t1 only rows whose
timestamp_monotonic_s falls in [t0, t1] are used -- this is the per-round
check e1_analyze.py uses to gate individual reps (methodology gate: n>=4,
rep pooling forbidden, so bad reps must be excluded per-rep, not averaged
away at the cell level). The time-weighted computation uses the SAME
window: the trailing duration of the last in-window sample is capped to
`t1` when a window is given.

D92/D54 cells also carry a GUARD-SATISFIER division (pdmux_e1_d54.yml /
pdmux_e1_d92.yml) that is never selected by FixedPolicy but does appear in
`sm_counts`; this script does not need to special-case it because it only
ever looks at the REALIZED (prefill_sms, decode_sms) pair actually reported
per sample, not at the config's division list.

Exit status is always 0; the printed verdict lines are what the campaign log
records.
"""
import collections
import json
import statistics as st
import sys

D_TO_PREFILL = {16: 92, 24: 84, 44: 64, 54: 54, 92: 16}
DEFAULT_MIN_N_PREFILL_ACTIVE = 20


def compute_gates(path, expect_d, t0=None, t1=None):
    """Core gate computation, importable by e1_analyze.py so the per-rep
    gate judgement used for the decision rule is the SAME code as this
    script's own CLI output (no re-parsing of tee'd stdout text)."""
    expect_p = D_TO_PREFILL.get(expect_d)
    if expect_p is None:
        raise ValueError(f"unknown D={expect_d}, not in {sorted(D_TO_PREFILL)}")

    n_bench = 0
    n_prefill_active = 0
    n_decode_active = 0
    n_both_active = 0
    realized_pair = collections.Counter()   # PRIMARY: histogram over the prefill-active population only
    admission_blocked = 0
    block_reasons = collections.Counter()
    p_batches, d_batches = [], []
    rows = []   # (ts, p_active, d_active) -- only what time-weighting needs, kept small

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
        if p_active and d_active:
            n_both_active += 1
        if d_active:
            n_decode_active += 1
            d_batches.append(e["decode_running_batch_size"])
        if e.get("prefill_admission_blocked"):
            admission_blocked += 1
            block_reasons[e.get("prefill_admission_block_reason") or "(unset)"] += 1
        if not p_active:
            continue
        # PRIMARY population: prefill genuinely in flight -- the only regime
        # where the target [108-D, D] partition is even supposed to be
        # selected (multiplexing_mixin.py's stream-selection: decode-only
        # windows correctly get the full unpartitioned (0,108) stream, that
        # is not a pin failure). This is an EVENT-conditional proportion
        # (bug #1 fix), intentionally NOT time-weighted -- see module
        # docstring for why that is the semantically correct choice here.
        n_prefill_active += 1
        pair = (e.get("prefill_sms"), e.get("decode_sms"))
        realized_pair[pair] += 1
        p_batches.append(e["prefill_active_batch_size"])

    # --- Bug #2 fix: TIME-WEIGHTED occupancy, over the SAME rows/window ---
    # Each row's state holds from its own ts until the NEXT row's ts (or,
    # for the last in-window row, until t1 if a window bound was given;
    # otherwise that trailing duration is dropped -- see module docstring).
    rows.sort(key=lambda r: r[0])
    t_total = t_prefill = t_decode = t_both = 0.0
    for i, (ts, p, d) in enumerate(rows):
        nxt = rows[i + 1][0] if i + 1 < len(rows) else t1
        if nxt is None:
            continue   # unbounded whole-file check, no next row: drop trailing duration
        dt = max(0.0, nxt - ts)
        t_total += dt
        if p:
            t_prefill += dt
        if d:
            t_decode += dt
        if p and d:
            t_both += dt

    out = dict(
        expect_p=expect_p, expect_d=expect_d, n_bench=n_bench,
        n_prefill_active=n_prefill_active, n_decode_active=n_decode_active,
        n_both_active=n_both_active,
        pin_frac=(realized_pair[(expect_p, expect_d)] / n_prefill_active) if n_prefill_active else float("nan"),
        # MANDATORY diagnostic (bug #2 fix, 2026-07-29): time-weighted, not count-weighted.
        prefill_active_time_frac=(t_prefill / t_total) if t_total else float("nan"),
        decode_active_time_frac=(t_decode / t_total) if t_total else float("nan"),
        concurrent_time_frac=(t_both / t_total) if t_total else float("nan"),
        t_total_s=t_total,
        # RETAINED for audit-trail/comparison ONLY -- do not interpret these as
        # wall-clock shares, that inference was bug #2 (see module docstring).
        prefill_active_frac_count_based=(n_prefill_active / n_bench) if n_bench else float("nan"),
        concurrent_frac_count_based=(n_both_active / n_bench) if n_bench else float("nan"),
        co_resident_frac_of_prefill_active=(n_both_active / n_prefill_active) if n_prefill_active else float("nan"),
        admission_blocked_frac=(admission_blocked / n_bench) if n_bench else float("nan"),
        block_reasons=dict(block_reasons),
        realized_pair=dict(realized_pair),
        prefill_batch_mean=(st.fmean(p_batches) if p_batches else float("nan")),
        decode_batch_mean=(st.fmean(d_batches) if d_batches else float("nan")),
    )
    return out


def gate_verdict(g, min_pin_frac=0.80, min_n_prefill_active=DEFAULT_MIN_N_PREFILL_ACTIVE,
                  concurrency_low_threshold=0.05):
    """PASS/FAIL from compute_gates() output, single source of truth for
    both the CLI printer and e1_analyze.py's cell/rep filtering.

    `pin_pass` is the ONLY hard, cite-blocking criterion (DESIGN.md sec 5,
    2026-07-28 revision): prefill-active population big enough to be
    informative (n >= min_n_prefill_active) AND the realized pair matches
    target at rate >= min_pin_frac among that population.
    `concurrency_low` is a WARNING flag only (no principled hard threshold
    exists yet -- coordinator directive: report, do not silently gate), now
    judged on the TIME-WEIGHTED `concurrent_time_frac` (2026-07-29 fix;
    was count-based and badly understated true concurrency -- see module
    docstring bug #2). `concurrency_low_threshold` default 0.05 is a
    placeholder judgment call, not data-derived; DESIGN.md sec 9.7 tracks
    this as an open choice."""
    if not g["n_bench"]:
        return dict(pin_pass=False, underpowered=True, concurrency_low=True,
                     reason="no benchmark snapshots")
    underpowered = g["n_prefill_active"] < min_n_prefill_active
    conc = g.get("concurrent_time_frac", float("nan"))
    concurrency_low = (conc == conc) and conc < concurrency_low_threshold
    if underpowered:
        return dict(pin_pass=False, underpowered=True,
                     concurrency_low=concurrency_low,
                     reason=f"UNDERPOWERED: only {g['n_prefill_active']} prefill-active "
                            f"samples (< {min_n_prefill_active}) -- frac threshold is not "
                            "informative at this sample size")
    pin_pass = g["pin_frac"] >= min_pin_frac
    return dict(pin_pass=pin_pass, underpowered=False,
                 concurrency_low=concurrency_low,
                 reason="" if pin_pass else "wrong SM")


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

    path = argv[0]
    expect_d = int(argv[1])
    min_pin_frac = float(argv[2]) if len(argv) > 2 else 0.80
    min_n_prefill_active = int(argv[3]) if len(argv) > 3 else DEFAULT_MIN_N_PREFILL_ACTIVE
    try:
        g = compute_gates(path, expect_d, t0, t1)
    except ValueError as exc:
        print(f"UNKNOWN_D {exc} (add it to D_TO_PREFILL if the grid changes)")
        return
    v = gate_verdict(g, min_pin_frac, min_n_prefill_active)
    expect_p = g["expect_p"]

    window = f" window=[{t0},{t1}]" if (t0 is not None or t1 is not None) else ""
    print(f"E1_PIN_CHECK D={expect_d}(P{expect_p}){window}: "
          f"n_bench_snapshots={g['n_bench']} n_prefill_active={g['n_prefill_active']} "
          f"n_decode_active={g['n_decode_active']} n_both_active={g['n_both_active']}")
    if not g["n_bench"]:
        print(f"E1_PIN_GATE D={expect_d}: FAIL -> no benchmark-phase snapshots -> INVALID")
        return

    # MANDATORY diagnostic (coordinator 2026-07-28/re-fixed 2026-07-29: report
    # always, TIME-WEIGHTED not count-weighted -- see module docstring bug #2).
    # Not folded into the pass/fail gate because concurrency_low_threshold is
    # a judgment call, not data-derived (DESIGN.md sec 9.7).
    print(f"E1_CONCURRENCY_DIAGNOSTIC(time-weighted, window={g['t_total_s']:.2f}s): "
          f"concurrent_time_frac={g['concurrent_time_frac']:.4f} "
          f"prefill_active_time_frac={g['prefill_active_time_frac']:.4f} "
          f"decode_active_time_frac={g['decode_active_time_frac']:.4f}"
          + ("  <-- LOW: D-axis only weakly exercised in this window, interpret any pin PASS with this caveat"
             if v["concurrency_low"] else ""))
    print(f"E1_CONCURRENCY_DIAGNOSTIC(COUNT-BASED, retained for audit-trail ONLY -- "
          f"DO NOT read as a wall-clock share, see module docstring bug #2): "
          f"concurrent_frac_count_based={g['concurrent_frac_count_based']:.4f} "
          f"prefill_active_frac_count_based={g['prefill_active_frac_count_based']:.4f}")

    if g["n_prefill_active"] == 0:
        print(f"E1_PIN_GATE D={expect_d}: FAIL -> no prefill-active snapshots -> INVALID "
              "(cannot judge partition pin without prefill in flight)")
        return

    dist = ", ".join(f"P{p}/D{d}:{v_}({v_/g['n_prefill_active']*100:.0f}%)"
                      for (p, d), v_ in sorted(g["realized_pair"].items(), key=lambda kv: -kv[1]))
    print(f"E1_REALIZED_hist(prefill-active n={g['n_prefill_active']}): {dist}")
    if v["underpowered"]:
        pin_verdict = f"FAIL -> {v['reason']}"
    else:
        pin_verdict = "PASS" if v["pin_pass"] else \
            f"FAIL -> realized pin {g['pin_frac']:.3f} < {min_pin_frac} -> measured at the wrong SM, INVALID"
    print(f"E1_PIN_GATE D={expect_d}(P{expect_p}): {pin_verdict} (frac={g['pin_frac']:.3f}, "
          f"n={g['n_prefill_active']}, min_n={min_n_prefill_active})")

    print(f"E1_CO_RESIDENCY(prefill-active): frac={g['co_resident_frac_of_prefill_active']:.3f} "
          "(share of the PRIMARY prefill-active population that ALSO had decode active -- "
          "informational, not the same as the concurrency diagnostic above which uses ALL "
          "bench snapshots as the denominator)")
    if g["prefill_batch_mean"] == g["prefill_batch_mean"]:
        print(f"E1_PREFILL_BATCH(prefill-active): mean={g['prefill_batch_mean']:.2f}")
    if g["decode_batch_mean"] == g["decode_batch_mean"]:
        print(f"E1_DECODE_BATCH(decode-active): mean={g['decode_batch_mean']:.2f}")
    ab_flag = "  <-- elevated, see reports/r2_decoupling_review_2026-07-24.md #4 caveat in this file's docstring" \
        if g["admission_blocked_frac"] > 0.10 else ""
    print(f"E1_ADMISSION_BLOCKED(all bench snapshots): frac={g['admission_blocked_frac']:.3f} "
          f"reasons={g['block_reasons']}{ab_flag}")


if __name__ == "__main__":
    main()
