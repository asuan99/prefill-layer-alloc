#!/usr/bin/env python3
"""Realized-partition gate for the E1 frontier campaign.

★★REVISED 2026-07-28 (coordinator directive, after smoke job 865832 exposed
a gate-definition bug -- see `e1_pin_check_865832_diagnosis.md` in this
directory for the full investigation). The original version of this file
conditioned its population on `engaged = prefill_active_batch_size>0 OR
decode_running_batch_size>0` (git history / diagnosis doc has the exact
before/after). That is WRONG for judging whether the target partition was
realized: `multiplexing_mixin.py`'s stream-selection logic
(`elif not self.running_batch.is_empty(): set_current_stream_idx(last
group)`, i.e. decode-only windows get the FULL unpartitioned (0,108) stream
BY DESIGN, not our [108-D,D] target) means decode-only samples are SUPPOSED
to show (0,108) -- counting them as "should have been at target, wasn't" is
not measuring a failure, it's measuring how rarely prefill happened to be
in flight during a sparse open-loop smoke (NP=20 @ rate 2-4 gave n_engaged
=69 of 21751 bench snapshots for the d16 cell, of which 97% were
legitimately decode-only-(0,108) windows -- see the diagnosis doc). Real
client-side TTFT differed 4.5x between cells (matching s8p_prefill's
independently-measured prefill-SM sensitivity in size and direction), so
the partition WAS being applied; the gate's population was just wrong.

**Corrected design (single source of truth: `compute_gates`/`gate_verdict`,
imported by `e1_analyze.py` so the sbatch's own stdout and the analyzer's
citeability filter never drift apart):**

1. **PRIMARY, cite-blocking gate = REALIZED PARTITION PIN, conditioned on
   `prefill_active_batch_size > 0` ONLY** (mirrors
   `s8p_prefill/prefill_pin_check.py:51,56-58,67` exactly -- decode state is
   irrelevant to this condition, matching that file's own convention for
   the prefill axis). Among prefill-active samples, the fraction whose
   realized `(prefill_sms, decode_sms)` pair equals the cell's target
   `(108-D, D)` must be >= `min_pin_frac` (default 0.80) **AND** the
   prefill-active population itself must have >= `min_n_prefill_active`
   samples (default 20) -- added because a tiny population (n=1 in the
   865832 d16 smoke) can trivially satisfy a frac threshold at either 0%
   or 100% without being informative. A cell/rep with `n_prefill_active <
   min_n_prefill_active` fails with reason "UNDERPOWERED", a distinct
   verdict from "wrong SM" so the two failure modes are never conflated in
   a report.
2. **MANDATORY DIAGNOSTIC (not gating, per coordinator: "그게 낮으면 D 축
   자체가 약해지므로 결과 해석에 필수" -- report always, do not silently
   gate on it since we lack a principled threshold yet):
   `concurrent_frac_of_bench` = fraction of ALL benchmark-phase snapshots
   (not just prefill-active ones) where BOTH `prefill_active_batch_size>0`
   AND `decode_running_batch_size>0` simultaneously. This answers "how much
   of the total experiment wall-clock actually exercises the
   prefill/decode SM contention the D-axis is about" -- a low value means
   the D comparison, even where the pin gate passes, was tested over a
   small slice of the run. Also reported for context:
   `prefill_active_frac_of_bench` (how much wall-clock time even has
   prefill in flight at all).

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
away at the cell level).

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
        # is not a pin failure).
        n_prefill_active += 1
        pair = (e.get("prefill_sms"), e.get("decode_sms"))
        realized_pair[pair] += 1
        p_batches.append(e["prefill_active_batch_size"])

    out = dict(
        expect_p=expect_p, expect_d=expect_d, n_bench=n_bench,
        n_prefill_active=n_prefill_active, n_decode_active=n_decode_active,
        n_both_active=n_both_active,
        pin_frac=(realized_pair[(expect_p, expect_d)] / n_prefill_active) if n_prefill_active else float("nan"),
        concurrent_frac_of_bench=(n_both_active / n_bench) if n_bench else float("nan"),
        prefill_active_frac_of_bench=(n_prefill_active / n_bench) if n_bench else float("nan"),
        co_resident_frac_of_prefill_active=(n_both_active / n_prefill_active) if n_prefill_active else float("nan"),
        admission_blocked_frac=(admission_blocked / n_bench) if n_bench else float("nan"),
        block_reasons=dict(block_reasons),
        realized_pair=dict(realized_pair),
        prefill_batch_mean=(st.fmean(p_batches) if p_batches else float("nan")),
        decode_batch_mean=(st.fmean(d_batches) if d_batches else float("nan")),
    )
    return out


def gate_verdict(g, min_pin_frac=0.80, min_n_prefill_active=DEFAULT_MIN_N_PREFILL_ACTIVE):
    """PASS/FAIL from compute_gates() output, single source of truth for
    both the CLI printer and e1_analyze.py's cell/rep filtering.

    `pin_pass` is the ONLY hard, cite-blocking criterion (DESIGN.md sec 5,
    2026-07-28 revision): prefill-active population big enough to be
    informative (n >= min_n_prefill_active) AND the realized pair matches
    target at rate >= min_pin_frac among that population.
    `concurrency_low` is a WARNING flag only (no principled hard threshold
    exists yet -- coordinator directive: report, do not silently gate)."""
    if not g["n_bench"]:
        return dict(pin_pass=False, underpowered=True, concurrency_low=True,
                     reason="no benchmark snapshots")
    underpowered = g["n_prefill_active"] < min_n_prefill_active
    if underpowered:
        return dict(pin_pass=False, underpowered=True,
                     concurrency_low=(g["concurrent_frac_of_bench"] < 0.01),
                     reason=f"UNDERPOWERED: only {g['n_prefill_active']} prefill-active "
                            f"samples (< {min_n_prefill_active}) -- frac threshold is not "
                            "informative at this sample size")
    pin_pass = g["pin_frac"] >= min_pin_frac
    return dict(pin_pass=pin_pass, underpowered=False,
                 concurrency_low=(g["concurrent_frac_of_bench"] < 0.01),
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

    # MANDATORY diagnostic (coordinator 2026-07-28: report always, this is
    # not folded into the pass/fail gate because we lack a principled
    # threshold for "too low", but a low value materially weakens how much
    # of the run actually tested the D-axis and MUST be cited alongside any
    # result from this cell).
    print(f"E1_CONCURRENCY_DIAGNOSTIC: concurrent_frac_of_bench={g['concurrent_frac_of_bench']:.4f} "
          f"(share of ALL benchmark snapshots with BOTH prefill and decode simultaneously active -- "
          "this is how much of total wall-clock actually exercised the D-axis tradeoff) "
          f"prefill_active_frac_of_bench={g['prefill_active_frac_of_bench']:.4f}"
          + ("  <-- LOW: D-axis only weakly exercised in this window, interpret any pin PASS with this caveat"
             if v["concurrency_low"] else ""))

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
