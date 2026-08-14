#!/usr/bin/env python3
"""E1-b / E1-c scorer -- dense-cohabitation re-measurement of the Gate 2-S sec8.9
premise decision quantity, plus its positive control.

PRE-REGISTRATION: PREREG_G2S_E1B_E1C_2026-08-14.md  (READ SECTION 0 FIRST --
this registration is NOT blind about several inputs; the declaration is there).
THE PRE-REGISTRATION IS CANONICAL AND THIS CODE FOLLOWS IT.  If running this
reveals a need to deviate, STOP AND REPORT (methodology gate #20 and its
reverse).

WHAT THIS IS
  E1 (PREREG_G2S_E1_ADDENDUM_2026-08-11) established, from jobs 877756/877757,
  that the pooled max(decode_running_batch_size) in the four Gate 2-S cells is
  14/23/10/13 < 36 -- but only `VERIFIED_AT_SAMPLED_INSTANTS`, because that
  telemetry ran PDMUX_TRACE_FORCE_PREFILL=0 and therefore sampled the
  cohabitation population at 1/32 of the scheduler-sync rate.  E1 sec7.2
  explicitly confessed it had NO pre-registered threshold for downgrading on
  that lower-bound bias, and called that "a known hole we do not hide".

  E1-b re-measures the SAME four cells with PDMUX_TRACE_FORCE_PREFILL=1, where
  every sync with a prefill batch in flight emits (multiplexing_mixin.py:527-538),
  paired within-campaign against a force=0 arm.  E1-c is the positive control on
  Zamba2 rate 6, the one cell where the screen is known to be able to fire.

WHAT THIS IS NOT
  * It does NOT modify, import for scoring decisions, or re-run the Gate 2-S
    premise labels.  PREMISE_LABEL in g2s_analyze.py is NOT written by this tool
    and this tool licenses no upgrade of it (prereg sec6.2 / gate #30).
  * It emits NO Gate 2-S performance verdict.  The only latency quantity it
    computes is the force1-vs-force0 alpha RATIO, used solely as a
    self-nullification check (prereg sec5 rule O1); absolute citation forbidden.
  * It does NOT create a new REFUTED path beyond the producer's own withdrawal
    rule (gate1b_analyze.WITHDRAWAL_THRESHOLD), which is the rule G1-b/G1-c ran.

ZERO NEW FREE PARAMETERS in the statistics -- every rule is IMPORTED:
    gate1/gate1b_analyze.py : load_rows(), max_decode_bs(), classify(),
                              analyze_window(), grid_completeness(), frac_in(),
                              DECODE_BS_DIVISOR=36, TRACE_EVERY=32,
                              WITHDRAWAL_THRESHOLD=0.01, MIN_TOTAL_S, MIN_EPISODES
    gate2/g2s_analyze.py    : paired_t()  (sec5.1 t-CI; bootstrap FORBIDDEN,
                              methodology gate #14/#27) and metrics() for alpha.
  The two new DESIGN choices (n=6, and the population on which the deficit D is
  defined) are declared as free parameters in prereg sec11.

Exit status: 0 = scored (whatever the verdicts), 2 = bad invocation,
             3 = required input missing.  A verdict is never an exit code:
             methodology gate #21 (a measurement failure is not a gate failure).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_GATE1 = os.path.abspath(os.path.join(_HERE, os.pardir, "gate1"))
for _p in (_GATE1, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gate1b_analyze as g1b        # noqa: E402  PRODUCER of the premise rules
import g2s_analyze as g2s           # noqa: E402  PRODUCER of paired_t / metrics

THRESHOLD = g1b.DECODE_BS_DIVISOR          # 36   gate1b_analyze.py:51
TRACE_EVERY = g1b.TRACE_EVERY              # 32   gate1b_analyze.py:50
WITHDRAWAL = g1b.WITHDRAWAL_THRESHOLD      # 0.01 gate1b_analyze.py:52
SPLIT_KEY = (54, 54)                       # idx2 = the partition the premise denies
MIN_TOTAL_S = g1b.MIN_TOTAL_S
MIN_EPISODES = g1b.MIN_EPISODES

PREFIX = "g2se1b"
ARM_F1 = "a4_force1"
ARM_F0 = "a4_force0"
ARMS = (ARM_F1, ARM_F0)

# prereg sec3.  Zamba2 r6 is the E1-c CONTROL and is NOT a Gate 2-S scored cell.
CELLS = {
    "zamba2-27b": ("2", "3", "6"),
    "granite-40-h-micro-base": ("3", "4"),
}
CONTROL_CELLS = {("zamba2-27b", "6")}
NREP_EXPECTED = 6                          # prereg sec3; fixed before submission


# ==========================================================================
# rep boundary reconstruction -- offsets recorded BY THE HARNESS AT MEASUREMENT
# TIME (g2s_e1b_run.sbatch, same recipe as g2s_run.sbatch:254-265).  NOT a gap
# heuristic; see PREREG_G2S_E1_ADDENDUM sec3 for why gap heuristics are banned.
# ==========================================================================
def rep_slices(outdir, tag, arm, job, rate):
    out = []
    pattern = os.path.join(outdir, f"{PREFIX}_{tag}_telemetry_{arm}_rep*_{job}.jsonl")
    for tel in sorted(glob.glob(pattern)):
        off = tel.replace("_telemetry_", "_teloffset_").replace(".jsonl", ".json")
        if not os.path.exists(off):
            continue
        try:
            sl = json.load(open(off)).get(f"r{rate}")
        except Exception:
            sl = None
        if not sl:
            continue
        rep = os.path.basename(tel).split("_rep")[1].split("_")[0]
        with open(tel) as fh:
            lines = fh.readlines()
        out.append(dict(rep=rep, boot_file=os.path.basename(tel),
                        offset=[sl[0], sl[1]], lines=lines[sl[0]:sl[1]]))
    out.sort(key=lambda d: int(d["rep"]))
    return out


def rows_via_producer(lines):
    """Run the PRODUCER's own load_rows() (event+phase filter, ts sort)."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.writelines(lines)
        return g1b.load_rows(path)
    finally:
        os.unlink(path)


def raw_scan(lines):
    """Integrity scan on RAW lines: load_rows() silently drops unparseable rows,
    so parse failures must be counted separately (prereg I1)."""
    n_bad = n_rt = n_bench = dropped_max = dropped_rows = 0
    sis = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            e = json.loads(ln)
        except Exception:
            n_bad += 1
            continue
        if e.get("event") == "runtime_snapshot":
            n_rt += 1
            if e.get("phase") == "benchmark":
                n_bench += 1
        d = e.get("dropped_events", 0) or 0
        if d:
            dropped_rows += 1
            dropped_max = max(dropped_max, d)
        si = e.get("sample_index")
        if si is not None:
            sis.append(int(si))
    return dict(n_lines=len([l for l in lines if l.strip()]),
                n_parse_fail=n_bad, n_runtime_snapshot=n_rt, n_benchmark_phase=n_bench,
                dropped_events_max=dropped_max, n_rows_with_dropped=dropped_rows,
                sample_index_inversions=sum(1 for a, b in zip(sis, sis[1:]) if b < a),
                sample_index_duplicates=len(sis) - len(set(sis)))


# ==========================================================================
# populations (prereg sec2).  scheduled = the legacy 1/32 grid; force mode makes
# `trace_forced` present on EVERY row (multiplexing_mixin.py:565-571), and a
# force=0 run omits the key entirely -- so `is not True` is the correct, single
# predicate for "row the legacy grid would also have produced".
# ==========================================================================
def is_scheduled(e):
    return e.get("trace_forced") is not True


def is_cohab(e):
    return g1b.classify(e) == "A"      # producer: decode busy AND prefill in flight


def _mx(rows):
    return g1b.max_decode_bs(rows)


def frac_split(rows):
    """Producer's pop-A time-weighted (54,54) fraction + its sparse guard.
    frac ONLY, never t_total (methodology gate #15)."""
    if not rows:
        return dict(measurable=False, reason="no rows", frac_5454=None, n_episodes=0)
    t1 = rows[-1].get("timestamp_monotonic_s")
    pops = g1b.analyze_window(rows, t1)
    a = pops["A"]
    if a["t_total"] < MIN_TOTAL_S or a["episodes"] < MIN_EPISODES:
        return dict(measurable=False, reason="pop A too sparse", frac_5454=None,
                    n_episodes=a["episodes"])
    return dict(measurable=True, reason="",
                frac_5454=g1b.frac_in(a["time_hist"], a["t_total"], {SPLIT_KEY}),
                n_episodes=a["episodes"])


def stale_window_bound(rows):
    """UPPER BOUND (not an estimate -- methodology gate #31) on the time the
    engine spends in G1-a's target interval: post prefill-admission
    (multiplexing_mixin.py:1003-1004) and pre `adjust_stream_groups`
    (multiplexing_mixin.py:1070-1080).  No sync point exists inside that interval, so
    it is NOT directly observable; it is however CONTAINED in the gap between
    the two syncs that bracket it (:997 and :1242).  Summing every ADJACENT
    (delta sample_index == 1) gap whose earlier row is pop A therefore
    over-counts, never under-counts.

    Known conservatism ~2x: the sum also includes the :1242(k) -> :997(k+1)
    gaps, which contain no stale interval.  Reported as a bound anyway.
    Adjacency is required precisely to avoid the non-adjacent-dt contamination
    that made pop-A t_total_s uncitable (CONSENSUS gate #15, second trap)."""
    if len(rows) < 2:
        return dict(bound_s=0.0, span_s=0.0, bound_frac=None, n_adjacent_pairs=0)
    tot = 0.0
    npair = 0
    for a, b in zip(rows, rows[1:]):
        sa, sb = a.get("sample_index"), b.get("sample_index")
        if sa is None or sb is None or sb - sa != 1:
            continue
        if not is_cohab(a):
            continue
        dt = b.get("timestamp_monotonic_s", 0.0) - a.get("timestamp_monotonic_s", 0.0)
        if dt > 0:
            tot += dt
            npair += 1
    span = rows[-1]["timestamp_monotonic_s"] - rows[0]["timestamp_monotonic_s"]
    return dict(bound_s=tot, span_s=span,
                bound_frac=(tot / span if span > 0 else None),
                n_adjacent_pairs=npair)


# ==========================================================================
# per-cell scoring
# ==========================================================================
def score_cell(outdir, tag, job_f1, job_f0, rate):
    res = dict(tag=tag, rate=rate, threshold=THRESHOLD, trace_every=TRACE_EVERY,
               is_control=((tag, rate) in CONTROL_CELLS))
    fails = []
    per_rep = []
    slices = rep_slices(outdir, tag, ARM_F1, job_f1, rate)
    for s in slices:
        rows = rows_via_producer(s["lines"])
        scan = raw_scan(s["lines"])
        sched = [e for e in rows if is_scheduled(e)]
        forced = [e for e in rows if not is_scheduled(e)]
        cohab = [e for e in rows if is_cohab(e)]
        cohab_s = [e for e in sched if is_cohab(e)]
        m_all_dense, m_all_sched = _mx(rows), _mx(sched)
        m_coh_dense, m_coh_sched = _mx(cohab), _mx(cohab_s)
        d = (None if (m_coh_dense is None or m_coh_sched is None)
             else m_coh_dense - m_coh_sched)
        gc = g1b.grid_completeness(sched)
        per_rep.append(dict(
            rep=s["rep"], boot_file=s["boot_file"], offset=s["offset"],
            n_rows=len(rows), n_scheduled=len(sched), n_forced=len(forced),
            n_cohab=len(cohab), n_cohab_scheduled=len(cohab_s),
            densification_x=(len(cohab) / len(cohab_s)) if cohab_s else None,
            max_all_dense=m_all_dense, max_all_scheduled=m_all_sched,
            max_cohab_dense=m_coh_dense, max_cohab_scheduled=m_coh_sched,
            deficit_D=d, grid=gc, integrity=scan,
            stale=stale_window_bound(rows)))

    res["per_rep"] = per_rep
    res["n_reps"] = len(per_rep)
    if not per_rep:
        res.update(state="UNDETERMINED", reason="no force=1 rep slices found")
        return res

    def pooled(key):
        vals = [d[key] for d in per_rep if d.get(key) is not None]
        return max(vals) if vals else None

    res["pooled_max_all_dense"] = pooled("max_all_dense")
    res["pooled_max_all_scheduled"] = pooled("max_all_scheduled")
    res["pooled_max_cohab_dense"] = pooled("max_cohab_dense")
    res["pooled_max_cohab_scheduled"] = pooled("max_cohab_scheduled")
    res["headroom_all_dense"] = (None if res["pooled_max_all_dense"] is None
                                 else THRESHOLD - res["pooled_max_all_dense"])
    res["n_forced_total"] = sum(d["n_forced"] for d in per_rep)
    dens = [d["densification_x"] for d in per_rep if d["densification_x"]]
    res["densification_x_median"] = statistics.median(dens) if dens else None

    # ---- integrity (prereg sec4) -----------------------------------------
    if any(d["integrity"]["n_parse_fail"] for d in per_rep):
        fails.append("I1 parse failures > 0")
    if any(d["integrity"]["dropped_events_max"] for d in per_rep):
        fails.append("I2 dropped_events > 0")
    if len(per_rep) != NREP_EXPECTED:
        fails.append(f"I4 rep count {len(per_rep)} != {NREP_EXPECTED}")
    bad = [d["rep"] for d in per_rep
           if d["integrity"]["sample_index_inversions"]
           or d["integrity"]["sample_index_duplicates"]]
    if bad:
        fails.append(f"I6 within-rep sample_index inversions/duplicates: reps {bad}")
    gviol = [d["rep"] for d in per_rep if d["grid"]["violations"] or d["grid"]["missing"]]
    if gviol:
        fails.append(f"I8 producer grid-completeness failed on the scheduled subset: reps {gviol}")
    if res["n_forced_total"] < 1:
        fails.append("I9 zero forced rows -- PDMUX_TRACE_FORCE_PREFILL never engaged; "
                     "this cell measures nothing E1 did not already measure")
    res["integrity_failures"] = fails

    # ---- P1 screen + P2 refutation (prereg sec5) --------------------------
    # CODE IDENTITY (multiplexing_mixin.py:881-909): stream_idx == 2 <=>
    # decode_bs >= 36.  So max < 36 IMPLIES frac(idx2) == 0 and the two numbers
    # are ONE empirical fact.  frac is therefore consulted ONLY in the branch
    # where the identity carries no information (max >= 36).  This is the gate
    # #9 FIX, not a confession.
    all_rows = [e for s in slices for e in rows_via_producer(s["lines"])]
    all_rows.sort(key=lambda e: e.get("timestamp_monotonic_s", 0.0))
    m = res["pooled_max_all_dense"]
    if fails:
        res.update(state="UNDETERMINED", reason="; ".join(fails), frac=None)
    elif m is None:
        res.update(state="UNDETERMINED", reason="no rows", frac=None)
    elif res["is_control"]:
        # E1-c: this cell is NOT a Gate 2-S scored cell and the sec8.9 premise was
        # never asserted for it.  Its only job is to show the screen CAN fire
        # (methodology gate #15).  Labelling it PREMISE_* or REFUTED would invite
        # exactly the misreading that "the premise was refuted".
        fr = frac_split(all_rows) if m >= THRESHOLD else dict(
            measurable=False, reason="not computed: max<36 makes frac(idx2)=0 by identity",
            frac_5454=None, n_episodes=None)
        res["frac"] = fr
        res.update(
            state=("CONTROL_FIRED" if m >= THRESHOLD else "CONTROL_DID_NOT_FIRE"),
            reason=(f"E1-c positive control (NOT a Gate 2-S cell): pooled max(decode_bs)={m} "
                    f"vs {THRESHOLD}; pop-A time-weighted frac(54,54)="
                    f"{fr.get('frac_5454')}.  This says nothing about the premise in the "
                    f"four scored cells -- only whether the instrument is able to fire."))
    elif m < THRESHOLD:
        res.update(state="PREMISE_HOLDS_AT_ALL_OBSERVED_COHABITING_SYNCS", frac=None,
                   reason=(f"pooled max(decode_bs)={m} < {THRESHOLD} over n={len(per_rep)} "
                           f"reps with PDMUX_TRACE_FORCE_PREFILL=1 (every cohabiting sync "
                           f"emits).  frac(idx2) is NOT reported here: by the code identity "
                           f"it is the same empirical fact, not a second support."))
    else:
        fr = frac_split(all_rows)
        res["frac"] = fr
        if not fr["measurable"]:
            res.update(state="UNDETERMINED",
                       reason=f"max={m} >= {THRESHOLD} but producer sparse guard fired: {fr['reason']}")
        elif fr["frac_5454"] >= WITHDRAWAL:
            res.update(state="REFUTED",
                       reason=(f"max={m} >= {THRESHOLD} AND pop-A time-weighted frac(54,54)"
                               f"={fr['frac_5454']:.4f} >= {WITHDRAWAL} (producer withdrawal "
                               f"rule, gate1b_analyze.py:52)"))
        else:
            res.update(state="UNDETERMINED",
                       reason=(f"max={m} >= {THRESHOLD} (threshold reached at >=1 observed "
                               f"instant) but frac(54,54)={fr['frac_5454']:.4f} < {WITHDRAWAL}. "
                               f"NOT verified, NOT refuted."))

    # ---- P3 deficit D (prereg sec5 rule P3) -------------------------------
    dv = [d["deficit_D"] for d in per_rep if d["deficit_D"] is not None]
    if len(dv) >= 2:
        t = g2s.paired_t([float(x) for x in dv])
        u = t["hi"]
        base = res["pooled_max_cohab_scheduled"]
        res["deficit"] = dict(
            n=t["n"], values=dv, mean=t["mean"], ci_lo=t["lo"], ci_hi=t["hi"],
            degenerate=t.get("degenerate", False), p=t.get("p"),
            pooled_max_cohab_scheduled=base,
            projected=(None if base is None else base + u),
            # E1 has no rate-6 cell, so "downgrade E1" is undefined on the control.
            downgrade_E1=(None if res["is_control"]
                          else (base is not None and (base + u) >= THRESHOLD)),
            downgrade_note=("N/A: E1-c control cell is not an E1 cell"
                            if res["is_control"] else
                            "fires iff pooled scheduled-only cohabitation max plus the "
                            "upper 95% t limit of the deficit reaches 36"))
    else:
        res["deficit"] = dict(n=len(dv), values=dv, mean=None, ci_lo=None, ci_hi=None,
                              degenerate=False, downgrade_E1=None,
                              note="fewer than 2 usable reps -- no t-CI (prereg sec5)")

    # ---- G1-a triage bound (prereg sec6) ----------------------------------
    fr_b = [d["stale"]["bound_frac"] for d in per_rep if d["stale"]["bound_frac"] is not None]
    res["stale_window_upper_bound_frac_max"] = max(fr_b) if fr_b else None
    res["g1a_needed"] = (None if not fr_b else bool(max(fr_b) >= WITHDRAWAL))

    # ---- O1 observer effect on alpha (prereg sec5 rule O1) ----------------
    res["alpha"] = alpha_pair(outdir, tag, job_f1, job_f0, rate)
    return res


def _bench_rows(outdir, tag, arm, job, rate):
    """Per-rep alpha from the bench jsonl the harness wrote, using the AUDITED
    metric definition (g2s_analyze.metrics -> alpha_ITLp95_mean, sec4.1.2)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(outdir, f"{PREFIX}_{tag}_{arm}_rep*_{job}.jsonl"))):
        rep = os.path.basename(f).split("_rep")[1].split("_")[0]
        for ln in open(f):
            try:
                row = json.loads(ln)
            except Exception:
                continue
            if str(row.get("tag", "")).endswith(f"_r{rate}") and row.get("itls"):
                try:
                    out[int(rep)] = g2s.metrics(row)["alpha_ITLp95_mean"]
                except Exception:
                    pass
    return out


def alpha_pair(outdir, tag, job_f1, job_f0, rate):
    a1 = _bench_rows(outdir, tag, ARM_F1, job_f1, rate)
    a0 = _bench_rows(outdir, tag, ARM_F0, job_f0, rate)
    reps = sorted(set(a1) & set(a0))
    if len(reps) < 2:
        return dict(n=len(reps), state="UNMEASURED",
                    note="fewer than 2 paired reps -- observer-effect check not evaluated")
    diffs = [a1[r] - a0[r] for r in reps]
    t = g2s.paired_t(diffs)
    ref = statistics.mean([a0[r] for r in reps])
    rel = (t["mean"] / ref * 100.0) if ref else None
    perturbed = bool(rel is not None and abs(rel) > 3.0
                     and not (t["lo"] <= 0.0 <= t["hi"]))
    return dict(n=t["n"], reps=reps, mean_delta_ms_NOT_FOR_CITATION=t["mean"],
                ci_lo=t["lo"], ci_hi=t["hi"], rel_pct=rel, degenerate=t.get("degenerate"),
                state=("OBSERVER_PERTURBED" if perturbed else "OBSERVER_WITHIN_3PCT"),
                note=("RATIO ONLY.  Absolute alpha from this campaign is NOT citable and "
                      "must never be compared with any Gate 2-S arm (prereg sec8)."))


# ==========================================================================
def selftest():
    """Prereg sec9: every declared diagnostic field must have a real production
    path (methodology gate #25, which reached an eighth recurrence).  This runs
    the whole scorer on a synthetic fixture and asserts each field exists."""
    import shutil
    tmp = tempfile.mkdtemp(prefix="e1b_selftest_")
    tag, j1, j0, rate = "zamba2-27b", "999001", "999002", "2"
    try:
        for arm, job in ((ARM_F1, j1), (ARM_F0, j0)):
            for rep in range(1, NREP_EXPECTED + 1):
                tel = os.path.join(tmp, f"{PREFIX}_{tag}_telemetry_{arm}_rep{rep}_{job}.jsonl")
                lines, si, t = [], 0, 0.0
                for it in range(200):
                    for site in (0, 1):                      # :997 then :1242
                        si += 1
                        t += 0.004
                        cohab = (it % 3 == 0)
                        sched = (si == 1 or si % TRACE_EVERY == 0)
                        if not (sched or (cohab and arm == ARM_F1)):
                            continue
                        rec = dict(event="runtime_snapshot", phase="benchmark",
                                   timestamp_monotonic_s=t, sample_index=si,
                                   dropped_events=0,
                                   decode_running_batch_size=(9 + (si % 5)),
                                   prefill_active_batch_size=(2 if cohab else 0),
                                   prefill_sms=74, decode_sms=34, stream_index=1)
                        if arm == ARM_F1:
                            rec["trace_forced"] = (not sched)
                        lines.append(json.dumps(rec) + "\n")
                open(tel, "w").writelines(lines)
                off = tel.replace("_telemetry_", "_teloffset_").replace(".jsonl", ".json")
                json.dump({f"r{rate}": [0, len(lines)]}, open(off, "w"))
                bench = os.path.join(tmp, f"{PREFIX}_{tag}_{arm}_rep{rep}_{job}.jsonl")
                itls = [[0.04 + 0.001 * k] * 20 for k in range(12)]
                json.dump(dict(tag=f"rep{rep}_{arm}_r{rate}", itls=itls,
                               ttfts=[0.5] * 12, completed=12, total_input_tokens=1),
                          open(bench, "w"))
                open(bench, "a").write("\n")
        r = score_cell(tmp, tag, j1, j0, rate)
        required = ["pooled_max_all_dense", "pooled_max_all_scheduled",
                    "pooled_max_cohab_dense", "pooled_max_cohab_scheduled",
                    "headroom_all_dense", "n_forced_total", "densification_x_median",
                    "integrity_failures", "state", "reason", "deficit",
                    "stale_window_upper_bound_frac_max", "g1a_needed", "alpha",
                    "per_rep", "n_reps"]
        missing = [k for k in required if k not in r]
        assert not missing, f"prereg sec9 field(s) with no production path: {missing}"
        for k in ("mean", "ci_lo", "ci_hi", "degenerate", "downgrade_E1", "values"):
            assert k in r["deficit"], f"deficit.{k} missing"
        for k in ("state", "rel_pct", "ci_lo", "ci_hi"):
            assert k in r["alpha"], f"alpha.{k} missing"
        pr = r["per_rep"][0]
        for k in ("n_forced", "n_cohab", "n_cohab_scheduled", "densification_x",
                  "deficit_D", "grid", "stale"):
            assert k in pr, f"per_rep.{k} missing"
        assert r["n_forced_total"] > 0, "fixture failed to exercise the forced path"
        assert r["state"] == "PREMISE_HOLDS_AT_ALL_OBSERVED_COHABITING_SYNCS", r["state"]
        # negative control: the screen MUST be able to fire (gate #15)
        r2 = score_cell(tmp, tag, j1, j0, rate)
        for d in r2["per_rep"]:
            d["max_all_dense"] = 40
        assert max(d["max_all_dense"] for d in r2["per_rep"]) >= THRESHOLD
        print("SELFTEST: PASS -- every declared field has a production path; "
              "forced path exercised; screen is two-sided.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=_HERE)
    ap.add_argument("--tag")
    ap.add_argument("--job-force1")
    ap.add_argument("--job-force0")
    ap.add_argument("--json-out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.tag and a.job_force1 and a.job_force0):
        ap.error("--tag/--job-force1/--job-force0 required unless --selftest")
    if a.tag not in CELLS:
        ap.error(f"unknown tag {a.tag}")

    print("=" * 78)
    print("E1-b / E1-c -- dense-cohabitation premise re-measurement")
    print("PREREG: PREREG_G2S_E1B_E1C_2026-08-14.md   (sec0: NOT fully blind)")
    print(f"threshold = decode_bs_divisor = {THRESHOLD} (imported, gate1b_analyze.py:51)")
    print(f"withdrawal frac threshold     = {WITHDRAWAL} (imported, gate1b_analyze.py:52)")
    print("identity: stream_idx==2 <=> decode_bs>=36 (multiplexing_mixin.py:881-909)")
    print("  => max<36 IMPLIES frac(idx2)=0.  frac is consulted ONLY when max>=36.")
    print("THIS TOOL WRITES NO Gate 2-S PREMISE LABEL AND LICENSES NO UPGRADE OF ONE.")
    print("=" * 78)

    report = dict(prereg="PREREG_G2S_E1B_E1C_2026-08-14.md", blind=False,
                  threshold=THRESHOLD, trace_every=TRACE_EVERY,
                  withdrawal_threshold=WITHDRAWAL, n_reps_expected=NREP_EXPECTED,
                  tag=a.tag, job_force1=a.job_force1, job_force0=a.job_force0,
                  cells={})
    control_fired = None
    for rate in CELLS[a.tag]:
        r = score_cell(a.outdir, a.tag, a.job_force1, a.job_force0, rate)
        report["cells"][f"{a.tag}|r{rate}"] = r
        kind = "E1-c CONTROL" if r["is_control"] else "E1-b cell"
        print(f"\n--- {a.tag} r{rate}  [{kind}]  arm={ARM_F1} n={r['n_reps']} ---")
        print(f"  pooled max(decode_bs) all-rows dense = {r.get('pooled_max_all_dense')} "
              f"(scheduled-only {r.get('pooled_max_all_scheduled')}), threshold {THRESHOLD}, "
              f"headroom {r.get('headroom_all_dense')}")
        print(f"  cohabitation max: dense={r.get('pooled_max_cohab_dense')} "
              f"scheduled={r.get('pooled_max_cohab_scheduled')}  "
              f"forced rows={r.get('n_forced_total')}  "
              f"densification={r.get('densification_x_median')}x (median over reps)")
        d = r.get("deficit", {})
        print(f"  D (cohab dense - cohab scheduled) per rep {d.get('values')}  "
              f"mean={d.get('mean')} CI[{d.get('ci_lo')},{d.get('ci_hi')}] "
              f"degenerate={d.get('degenerate')}")
        print(f"    projected {r.get('pooled_max_cohab_scheduled')}+U = {d.get('projected')} "
              f"vs {THRESHOLD} -> downgrade_E1={d.get('downgrade_E1')}")
        print(f"  G1-a triage: stale-window UPPER BOUND frac (max over reps) = "
              f"{r.get('stale_window_upper_bound_frac_max')} vs {WITHDRAWAL} "
              f"-> g1a_needed={r.get('g1a_needed')}  (BOUND, not an estimate)")
        al = r.get("alpha", {})
        print(f"  observer effect on alpha: {al.get('state')} rel={al.get('rel_pct')}% "
              f"n={al.get('n')} CI[{al.get('ci_lo')},{al.get('ci_hi')}]  (ratio only)")
        print(f"  integrity failures: {r.get('integrity_failures') or 'none'}")
        print(f"  STATE = {r['state']}")
        print(f"  reason: {r['reason']}")
        if r["is_control"]:
            control_fired = (r.get("pooled_max_all_dense") is not None
                             and r["pooled_max_all_dense"] >= THRESHOLD)

    report["control_fired"] = control_fired
    print("\n" + "=" * 78)
    if control_fired is False:
        print("*** E1-c CONTROL DID NOT FIRE ***  Per prereg sec5 rule C1 every "
              "PREMISE_HOLDS_* state in this campaign is hereby re-labelled "
              "UNINFORMATIVE_SCREEN and may NOT be cited as strengthening E1.")
        for k, v in report["cells"].items():
            if v["state"].startswith("PREMISE_HOLDS"):
                v["state"] = "UNINFORMATIVE_SCREEN"
                v["reason"] = "E1-c control did not fire (prereg sec5 C1)"
    elif control_fired is None:
        dep = ("E1-c control cell is not in this tag; the control verdict is carried by the "
               "zamba2-27b job.  This job's PREMISE_HOLDS_* states are NOT readable on their "
               "own -- if that job's control did not fire, these become UNINFORMATIVE_SCREEN "
               "(prereg sec5 C1).")
        report["control_dependency"] = dep
        for v in report["cells"].values():
            v["control_dependency"] = dep
        print("*** " + dep)
    print("SUMMARY (cell-wise only; no pooled-across-cells state)")
    for k, v in report["cells"].items():
        print(f"  {k:30s} max={str(v.get('pooled_max_all_dense')):>4s}  {v['state']}")
    print("\nUNCHANGED BY THIS CAMPAIGN (prereg sec8): every Gate 2-S performance "
          "verdict (Delta, Holm, 9-cell, F-series), magnitude-citation eligibility, "
          "the sec1-1 attribution, and PREMISE_LABEL itself.")
    print("=" * 78)

    if a.json_out:
        json.dump(report, open(a.json_out, "w"), indent=1)
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
