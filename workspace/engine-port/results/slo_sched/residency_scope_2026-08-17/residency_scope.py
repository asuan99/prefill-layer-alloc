#!/usr/bin/env python3
"""A-2 conditional-label SCOPE SIZE: put the three quoted residency numbers on
one denominator.

WHAT THIS IS NOT
----------------
This script does NOT recompute, re-score or re-interpret any G16 decision
quantity (donor / Delta / Delta_SLO / S_itl / M_itl / gap_upper).  It reads
those only as *given constants* from `G16_RESULTS_2026-08-17.json` for the
task-3 interpretation table.  It introduces NO new estimand: every number it
produces is one of the two canonical residency estimands already in the
repository, evaluated on a stated (denominator, window) pair.

THE TWO CANONICAL ESTIMANDS (both pre-existing, neither invented here)
---------------------------------------------------------------------
E-CNT  `results/s8_scaleup/realized_pin_check.py`  (addendum A-2's own tool):
       benchmark-phase `runtime_snapshot` rows with
       `decode_running_batch_size > 0`; `decode_sms or 108`; the reported
       fraction is a COUNT fraction over that decode-active sample.
       -> this is the denominator behind A-2's "53-68% of decode-active
          samples run at (0,108)".
E-TIM  `benchmarks/pdmux_eval/analyze.py:controller_summary` (H3'-b):
       benchmark-phase `runtime_snapshot` rows carrying `decode_sms`;
       TIME-weighted residency over the whole benchmark span.
       -> this is the denominator behind both the smoke value (4.5-8.1%) and
          the FULL value (0.0767 -> 0.2028) quoted in the handoff.

REPORTING AXES (2x2 + window restriction; each cell states its denominator)
--------------------------------------------------------------------------
             | all benchmark snapshots      | decode-active only
  count      | W_cnt_all                    | W_cnt_da   == E-CNT (canonical)
  time       | W_tim_all == E-TIM (canonic) | W_tim_da
  window     | whole benchmark span         | restricted to the 6 pre-registered
             | (incl. inter-round idle)     | round windows, split LO / HI

CONTROLS
--------
PC-1 (positive, cross-implementation): the canonical tool
     `realized_pin_check.py` is executed as a SUBPROCESS on every telemetry
     file and its printed `REALIZED_hist` / `CO_RESIDENT_frac` are compared to
     this script's single-pass recomputation.  Must match exactly.
PC-2 (positive, cross-implementation): `pdmux_eval.analyze.controller_summary`
     is imported and run on this script's loaded events; compared to the
     harness-written `*_controller_summary.json` (produced at run time on the
     compute node) AND to `residency_fraction_by_boot` in the campaign report.
NC-1 (negative): PC-1 comparator re-run against the WRONG arm SM (D+10) must
     FAIL, and PC-2 comparator re-run against a truncated event list must FAIL.
     Without these, "all controls passed" could be an empty signature
     (methodology gate #44 / lesson item 47).
IC-* (identity checks, gate #40): see `identity_checks()`.

Usage:  python3 residency_scope.py [--out OUT.json] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SLO = os.path.dirname(HERE)                                   # results/slo_sched
ENGINE_PORT = os.path.dirname(os.path.dirname(SLO))           # workspace/engine-port
BENCH = os.path.join(ENGINE_PORT, "benchmarks")
PIN_TOOL = os.path.join(os.path.dirname(SLO), "s8_scaleup", "realized_pin_check.py")
REPORT = os.path.join(SLO, "G16_RESULTS_2026-08-17.json")

sys.path.insert(0, BENCH)
from pdmux_eval.analyze import controller_summary  # noqa: E402  (canonical E-TIM)

RUN_RE = re.compile(
    r"^g16_(?P<block>blk\d+|smoke\d+)_(?P<arm>d\d+)_boot(?P<boot>\d+)_(?P<job>\d+)$")

T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
          7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 26: 2.056, 27: 2.052}


def arm_sm(arm: str) -> int:
    return int(arm[1:])


# --------------------------------------------------------------------------
# single pass over one telemetry file
# --------------------------------------------------------------------------
def load_snapshots(path):
    """Rows passing the controller_summary filter, in file order.

    The E-CNT filter (`decode_running_batch_size > 0`) is a strict subset of
    this, so one pass serves both estimands.  Fields kept are exactly the ones
    the two canonical tools read.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            if '"runtime_snapshot"' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (e.get("event") != "runtime_snapshot"
                    or e.get("phase") != "benchmark"
                    or "decode_sms" not in e):
                continue
            rows.append((
                float(e["timestamp_monotonic_s"]),
                int(e["decode_sms"]),
                int(e.get("decode_running_batch_size", 0) or 0),
                int(e.get("prefill_active_batch_size", 0) or 0),
            ))
    rows.sort(key=lambda r: r[0])
    return rows


def time_weighted(rows, states_of_interest=None):
    """E-TIM accumulation, byte-for-byte the controller_summary rule:
    residency[state(i)] += t(i+1) - t(i).  Returns (residency_all,
    residency_decode_active) in seconds."""
    all_s = collections.defaultdict(float)
    da_s = collections.defaultdict(float)
    for i in range(len(rows) - 1):
        dt = max(0.0, rows[i + 1][0] - rows[i][0])
        st = rows[i][1]
        all_s[st] += dt
        if rows[i][2] > 0:
            da_s[st] += dt
    return dict(all_s), dict(da_s)


def count_based(rows):
    """E-CNT accumulation.  `decode_sms or 108` is realized_pin_check's own
    mapping (stream group 0 == unpartitioned -> decode unrestricted)."""
    all_c = collections.Counter()
    da_c = collections.Counter()
    da_prefill_inflight = 0
    da_state0 = 0
    for t, sm, dbs, pbs in rows:
        all_c[sm] += 1
        if dbs > 0:
            da_c[sm or 108] += 1
            if sm == 0:
                da_state0 += 1
            if pbs > 0:
                da_prefill_inflight += 1
    return all_c, da_c, da_prefill_inflight, da_state0


# --------------------------------------------------------------------------
def run_pin_tool(path, expect):
    out = subprocess.run([sys.executable, PIN_TOOL, path, str(expect)],
                         capture_output=True, text=True, check=True).stdout
    hist, n, frac, co = {}, None, None, None
    for line in out.splitlines():
        if line.startswith("REALIZED_hist"):
            n = int(re.search(r"decode-active n=(\d+)", line).group(1))
            for k, v in re.findall(r"(\d+)SM:(\d+)\(", line):
                hist[int(k)] = int(v)
        elif line.startswith("CO_RESIDENT_frac"):
            co = float(line.split()[1])
        elif line.startswith("REALIZED_PIN"):
            frac = float(re.search(r"frac=([0-9.]+)", line).group(1))
    return {"n_decode_active": n, "hist": hist, "frac_at_D": frac,
            "co_resident_frac": co, "raw": out}


# --------------------------------------------------------------------------
def phase_windows(sidecar):
    out = {"LO": [], "HI": []}
    for r in sidecar.get("rounds", []):
        out[r["phase"]].append((float(r["t_start"]), float(r["t_end"])))
    return out


def residency_in_windows(rows, windows):
    """Apply the canonical E-TIM rule INSIDE each pre-registered round window
    and sum over rounds (so inter-round idle is not attributed to any state).
    Returns (all_seconds, decode_active_seconds, counts_all, counts_da)."""
    all_s = collections.defaultdict(float)
    da_s = collections.defaultdict(float)
    all_c = collections.Counter()
    da_c = collections.Counter()
    for (t0, t1) in windows:
        sub = [r for r in rows if t0 <= r[0] <= t1]
        for i in range(len(sub) - 1):
            dt = max(0.0, sub[i + 1][0] - sub[i][0])
            all_s[sub[i][1]] += dt
            if sub[i][2] > 0:
                da_s[sub[i][1]] += dt
        for t, sm, dbs, pbs in sub:
            all_c[sm] += 1
            if dbs > 0:
                da_c[sm or 108] += 1
    return dict(all_s), dict(da_s), all_c, da_c


# --------------------------------------------------------------------------
def frac(d, key):
    tot = sum(d.values())
    return (d.get(key, 0.0) / tot) if tot else float("nan")


def msd(xs):
    xs = [x for x in xs if x == x]
    n = len(xs)
    if n == 0:
        return {"n": 0}
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else float("nan")
    t = T_CRIT.get(n - 1)
    half = t * sd / math.sqrt(n) if (n > 1 and t) else float("nan")
    return {"n": n, "mean": m, "sd": sd, "min": min(xs), "max": max(xs),
            "t_ci_lo": m - half, "t_ci_hi": m + half, "t_df": n - 1}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "residency_scope_2026-08-17.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(SLO, "g16_*_telemetry.jsonl")))
    if args.limit:
        paths = paths[:args.limit]

    report = json.load(open(REPORT))
    rep_res = report["campaign"]["LO"]["side_outputs"]["residency_fraction_by_boot"]
    rep_res_hi = report["campaign"]["HI"]["side_outputs"]["residency_fraction_by_boot"]

    boots = []
    controls = {"PC1_pin_tool_match": [], "PC2_ctrlsummary_match": [],
                "PC2b_report_match": [], "NC1_wrong_D_detects": [],
                "NC1b_truncated_detects": []}

    for path in paths:
        run_id = os.path.basename(path)[:-len("_telemetry.jsonl")]
        m = RUN_RE.match(run_id)
        assert m, run_id
        arm, block = m.group("arm"), m.group("block")
        D = arm_sm(arm)
        mode = "smoke" if block.startswith("smoke") else "full"

        rows = load_snapshots(path)
        all_s, da_s = time_weighted(rows)
        all_c, da_c, da_pf, da_s0 = count_based(rows)

        # ---------- PC-1 : canonical E-CNT tool, subprocess ----------
        pin = run_pin_tool(path, D)
        mine_frac_D = da_c.get(D, 0) / sum(da_c.values()) if sum(da_c.values()) else float("nan")
        ok1 = (pin["n_decode_active"] == sum(da_c.values())
               and abs(pin["frac_at_D"] - round(mine_frac_D, 3)) < 5e-4
               and abs(pin["co_resident_frac"] - round(da_pf / max(1, sum(da_c.values())), 3)) < 5e-4
               and {k: v for k, v in pin["hist"].items()} == dict(da_c))
        controls["PC1_pin_tool_match"].append([run_id, bool(ok1)])
        # NC-1 : same comparator against the WRONG arm must NOT match
        if D != 108:
            pin_bad = run_pin_tool(path, D + 10)
            controls["NC1_wrong_D_detects"].append(
                [run_id, bool(abs(pin_bad["frac_at_D"] - round(mine_frac_D, 3)) >= 5e-4)])

        # ---------- PC-2 : canonical E-TIM function ----------
        events = [{"event": "runtime_snapshot", "phase": "benchmark",
                   "decode_sms": r[1], "timestamp_monotonic_s": r[0]} for r in rows]
        cs = controller_summary(events)
        harness_path = os.path.join(SLO, run_id + "_controller_summary.json")
        harness = json.load(open(harness_path)) if os.path.exists(harness_path) else None

        def close(a, b):
            if a is None or b is None:
                return False
            ka = {int(k): v for k, v in a.items()}
            kb = {int(k): v for k, v in b.items()}
            return ka.keys() == kb.keys() and all(abs(ka[k] - kb[k]) < 1e-9 for k in ka)

        # lesson item 21: distinguish "artifact missing" (measurement gap)
        # from "values disagree" (control failure).  Smoke job 884292 lost
        # H3'-b on 3/3 boots -- that is a MISSING artifact, not a mismatch.
        if harness is None:
            pc2 = "artifact_missing"
        elif close(cs["residency_fraction"], harness.get("residency_fraction")):
            pc2 = "match"
        else:
            pc2 = "MISMATCH"
        controls["PC2_ctrlsummary_match"].append([run_id, pc2])
        # the campaign report keys residency by the *_LO.jsonl / *_HI.jsonl
        # artifact name (BootRecord.path), not by the telemetry name.
        rep_here = rep_res.get(run_id + "_LO.jsonl")
        if mode == "full":
            controls["PC2b_report_match"].append(
                [run_id, bool(close(cs["residency_fraction"], rep_here)),
                 bool(close(rep_here, rep_res_hi.get(run_id + "_HI.jsonl")))])
        # NC-1b : truncated events must be DETECTED as different
        cs_trunc = controller_summary(events[:int(0.9 * len(events))])
        controls["NC1b_truncated_detects"].append(
            [run_id, bool(not close(cs["residency_fraction"], cs_trunc["residency_fraction"]))])

        # ---------- phase-restricted (pre-registered round windows) ----------
        sidecar_path = os.path.join(SLO, run_id + "_sidecar.json")
        sc = json.load(open(sidecar_path)) if os.path.exists(sidecar_path) else {}
        win = phase_windows(sc)
        per_phase = {}
        for ph in ("LO", "HI"):
            if not win.get(ph):
                continue
            a_s, d_s, a_c, d_c = residency_in_windows(rows, win[ph])
            per_phase[ph] = {
                "W_tim_all": frac(a_s, D), "W_tim_da": frac(d_s, D),
                "W_cnt_all": frac(a_c, D), "W_cnt_da": frac(d_c, D),
                "sec_at_D": a_s.get(D, 0.0), "sec_total": sum(a_s.values()),
                "sec_decode_active": sum(d_s.values()),
                "n_snapshots": sum(a_c.values()), "n_decode_active": sum(d_c.values()),
                "states_time_s": {str(k): v for k, v in sorted(a_s.items())},
            }

        boots.append({
            "run_id": run_id, "arm": arm, "decode_sm": D, "block": block,
            "mode": mode, "node": sc.get("node"), "gpu_uuid": sc.get("gpu_uuid"),
            "NP": sc.get("NP"), "ROUNDS": sc.get("ROUNDS"),
            "t_boot0": sc.get("t_boot0"),
            # --- the four unified-denominator cells, whole benchmark span ---
            "W_cnt_all": frac(all_c, D),
            "W_cnt_da": mine_frac_D,                       # == E-CNT canonical
            "W_tim_all": frac(all_s, D),                   # == E-TIM canonical
            "W_tim_da": frac(da_s, D),
            "cnt_da_at_108bin": frac(da_c, 108),
            "co_resident_frac": da_pf / max(1, sum(da_c.values())),
            "n_snapshots": len(rows), "n_decode_active": sum(da_c.values()),
            "n_decode_active_state0": da_s0,
            "sec_at_D": all_s.get(D, 0.0), "sec_total": sum(all_s.values()),
            "sec_decode_active": sum(da_s.values()),
            "states_time_s": {str(k): v for k, v in sorted(all_s.items())},
            "states_count": {str(k): v for k, v in sorted(all_c.items())},
            "split_transitions": cs["split_transitions"],
            "per_phase": per_phase,
        })
        print(f"  {run_id:38s} cnt_da={mine_frac_D:.4f} tim_all={frac(all_s, D):.4f} "
              f"tim_da={frac(da_s, D):.4f} n_da={sum(da_c.values())}", flush=True)

    out = {"boots": boots, "controls": controls}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote", args.out)

    # ---------------- control verdict ----------------
    def allok(rows_, idx=1):
        return all(r[idx] for r in rows_) and len(rows_) > 0
    print("\n=== CONTROLS ===")
    print("PC-1 realized_pin_check subprocess == my recompute :",
          allok(controls["PC1_pin_tool_match"]), len(controls["PC1_pin_tool_match"]))
    pc2 = collections.Counter(r[1] for r in controls["PC2_ctrlsummary_match"])
    print("PC-2 controller_summary(mine) vs harness json      :",
          dict(pc2), "-> pass" if pc2["MISMATCH"] == 0 else "-> FAIL")
    print("PC-2b == campaign report by_boot (LO==HI view)     :",
          allok(controls["PC2b_report_match"]), allok(controls["PC2b_report_match"], 2),
          len(controls["PC2b_report_match"]))
    print("NC-1 wrong-D comparison DETECTED as mismatch       :",
          allok(controls["NC1_wrong_D_detects"]), len(controls["NC1_wrong_D_detects"]))
    print("NC-1b truncated-events DETECTED as mismatch        :",
          allok(controls["NC1b_truncated_detects"]), len(controls["NC1b_truncated_detects"]))


if __name__ == "__main__":
    main()
