#!/usr/bin/env python3
"""E1 addendum: Gate 2-S's OWN per-cell A4 max(decode_running_batch_size).

Pre-registration: PREREG_G2S_E1_ADDENDUM_2026-08-11.md  (READ SECTION 0 FIRST --
this registration is POST-HOC, NOT BLIND, for the two Granite cells).

WHAT THIS IS
  PREREG_GATE2S sec8.9's premise ("the A4 selector never reaches idx2=(54,54) in
  these cells") is currently supported by observations TRANSFERRED from
  replication grids: job 875293 (G1-b, Zamba2, n=1) and job 877974 (G1-c,
  Granite, n=1).  This script computes the decision quantity from THIS
  campaign's own cells (jobs 877756/877757), n=10 reps, arm A4, so the transfer
  argument is no longer needed.

WHAT THIS IS NOT
  * It does NOT modify, import, or re-run g2s_analyze.py.  That scorer is the
    audited producer of the rev20 result and is untouched.
  * It does NOT recompute frac(idx2).  sec8.9.1 already did, and by the code
    identity below the two numbers carry ONE empirical fact, not two.
  * It does NOT emit or read any TTFT/TPOT/ITL/goodput number.
  * It does NOT create a new REFUTED path.  Refutation stays sec8.9.1's monopoly.
  * It says NOTHING about magnitude-citation eligibility (F-series gate).

ZERO NEW FREE PARAMETERS -- every rule is imported from the PRODUCER:
    gate1/gate1b_analyze.py : max_decode_bs()  (module docstring lines 8-9: over
                              ALL benchmark-phase rows in the window, NOT just
                              population A), load_rows() (event/phase filter),
                              classify() (pop A/B/C), DECODE_BS_DIVISOR = 36
    pdmux_a100_smoke.yml    : sm_group_num 4, decode_bs_divisor 36, no manual_divisions
    src/multiplex/multiplexing_mixin.py:893-909 : stream_idx == 2 <=> decode_bs >= 36
  (methodology gate #14: a gate that copies the code it means to check is close
   to an identity -- so the check is anchored on the PRODUCER, not on the
   consumer being checked.)

Exit status is always 0.  This is a diagnostic, not a gate on the campaign.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_GATE1 = os.path.abspath(os.path.join(_HERE, os.pardir, "gate1"))
sys.path.insert(0, _GATE1)

import gate1b_analyze as g1b          # noqa: E402  PRODUCER -- rules imported, not copied

# --- the two imported constants, restated only so the report can print them ---
THRESHOLD = g1b.DECODE_BS_DIVISOR      # 36  (gate1b_analyze.py:53)
TRACE_EVERY = g1b.TRACE_EVERY          # 32  (gate1b_analyze.py:52)

# sec3 of PREREG_GATE2S: the four scored cells, and the arm the premise is about.
CELLS = {
    "zamba2-27b": ("2", "3"),
    "granite-40-h-micro-base": ("3", "4"),
}
ARM = "agnostic"                        # = A4
N_REPS_EXPECTED = 10                    # PREREG_GATE2S sec3


# ==========================================================================
# rep boundary reconstruction -- prereg sec3.  NOT a gap heuristic: the offsets
# were recorded BY THE HARNESS AT MEASUREMENT TIME (g2s_run.sbatch:254-268,
# PREREG_GATE2S sec9.1) and the concatenation recipe is the harness's own
# (g2s_run.sbatch:274-289: sorted(glob(...)), lines[s0:s1]).
# ==========================================================================
def rep_slices(outdir, tag, job, rate):
    out = []
    pattern = os.path.join(outdir, f"g2s_{tag}_telemetry_{ARM}_rep*_{job}.jsonl")
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
    return out


def raw_scan(lines):
    """Integrity scan on RAW lines (load_rows silently drops unparseable rows,
    so parse failures must be counted separately -- prereg I1)."""
    n_bad = 0
    n_rt = 0
    n_bench = 0
    dropped_max = 0
    dropped_rows = 0
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
    inversions = sum(1 for a, b in zip(sis, sis[1:]) if b < a)
    dups = len(sis) - len(set(sis))
    gaps = [b - a for a, b in zip(sis, sis[1:]) if b > a]
    return dict(n_lines=len([l for l in lines if l.strip()]),
                n_parse_fail=n_bad, n_runtime_snapshot=n_rt,
                n_benchmark_phase=n_bench,
                dropped_events_max=dropped_max, n_rows_with_dropped=dropped_rows,
                sample_index_inversions=inversions, sample_index_duplicates=dups,
                sample_index_gap_median=(statistics.median(gaps) if gaps else None),
                sample_index_gap_max=(max(gaps) if gaps else None))


def max_via_producer(lines):
    """Write the slice to a temp file and run the PRODUCER's own load_rows() +
    max_decode_bs().  Nothing is reimplemented here."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.writelines(lines)
        rows = g1b.load_rows(path)                 # event+phase filter, ts sort
        return g1b.max_decode_bs(rows), rows
    finally:
        os.unlink(path)


def analyze_cell(outdir, tag, job, rate):
    res = dict(tag=tag, job=job, rate=rate, arm=ARM, threshold=THRESHOLD,
               trace_every=TRACE_EVERY)
    cell_path = os.path.join(outdir, f"g2s_{tag}_telemetry_{ARM}_r{rate}_{job}.jsonl")
    res["cell_file"] = os.path.basename(cell_path)
    if not os.path.exists(cell_path):
        res["state"] = "UNDETERMINED"
        res["reason"] = "cell telemetry file missing"
        return res

    # ---- path 1 (PRIMARY, prereg sec3): per-boot slices, rep by rep ----------
    slices = rep_slices(outdir, tag, job, rate)
    per_rep = []
    for s in slices:
        m, rows = max_via_producer(s["lines"])
        scan = raw_scan(s["lines"])
        ts = [r["timestamp_monotonic_s"] for r in rows]
        per_rep.append(dict(rep=s["rep"], boot_file=s["boot_file"],
                            offset=s["offset"], n_lines=scan["n_lines"],
                            max_decode_bs=m, headroom=(None if m is None else THRESHOLD - m),
                            below_threshold=(m is not None and m < THRESHOLD),
                            t_first=(ts[0] if ts else None), t_last=(ts[-1] if ts else None),
                            integrity=scan))
    per_rep.sort(key=lambda d: int(d["rep"]))
    res["per_rep"] = per_rep

    rep_maxes = [d["max_decode_bs"] for d in per_rep if d["max_decode_bs"] is not None]
    pooled_from_reps = max(rep_maxes) if rep_maxes else None

    # ---- path 2 (CROSS-CHECK, prereg I5): the concatenated cell file ---------
    with open(cell_path) as fh:
        cell_lines = fh.readlines()
    cell_scan = raw_scan(cell_lines)
    pooled_from_cell, _cell_rows = max_via_producer(cell_lines)

    res["pooled_max_decode_bs"] = pooled_from_cell
    res["pooled_max_from_reps"] = pooled_from_reps
    res["headroom"] = (None if pooled_from_cell is None else THRESHOLD - pooled_from_cell)
    res["cell_integrity"] = cell_scan

    # ---- descriptive: chronological rep order + inter-boot gaps -------------
    chron = sorted([d for d in per_rep if d["t_first"] is not None],
                   key=lambda d: d["t_first"])
    boundary = []
    for a, b in zip(chron, chron[1:]):
        boundary.append(dict(after_rep=a["rep"], before_rep=b["rep"],
                             gap_s=round(b["t_first"] - a["t_last"], 3)))
    res["chronological_rep_order"] = [d["rep"] for d in chron]
    res["inter_boot_gaps_s_DESCRIPTIVE"] = boundary
    res["n_boundaries"] = len(boundary)

    # ---- integrity gates I1..I6 (prereg sec4) -------------------------------
    fails = []
    if cell_scan["n_parse_fail"] or any(d["integrity"]["n_parse_fail"] for d in per_rep):
        fails.append("I1 parse failures > 0")
    if cell_scan["dropped_events_max"] or any(d["integrity"]["dropped_events_max"] for d in per_rep):
        fails.append(f"I2 dropped_events > 0 (max={cell_scan['dropped_events_max']})")
    n_slice_lines = sum(d["n_lines"] for d in per_rep)
    if n_slice_lines != cell_scan["n_lines"]:
        fails.append(f"I3 slice sum {n_slice_lines} != cell lines {cell_scan['n_lines']}")
    if len(per_rep) != N_REPS_EXPECTED:
        fails.append(f"I4 rep count {len(per_rep)} != {N_REPS_EXPECTED}")
    if pooled_from_cell != pooled_from_reps:
        fails.append(f"I5 pooled mismatch {pooled_from_cell} != {pooled_from_reps}")
    inv = [d["rep"] for d in per_rep if d["integrity"]["sample_index_inversions"]]
    dup = [d["rep"] for d in per_rep if d["integrity"]["sample_index_duplicates"]]
    if inv or dup:
        fails.append(f"I6 within-rep sample_index inversions={inv} duplicates={dup}")
    res["integrity_failures"] = fails
    res["I3_slice_sum_lines"] = n_slice_lines
    # I7 descriptive only
    res["I7_all_rows_runtime_snapshot_benchmark"] = (
        cell_scan["n_benchmark_phase"] == cell_scan["n_lines"])
    # concatenation artifact: glob order gives exactly (n_reps - 1) inversions
    res["cell_file_sample_index_inversions_EXPECTED_n_reps_minus_1"] = \
        cell_scan["sample_index_inversions"]

    # ---- verdict (prereg sec5) ---------------------------------------------
    if fails:
        res["state"] = "UNDETERMINED"
        res["reason"] = "; ".join(fails)
    elif pooled_from_cell is None:
        res["state"] = "UNDETERMINED"
        res["reason"] = "no rows"
    elif pooled_from_cell < THRESHOLD:
        res["state"] = "VERIFIED_AT_SAMPLED_INSTANTS"
        res["reason"] = (
            f"pooled max(decode_bs)={pooled_from_cell} < {THRESHOLD} "
            f"(headroom {THRESHOLD - pooled_from_cell}); n={len(per_rep)} reps, "
            f"arm A4, THIS campaign's own cell.  Sampled at TRACE_EVERY={TRACE_EVERY} "
            f"-- observed max is a LOWER BOUND on the true max (prereg sec2.5), so this "
            f"is 'never at any sampled instant', NOT 'never'.")
    else:
        res["state"] = "UNDETERMINED"
        res["reason"] = (
            f"pooled max(decode_bs)={pooled_from_cell} >= {THRESHOLD}: threshold was "
            f"reached at >=1 sampled instant.  NOT verified.  Refutation is sec8.9.1's "
            f"monopoly (frac(idx2) >= 0.01) and this addendum does not create one.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=_HERE)
    ap.add_argument("--pair", action="append", required=True,
                    help="tag:job  e.g. zamba2-27b:877756")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    print("=" * 78)
    print("E1 ADDENDUM -- Gate 2-S per-cell A4 max(decode_running_batch_size)")
    print("PREREG: PREREG_G2S_E1_ADDENDUM_2026-08-11.md")
    print("*** POST-HOC, NOT BLIND for the Granite cells (prereg sec0) ***")
    print("*** NEW statistic: sec8.9.1's one-way safety does NOT transfer -> RE-AUDIT ***")
    print(f"threshold = decode_bs_divisor = {THRESHOLD}  (imported, gate1b_analyze.py:53)")
    print(f"identity  : stream_idx == 2 <=> decode_bs >= {THRESHOLD} "
          f"(multiplexing_mixin.py:900-909) -> max<{THRESHOLD} IMPLIES frac(idx2)=0;")
    print("            the two numbers are ONE empirical fact, not two (prereg sec2.3)")
    print("=" * 78)

    report = dict(prereg="PREREG_G2S_E1_ADDENDUM_2026-08-11.md",
                  blind=False,
                  post_hoc_known_values={"granite-40-h-micro-base r3": 10,
                                         "granite-40-h-micro-base r4": 13,
                                         "granite rep-wise range": [9, 13]},
                  threshold=THRESHOLD, trace_every=TRACE_EVERY,
                  producer_rules="gate1/gate1b_analyze.py (imported, not copied)",
                  cells={})

    for pair in a.pair:
        tag, job = pair.split(":")
        for rate in CELLS[tag]:
            r = analyze_cell(a.outdir, tag, job, rate)
            report["cells"][f"{tag}|r{rate}"] = r
            print(f"\n--- {tag}  r{rate}  (job {job}, arm A4) ---")
            print(f"  pooled max(decode_bs) = {r.get('pooled_max_decode_bs')}   "
                  f"threshold = {THRESHOLD}   headroom = {r.get('headroom')}")
            pr = r.get("per_rep", [])
            print("  per-rep max (DESCRIPTIVE -- max is associative, so this is NOT a "
                  "second test; prereg sec2.4):")
            print("    " + "  ".join(f"rep{d['rep']}={d['max_decode_bs']}" for d in pr))
            if pr:
                vals = [d["max_decode_bs"] for d in pr if d["max_decode_bs"] is not None]
                print(f"    rep-max range = [{min(vals)}, {max(vals)}]  "
                      f"median = {statistics.median(vals)}")
            ci = r.get("cell_integrity", {})
            print(f"  integrity: lines={ci.get('n_lines')} parse_fail={ci.get('n_parse_fail')} "
                  f"dropped_events_max={ci.get('dropped_events_max')} "
                  f"runtime_snapshot/benchmark={ci.get('n_benchmark_phase')} "
                  f"slice_sum={r.get('I3_slice_sum_lines')} reps={len(pr)}")
            print(f"             within-rep si inversions/dups: "
                  f"{[d['integrity']['sample_index_inversions'] for d in pr]} / "
                  f"{[d['integrity']['sample_index_duplicates'] for d in pr]}")
            print(f"             cell-file si inversions = "
                  f"{r.get('cell_file_sample_index_inversions_EXPECTED_n_reps_minus_1')} "
                  f"(expected {len(pr)-1 if pr else 0} = glob-order concatenation artifact)")
            gaps = [g["gap_s"] for g in r.get("inter_boot_gaps_s_DESCRIPTIVE", [])]
            if gaps:
                print(f"             inter-boot gaps (chronological, DESCRIPTIVE): "
                      f"min={min(gaps)} max={max(gaps)}")
            print(f"  STATE = {r['state']}")
            print(f"  reason: {r['reason']}")

    print("\n" + "=" * 78)
    print("SUMMARY (cell-wise only -- no pooled-across-cells state, prereg sec5)")
    for k, v in report["cells"].items():
        print(f"  {k:34s} max={str(v.get('pooled_max_decode_bs')):>4s}  "
              f"{v['state']}")
    print("\nUNCHANGED BY THIS ADDENDUM (prereg sec6.2): all Gate 2-S performance "
          "verdicts (Delta, Holm, 9-cell, F-series), sec8.9.1 itself, sec5.6.1 naming "
          "rules, rev22's 6 release conditions and 8 prohibitions, and -- explicitly -- "
          "ANYTHING about magnitude-citation eligibility (F-series gate's business, "
          "methodology gate #28).")
    print("=" * 78)

    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
