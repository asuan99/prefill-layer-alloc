#!/usr/bin/env python3
"""Re-READ an archived P1 job with the repaired reader and repaired windows.

★WHAT THIS IS NOT.  It is not a measurement, and it is not a P1-e result.  No
server is booted, no engine is exercised, and no arm is scored (PREREG_CP0_
2026-08-28.md section 9).  It re-counts bytes that already exist on disk.  The
strongest statement it supports is:

    "the repaired analyzer, re-reading the same raw data, reproduces the
     registered expectations"

and NOT "P1-e passed".  P1-e on a new engine run is still unexecuted: this
script cannot show that the repaired PROBE emits the right lines on a GPU, only
that the repaired READER reads these ones correctly.  Two of the seven P1 tests
in the archived job (P1-b, P1-d) had empty probe files and are untouched here.

WHY A RE-READ WAS NEEDED (the two defects, both in the instrument)

  1. channel 2 counted the wrong population.  ``ScheduleBatch`` does not clear
     ``extend_num_tokens`` when the batch object is reused for a decode step,
     so the probe's ``ext > 0`` test admitted decode forwards carrying a stale
     value: 101 of the archived boot's 116 forward lines have
     ``forward_mode == "DECODE"``.  The reader now classifies by
     ``forward_mode``; the excluded lines stay in the raw file and in the
     record.

  2. the phase windows overlapped.  ``p1_accept_workload.py`` registered
     ``t_end + 1.0`` per phase with no clamp, and phase B starts ~1.5 ms after
     phase S ends, so a second of phase B was counted inside phase S as well.
     The windows are rebuilt here from the SAME recorded phase timestamps
     through the repaired ``close_segments``.

``p1e_expectations.json`` is NOT touched by either repair.  The expectations
are hand-derived per phase for non-overlapping windows; what was wrong is the
instrument that was compared against them.

USAGE
    python reanalyze_p1e.py --job p1_accept_899768 \
        --out p1_accept_899768/REANALYSIS_2026-09-01.json
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "cp0.p1e-reanalysis/v1"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _git_head():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(HERE), check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True, help="directory of an archived P1 job")
    ap.add_argument("--boot", default="cps512_on",
                    help="which boot's probe file to re-read")
    ap.add_argument("--workload", default="workload_cps512_on.json")
    ap.add_argument("--expectations", default=str(HERE / "p1e_expectations.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    AN = _load("cp0_analyze_chunk_probe", HERE / "analyze_chunk_probe.py")
    WL = _load("cp0_p1_accept_workload", HERE / "p1_accept_workload.py")

    job = Path(args.job)
    jsonl = job / f"chunk_{args.boot}.jsonl"
    wl_path = job / args.workload
    exp = json.loads(Path(args.expectations).read_text())
    workload = json.loads(wl_path.read_text())

    # the windows, rebuilt from the recorded phase timestamps
    phases = workload["phases"]
    raw_segments = [
        {"segment_id": f"p1e_{pid}", "phase": pid,
         "t_start": phases[pid]["t_start"], "t_end": phases[pid]["t_end"]}
        for pid in phases
    ]
    segments = WL.close_segments(raw_segments)

    rows, bad = AN.read_jsonl(jsonl)
    rec = AN.build_record(rows, {"segments": segments,
                                 "segment_grace_s": WL.SEGMENT_GRACE_S,
                                 "windows": "rebuilt by reanalyze_p1e.py"},
                          bad_lines=bad)

    archived_segments = None
    arch_path = job / f"segments_{args.boot}.json"
    if arch_path.is_file():
        archived_segments = json.loads(arch_path.read_text())["segments"]

    # comparison against the registered expectations, reported field by field
    comparison = {}
    for phase in exp["phases"]:
        pid = phase["id"]
        seg = next((s for s in rec["segments"]
                    if s.get("segment_id") == f"p1e_{pid}"), None)
        if seg is None:
            comparison[pid] = {"status": "NO_WINDOW"}
            continue
        e = seg["extend_tokens_per_forward"]
        got = {
            "n_requests_chunked": seg["n_requests_chunked"],
            "n_chunk_events": seg["n_chunk_events"],
            "n_chunk_events_admission": seg["n_chunk_events_admission"],
            "n_chunk_events_continuation": seg["n_chunk_events_continuation"],
            "n_prefill_forwards": e["n_forwards"],
            "sum_extend_tokens": e["sum"],
            "max_extend_tokens": e["max"],
            "mean_extend_tokens": e["mean"],
            "extend_tokens_per_forward_multiset": list(e["observed_multiset"]),
            "n_prompts_longer_than_cps":
                (phases.get(pid) or {}).get("n_prompts_longer_than_cps"),
        }
        want = phase["expected"]
        mism = []
        for k, v in want.items():
            if k not in got:
                continue
            a, b = got[k], v
            if k.endswith("multiset"):
                a, b = sorted(a), sorted(b)
            if a != b:
                mism.append({"field": k, "expected": v, "observed": got[k]})
        comparison[pid] = {
            "expected": want,
            "observed": got,
            "mismatches": mism,
            "exact_on": phase.get("exact_on"),
            "agrees_with_registered_expectations": not mism,
            # what was counted and deliberately not counted in this window
            "excluded_from_channel_2": {
                "n_nonprefill_forwards_with_extend_tokens":
                    seg["n_nonprefill_forwards_with_extend_tokens"],
                "sum_extend_tokens_not_counted":
                    seg["sum_extend_tokens_not_counted"],
                "n_unknown_mode_forwards": seg["n_unknown_mode_forwards"],
                "forward_mode_counts": seg["forward_mode_counts"],
            },
        }

    out = {
        "schema": SCHEMA,
        "kind": "RE-READ OF ARCHIVED RAW DATA -- NOT A MEASUREMENT",
        "statement_this_supports": (
            "the repaired analyzer, re-reading the same raw data, reproduces "
            "the registered expectations"
        ),
        "statement_this_does_NOT_support": (
            "P1-e passed.  No engine ran here; the repaired probe has not been "
            "exercised on a GPU, and P1-e on a new run is unexecuted."
        ),
        "job": str(job),
        "boot": args.boot,
        "source_files": {
            "probe_jsonl": str(jsonl),
            "workload": str(wl_path),
            "expectations": str(args.expectations),
        },
        "git_head": _git_head(),
        "repairs_applied": [
            "channel 2 counts prefill forward modes only (chunk_probe.py + "
            "analyze_chunk_probe.py); the archived file is pre-repair schema, "
            "so the reclassification happens in the reader",
            "phase windows clamped to the next window's start "
            "(p1_accept_workload.close_segments)",
        ],
        "windows_rebuilt": segments,
        "windows_as_archived": archived_segments,
        "comparison_with_registered_expectations": comparison,
        "boot_record": rec,
    }
    text = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
        sys.stderr.write(f"wrote {args.out}\n")
    else:
        sys.stdout.write(text + "\n")
    ok = all(c.get("agrees_with_registered_expectations") for c in comparison.values())
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
