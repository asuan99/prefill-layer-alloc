#!/usr/bin/env python3
"""Smoke check #9 regression test (2026-08-16, post-smoke-884292).

Smoke job 884292 printed

    G16_SMOKE_CHECK 9_H2_COUNT_AND_RESIDENCY=PASS
      runtime_snapshot_benchmark_count=21511 residency_fraction={}

and `G16_SMOKE_OVERALL=PASS`, while the harness had in fact produced NO
`*_controller_summary.json` at all on 3/3 arms (the H3'-b heredoc died on an
unbound `PDMUX_EVAL_DIR` before python started).  The old checker swallowed
that in `except Exception: print("{}")` and hardcoded `=PASS` -- a gate
labelling its own measurement failure as a success (methodology gate #9;
lessons item 21, mirror image).

This test drives the REAL bash text of the rewritten item-9 block, extracted
from `g16_grid.sbatch` between the `G16_SMOKE9_BLOCK_START` /
`G16_SMOKE9_BLOCK_END` sentinels (so it cannot drift from the harness), over:

  A. valid summary (non-empty residency_fraction summing to 1) -> 9=PASS
  B. `controller_summary.json` MISSING                          -> 9b=FAIL
  C. file present but unparseable JSON                          -> 9b=FAIL
  D. `residency_fraction` present but `{}`                       -> 9b=FAIL
  E. `residency_fraction` KEY ABSENT (controller_summary()'s     -> 9b=FAIL
     no-snapshot early return omits the key entirely, so
     `.get(k, {})` cannot tell this from case D)
  F. degenerate all-zero fractions (total == 0)                  -> 9b=FAIL
  G. telemetry snapshot count BELOW MIN_SNAPSHOTS               -> 9a=FAIL
  H. REAL smoke-884292 artifacts (telemetry + the absent          -> 9=FAIL
     controller_summary.json) -- the run that wrongly PASSED

Case H is the load-bearing one: it re-runs the new gate against the actual
files 884292 left on disk and requires the verdict to flip PASS -> FAIL.
A gate that cannot be shown to fail on the defect it was written for is not
evidence.

GPU 0. stdlib python + bash only; no torch, no sglang, no server.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SBATCH = os.path.join(HERE, "g16_grid.sbatch")
MIN_SNAPSHOTS_SMOKE = 5


def _extract_smoke9_block(text: str) -> str:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if "G16_SMOKE9_BLOCK_START" in l)
    end = next(i for i in range(start + 1, len(lines)) if "G16_SMOKE9_BLOCK_END" in lines[i])
    return "\n".join(lines[start : end + 1])


def _write_telemetry(path: str, n_snapshots: int) -> None:
    with open(path, "w") as fh:
        for i in range(n_snapshots):
            fh.write(json.dumps({
                "event": "runtime_snapshot",
                "phase": "benchmark",
                "decode_sms": 74,
                "timestamp_monotonic_s": 1000.0 + i,
            }) + "\n")
        # noise the counter must ignore
        fh.write(json.dumps({"event": "request_end", "phase": "benchmark"}) + "\n")


def _run(block: str, *, telemetry: str, summary: str, min_snapshots: int = MIN_SNAPSHOTS_SMOKE,
         telem_rc: int = 0, ctrlsummary_rc: int = 0) -> dict:
    # Every name the extracted block reads must be bound here, matching what
    # the real arm loop has in scope at that point (`set -u`): the telemetry
    # path + summary path + MIN_SNAPSHOTS + the two upstream return codes it
    # echoes for cross-reference. CTRLSUMMARY_RC is initialised at the top of
    # the arm loop and assigned right after the H3'-b heredoc, and every
    # earlier failure path `continue`s before reaching the smoke block, so it
    # is always bound there in the real harness.
    preamble = "\n".join([
        "set -uo pipefail",
        "PDMUX_TELEMETRY_PATH=" + shlex.quote(telemetry),
        "CTRLSUMMARY_JSON=" + shlex.quote(summary),
        f"MIN_SNAPSHOTS={min_snapshots}",
        f"TELEM_RC={telem_rc}",
        f"CTRLSUMMARY_RC={ctrlsummary_rc}",
        'SMOKE9=""; SMOKE9A=""; SMOKE9B=""',
    ])
    r = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", preamble + "\n\n" + block + "\n"],
        capture_output=True, text=True,
    )
    out = {"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr,
           "9a": None, "9b": None, "9": None, "reason": None, "count": None}
    for line in r.stdout.splitlines():
        if "9a_H2_SNAPSHOT_COUNT=" in line:
            out["9a"] = line.split("9a_H2_SNAPSHOT_COUNT=", 1)[1].split()[0]
            for tok in line.split():
                if tok.startswith("runtime_snapshot_benchmark_count="):
                    out["count"] = tok.split("=", 1)[1]
        elif "9b_H3B_RESIDENCY=" in line:
            out["9b"] = line.split("9b_H3B_RESIDENCY=", 1)[1].split()[0]
            for tok in line.split():
                if tok.startswith("reason="):
                    out["reason"] = tok.split("=", 1)[1]
        elif "9_H2_COUNT_AND_RESIDENCY=" in line:
            out["9"] = line.split("9_H2_COUNT_AND_RESIDENCY=", 1)[1].split()[0]
    return out


def _case(label: str, got: dict, *, want9: str, want9a: str, want9b: str) -> bool:
    ok = got["9"] == want9 and got["9a"] == want9a and got["9b"] == want9b
    print(f"[smoke9] {label}: 9={got['9']} 9a={got['9a']} 9b={got['9b']} "
          f"reason={got['reason']} count={got['count']} -> {'PASS' if ok else 'FAIL'} "
          f"(want 9={want9} 9a={want9a} 9b={want9b})")
    if not ok and got["rc"] != 0:
        print(f"    bash rc={got['rc']} stderr={got['stderr'].strip()[:400]!r}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke-job", default="884292",
                    help="job id whose REAL artifacts case H replays (default: the run that "
                         "wrongly reported PASS)")
    args = ap.parse_args()

    block = _extract_smoke9_block(open(SBATCH).read())
    ok = True

    with tempfile.TemporaryDirectory() as td:
        telem = os.path.join(td, "telemetry.jsonl")
        _write_telemetry(telem, 40)

        def summary_file(name: str, payload) -> str:
            p = os.path.join(td, name)
            if isinstance(payload, str):
                with open(p, "w") as fh:
                    fh.write(payload)
            else:
                with open(p, "w") as fh:
                    json.dump(payload, fh)
            return p

        # A: valid
        good = summary_file("good.json", {
            "split_transitions": 3,
            "residency_s": {"34": 10.0, "74": 30.0},
            "residency_fraction": {"34": 0.25, "74": 0.75},
            "dwell_s": [10.0, 30.0],
        })
        ok = _case("A valid summary", _run(block, telemetry=telem, summary=good),
                   want9="PASS", want9a="PASS", want9b="PASS") and ok

        # B: missing file (THE 884292 shape)
        ok = _case("B summary file missing",
                   _run(block, telemetry=telem, summary=os.path.join(td, "nope.json")),
                   want9="FAIL", want9a="PASS", want9b="FAIL") and ok

        # C: unparseable
        bad = summary_file("bad.json", "{not json,,,")
        ok = _case("C summary unparseable", _run(block, telemetry=telem, summary=bad),
                   want9="FAIL", want9a="PASS", want9b="FAIL") and ok

        # D: residency_fraction present but empty
        empty = summary_file("empty.json", {"split_transitions": 0, "residency_fraction": {}})
        ok = _case("D residency_fraction={}", _run(block, telemetry=telem, summary=empty),
                   want9="FAIL", want9a="PASS", want9b="FAIL") and ok

        # E: key absent -- controller_summary()'s no-snapshot early return
        absent = summary_file("absent.json",
                              {"split_transitions": 0, "residency_s": {}, "dwell_s": []})
        ok = _case("E residency_fraction key absent",
                   _run(block, telemetry=telem, summary=absent),
                   want9="FAIL", want9a="PASS", want9b="FAIL") and ok

        # F: degenerate all-zero fractions
        zero = summary_file("zero.json", {"residency_fraction": {"74": 0.0}})
        ok = _case("F residency_fraction all-zero", _run(block, telemetry=telem, summary=zero),
                   want9="FAIL", want9a="PASS", want9b="FAIL") and ok

        # G: snapshot count below MIN_SNAPSHOTS
        thin = os.path.join(td, "thin.jsonl")
        _write_telemetry(thin, 2)
        ok = _case("G snapshot count 2 < min 5", _run(block, telemetry=thin, summary=good),
                   want9="FAIL", want9a="FAIL", want9b="PASS") and ok

    # H: the REAL 884292 artifacts.
    job = args.smoke_job
    real_telem = glob.glob(os.path.join(HERE, f"g16_smoke1_d74_boot1_{job}_telemetry.jsonl"))
    real_summary = os.path.join(HERE, f"g16_smoke1_d74_boot1_{job}_controller_summary.json")
    if real_telem:
        got = _run(block, telemetry=real_telem[0], summary=real_summary)
        ok = _case(f"H REAL smoke job {job} artifacts (originally reported PASS)", got,
                   want9="FAIL", want9a="PASS", want9b="FAIL") and ok
        print(f"    [context] summary_exists={os.path.exists(real_summary)} "
              f"telemetry={os.path.basename(real_telem[0])}")
    else:
        print(f"[smoke9] H SKIPPED -- no telemetry artifact for job {job} in {HERE} "
              f"(this case is evidence, not a gate; rerun with --smoke-job <id>)")

    print(f"[smoke9] OVERALL={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
