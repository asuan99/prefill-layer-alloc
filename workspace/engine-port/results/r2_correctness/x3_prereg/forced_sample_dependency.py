#!/usr/bin/env python3
"""Does job 907100's O-tier overlap evidence come from FORCED telemetry samples?

X3 runs the r2_eval runner's configuration, in which `PDMUX_TRACE_FORCE_PREFILL`
is unset.  The pre-registration predicts that the O-tier confirmation (O2: a
snapshot with prefill and decode both active inside each probe window, on the D
division) then fails -- because the forced samples are the instrument that makes
prefill-in-flight states visible at all.  The engine's own comment gives the
mechanism (prefill is ~4.9% of wall time but 0.07% of SCHEDULED samples,
multiplexing_mixin.py:545-551); this script checks it on the actual run that the
gate was scored on, so the prediction rests on data and not only on a comment.

usage: forced_sample_dependency.py <job_dir> [labels...]
"""
import json
import pathlib
import sys


def main():
    job = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    labels = sys.argv[2:] or [lb for lb in (job / "boots.txt").read_text().split()]
    print(f"=== {job.name}: overlap snapshots by sample origin ===")
    print(f"{'boot':5} {'snapshots':>9} {'forced':>7} {'overlap':>8} {'forced':>7} "
          f"{'scheduled':>10}  (overlap = prefill>=1 AND decode>=1)")
    for lb in labels:
        f = job / f"tel_{lb}.jsonl"
        if not f.exists():
            print(f"{lb:5} MISSING {f}")
            continue
        snaps = forced = ov = ov_forced = 0
        for line in f.read_text(errors="replace").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") != "runtime_snapshot":
                continue
            snaps += 1
            fc = bool(r.get("trace_forced"))
            forced += fc
            if (r.get("prefill_active_batch_size") or 0) >= 1 and \
               (r.get("decode_running_batch_size") or 0) >= 1:
                ov += 1
                ov_forced += fc
        print(f"{lb:5} {snaps:>9} {forced:>7} {ov:>8} {ov_forced:>7} {ov - ov_forced:>10}")
    print("\nReading: the `scheduled` column is what a run WITHOUT forced samples "
          "would have had to work with.  O2 needs at least one such snapshot "
          "INSIDE each of the 8 probe windows, on the D division.")


if __name__ == "__main__":
    main()
