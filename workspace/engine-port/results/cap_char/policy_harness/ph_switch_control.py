#!/usr/bin/env python3
"""Positive-control smoke for the dynamic arm's switch observability (PREREG §17; NON-MEASUREMENT; no engine change).

Question (decision 52/53): when the bind+GATE controller changes its target D, does the server-log count the harness uses for the
gate-5 `switch_count` (SLO-BIND / SLO-SCHED lines) actually see it?  The first smoke never switched, so 0 proved nothing.
Design: ONE dyn boot (bind+GATE, TUNE-grid member anchor idx 3 / dwell 3, the policy D64 config, non-sticky machinery as in
the measured dyn arm), PHASE_EVENTS + telemetry ON in THIS boot only (it is not a measurement boot), driven by a load that pushes
the controller prefill-ward and back: HI = long prompts (8192 in / 32 out) at 1.5 req/s -> prefill backlog, LO = short requests at 0.5 req/s ->
slack -> drift to the anchor; two HI/LO cycles.
Observables: log_count = number of SLO-BIND|SLO-SCHED lines in the boot's server.log; tel_changes = number of changes of
decode_sms between consecutive CO-RESIDENT decode_iteration events (prefill_in_flight and not prefill_kernel_done_wait) -- a LOWER BOUND of
the controller's decision changes (a decision between two spans with no co-resident iteration, or a down-up pair, is invisible).
Verdict (registered before any data): TOL = max(2, ceil(0.25 * log_count)).
  SWITCH_CONTROL_PASS        tel >= 1 and log >= 1 and |log - tel| <= TOL         rc 0
  SWITCH_CONTROL_NO_SWITCH   tel == 0 and log == 0 (the load did not force a change: no positive control obtained)    rc 4
  SWITCH_LOG_UNOBSERVABLE    tel >= 1 and (log == 0 or |log - tel| > TOL)          rc 1  -> engine event addition is a USER decision
  SWITCH_CONTROL_UNRESOLVED  anything else (log >= 1 with tel == 0, boot failure)  rc 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))
import ph_spec as P  # noqa: E402

CONTROL_ARM = (3, 3)                      # (anchor_idx, dwell): a member of the TUNE grid {2,3,4} x {3,6}
HI = {"n": 40, "rate": 1.5, "in": 8192, "out": 32}
LO = {"n": 15, "rate": 0.5, "in": 128, "out": 64}
CYCLES = 2
LOG_RE = re.compile(r"SLO-(?:BIND|SCHED) (\d+)->(\d+)")


def tolerance(log_count):
    return max(2, math.ceil(0.25 * log_count))


def log_switch_count(text):
    return len(LOG_RE.findall(text))


def telemetry_target_changes(tel_path):
    """Changes of decode_sms between consecutive co-resident decode_iteration events, in iter_seq order (lower bound)."""
    rows = []
    for ln in open(tel_path):
        if '"decode_iteration"' not in ln:
            continue
        d = json.loads(ln)
        if d.get("event") == "decode_iteration" and d.get("prefill_in_flight") and not d.get("prefill_kernel_done_wait"):
            rows.append((int(d.get("iter_seq", 0)), int(d["decode_sms"])))
    rows.sort()
    return sum(1 for a, b in zip(rows, rows[1:]) if a[1] != b[1])


def control_verdict(tel_changes, log_count, booted=True):
    if not booted:
        return "SWITCH_CONTROL_UNRESOLVED"
    tol = tolerance(log_count)
    if tel_changes >= 1 and log_count >= 1 and abs(log_count - tel_changes) <= tol:
        return "SWITCH_CONTROL_PASS"
    if tel_changes == 0 and log_count == 0:
        return "SWITCH_CONTROL_NO_SWITCH"
    if tel_changes >= 1 and (log_count == 0 or abs(log_count - tel_changes) > tol):
        return "SWITCH_LOG_UNOBSERVABLE"
    return "SWITCH_CONTROL_UNRESOLVED"


RC = {"SWITCH_CONTROL_PASS": 0, "SWITCH_LOG_UNOBSERVABLE": 1, "SWITCH_CONTROL_NO_SWITCH": 4, "SWITCH_CONTROL_UNRESOLVED": 4}


def control_segments(tag="ctl"):
    import ph_trace as T
    segs = []
    for c in range(1, CYCLES + 1):
        for name, spec in (("HI", HI), ("LO", LO)):
            ts = T.arrivals(spec["rate"], ("ctl", c, name), n=spec["n"])
            reqs = [{"record": "request", "request_id": f"{tag}-c{c}{name}-{k:04d}", "phase": name, "arrival_s": round(t, 6),
                     "input_tokens": spec["in"], "output_tokens": spec["out"], "ignore_eos": True} for k, t in enumerate(ts)]
            segs.append({"segment_id": f"{tag}-c{c}{name}", "phase": name, "round": c, "requests": reqs})
    return segs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import ph_runner as RU
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arm = P.dynamic_arm(*CONTROL_ARM)
    env = RU.boot_env(arm, out, run_id=os.environ.get("PDMUX_JOB_NAME", "control"))
    env.update(PDMUX_PHASE_EVENTS="1", PDMUX_TELEMETRY_PATH=str(out / "tel.jsonl"))       # this boot only; never a measurement boot
    srv = RU.R.ProcessServer(out, RU.server_argv(arm), env)
    booted = srv.start() == "ok"
    if booted:
        try:
            endpoint = f"http://127.0.0.1:{srv.port}/generate"
            for s in control_segments():
                RU.replay_segment(RU.default_replay, endpoint, s, out / "seg")
        finally:
            srv.stop()
    else:
        srv.stop()
    log = (out / "server.log").read_text(errors="replace") if (out / "server.log").exists() else ""
    tel = out / "tel.jsonl"
    tc = telemetry_target_changes(tel) if tel.exists() else 0
    lc = log_switch_count(log)
    verdict = control_verdict(tc, lc, booted)
    json.dump({"verdict": verdict, "log_count": lc, "tel_changes": tc, "tolerance": tolerance(lc), "booted": booted,
               "NOTE": "SWITCH_CONTROL_NONMEASUREMENT: no number here is a result"}, open(out / "CONTROL_RESULT.json", "w"), indent=1, sort_keys=True)
    print("ph_switch_control:", verdict, "log", lc, "tel", tc)
    return RC[verdict]


if __name__ == "__main__":
    sys.exit(main())
