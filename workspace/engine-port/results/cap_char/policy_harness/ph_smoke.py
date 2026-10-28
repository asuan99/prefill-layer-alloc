#!/usr/bin/env python3
"""Q-3 smoke for the policy PREREG (NON-MEASUREMENT; numbers are never cited): does the engine do what the registered arms assume?

Boots (Nano, flashinfer, cudagraph ON, the D64 candidate config, non-sticky -- decision 49), PHASE_EVENTS + telemetry ON for the smoke only:
  s16, s64 : static non-sticky `R2_POLICY=fixed` -- the realized (v1, legacy) (prefill_sms, decode_sms) of prefill_span_start must equal the
             requested D for ~all spans AND differ between D16 and D64.  If the two splits are the same the seal is VOID
             (audit r2 caveat 1: 'smoke에서 D 간 분할이 같으면 이 봉인은 무효다').
  dyn      : bind+GATE (anchor idx 3 = D34, dwell 3) boots and decodes; reports the realized splits and the switch count.
v2 (rev1.1, authoritative): the observable is the co-resident decode_iteration split, see smoke_verdict_v2; v1 (span_start) is legacy.
Verdicts (v2): SMOKE_PASS | SMOKE_VOID_SAME_SPLIT | SMOKE_FAIL_NOT_APPLIED | SMOKE_FAIL_BOOT | SMOKE_UNRESOLVED.   rc 0 / 1 / 1 / 1 / 4.
v2 order: boot -> N_CO_MIN -> mode(s16) != mode(s64) -> frac_at_target >= APPLY_FRAC -> dyn N_CO_MIN.  The mode test is redundant once both
arms sit >= 95 % on different targets; it only decides which LABEL a failing run gets (audit r1, issue 2).  The dyn arm is checked for
boot and co-resident volume only -- its decision-change record (Q-3, second half) is the job of ph_switch_control.py (PREREG §17).
(The module header above describes the legacy v1 criterion, which counted prefill_span_start -- logged BEFORE the split transition.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))
import ph_spec as P  # noqa: E402

APPLY_FRAC = 0.95
N_CO_MIN = 30          # v2: minimum co-resident decode iterations per static arm, else UNRESOLVED (fixed before the re-score)


def realized_pairs(tel_path):
    c = Counter()
    for ln in open(tel_path):
        if '"prefill_span_start"' not in ln:
            continue
        d = json.loads(ln)
        if d.get("event") == "prefill_span_start":
            c[(int(d["prefill_sms"]), int(d["decode_sms"]))] += 1
    return c


def smoke_verdict(pairs_by_arm, booted=None):
    """pairs_by_arm: {'s16': Counter, 's64': Counter, 'dyn': Counter}; booted: {arm: bool}."""
    booted = booted or {a: True for a in pairs_by_arm}
    if not all(booted.get(a, False) for a in ("s16", "s64", "dyn")):
        return "SMOKE_FAIL_BOOT", {a: booted.get(a) for a in ("s16", "s64", "dyn")}
    info, modes = {}, {}
    for a, D in (("s16", 16), ("s64", 64)):
        c = pairs_by_arm.get(a) or Counter()
        n = sum(c.values())
        if n == 0:
            return "SMOKE_UNRESOLVED", {"why": f"{a}: no prefill_span_start events"}
        dec = Counter()
        for (p, d), k in c.items():
            dec[d] += k
        modes[a] = dec.most_common(1)[0][0]
        info[a] = {"spans": n, "decode_sms_hist": dict(dec), "frac_at_target": dec.get(D, 0) / n}
    if modes["s16"] == modes["s64"]:                    # the registered arms are not different arms: the seal is void (caveat 1)
        return "SMOKE_VOID_SAME_SPLIT", info
    for a, D in (("s16", 16), ("s64", 64)):
        if info[a]["frac_at_target"] < APPLY_FRAC:
            return "SMOKE_FAIL_NOT_APPLIED", info
    dyn = pairs_by_arm.get("dyn") or Counter()
    info["dyn"] = {"spans": sum(dyn.values()), "distinct_decode_sms": sorted({d for (_, d) in dyn})}
    return "SMOKE_PASS", info


def realized_coresident(tel_path):
    """v2 observable (rev1.1): decode_sms of `decode_iteration` events with prefill_in_flight AND NOT prefill_kernel_done_wait --
    the engine_gate_diag D9 definition.  Unlike prefill_span_start (logged BEFORE the split transition) this is the split the
    co-resident decode actually ran in."""
    c = Counter()
    for ln in open(tel_path):
        if '"decode_iteration"' not in ln:
            continue
        d = json.loads(ln)
        if d.get("event") == "decode_iteration" and d.get("prefill_in_flight") and not d.get("prefill_kernel_done_wait"):
            c[int(d["decode_sms"])] += 1
    return c


def smoke_verdict_v2(co_by_arm, booted=None):
    """co_by_arm: {'s16'|'s64'|'dyn': Counter{decode_sms: n_coresident_iterations}}.  v1 `smoke_verdict` is kept for the record."""
    booted = booted or {a: True for a in co_by_arm}
    if not all(booted.get(a, False) for a in ("s16", "s64", "dyn")):
        return "SMOKE_FAIL_BOOT", {a: booted.get(a) for a in ("s16", "s64", "dyn")}
    info, modes = {}, {}
    for a, D in (("s16", 16), ("s64", 64)):
        c = co_by_arm.get(a) or Counter()
        n = sum(c.values())
        info[a] = {"n_coresident": n, "decode_sms_hist": dict(c), "frac_at_target": (c.get(D, 0) / n) if n else None}
        if n < N_CO_MIN:
            return "SMOKE_UNRESOLVED", dict(info, why=f"{a}: {n} < {N_CO_MIN} co-resident iterations")
        modes[a] = c.most_common(1)[0][0]
    dyn = co_by_arm.get("dyn") or Counter()
    info["dyn"] = {"n_coresident": sum(dyn.values()), "distinct_decode_sms": sorted(dyn)}
    if modes["s16"] == modes["s64"]:
        return "SMOKE_VOID_SAME_SPLIT", info
    for a, D in (("s16", 16), ("s64", 64)):
        if info[a]["frac_at_target"] < APPLY_FRAC:
            return "SMOKE_FAIL_NOT_APPLIED", info
    if info["dyn"]["n_coresident"] < N_CO_MIN:
        return "SMOKE_UNRESOLVED", dict(info, why="dyn: too few co-resident iterations")
    return "SMOKE_PASS", info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--w1-lengths", required=True)
    a = ap.parse_args(argv)
    import ph_runner as RU
    import ph_trace as T
    R = RU.R
    w1 = json.load(open(a.w1_lengths))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = {"s16": P.static_arm(16), "s64": P.static_arm(64), "dyn": P.dynamic_arm(3, 3)}
    pairs, booted, co = {}, {}, {}
    for name, arm in arms.items():
        bdir = out / name
        bdir.mkdir(exist_ok=True)
        env = RU.boot_env(arm, bdir, run_id=os.environ.get("PDMUX_JOB_NAME", "smoke"))
        env.update(PDMUX_PHASE_EVENTS="1", PDMUX_TELEMETRY_PATH=str(bdir / "tel.jsonl"))   # smoke only; never in a measurement boot
        srv = R.ProcessServer(bdir, RU.server_argv(arm), env)
        booted[name] = srv.start() == "ok"
        if not booted[name]:
            srv.stop()
            continue
        try:
            endpoint = f"http://127.0.0.1:{srv.port}/generate"
            segs = T.he0_trace(w1, 1.2, 4.8, 60, 1, 0, f"smoke_{name}")             # LO then HI, placeholder rates
            for s in segs:
                RU.replay_segment(RU.default_replay, endpoint, s, bdir / "seg")
        finally:
            srv.stop()
        tel = bdir / "tel.jsonl"
        pairs[name] = realized_pairs(tel) if tel.exists() else Counter()
        co[name] = realized_coresident(tel) if tel.exists() else Counter()
    verdict_v1, info_v1 = smoke_verdict(pairs, booted)           # legacy (span_start-based; kept for the record)
    verdict, info = smoke_verdict_v2(co, booted)
    json.dump({"verdict": verdict, "info": info, "legacy_v1": {"verdict": verdict_v1, "info": info_v1}, "NOTE": "SMOKE_UNSEALED_NONMEASUREMENT: no number here is a result"},
              open(out / "smoke.json", "w"), indent=1, sort_keys=True)
    print("ph_smoke:", verdict)
    return {"SMOKE_PASS": 0, "SMOKE_UNRESOLVED": 4}.get(verdict, 1)


if __name__ == "__main__":
    sys.exit(main())
