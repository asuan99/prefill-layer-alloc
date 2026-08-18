#!/usr/bin/env python3
"""Does the 'joint ~= product => nearly independent' reading have any power?

Given marginals pT, pI the joint is Frechet-bounded:
    max(0, pT+pI-1) <= joint <= min(pT, pI)
When one marginal is near 1 the window collapses and EVERY dependence structure
looks 'independent'.  This prints the window width per arm and where the
observed joint and the independence point sit inside it.
"""
import json
p = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched/audit_binding_2026-08-18/binding_2x2.json"
d = json.load(open(p))
for ph in ("HI", "LO"):
    print("==", ph)
    print(f"{'arm':4} {'pT%':>6} {'pI%':>6} {'lo%':>6} {'hi%':>6} {'width':>6} "
          f"{'indep%':>7} {'obs%':>7} {'obs-indep':>9} {'pos_in_window':>13} {'power':>6}")
    for arm, v in d["phases"][ph].items():
        pT, pI = v["ttft_pass_pct"] / 100, v["itl_p95_pass_pct"] / 100
        lo, hi = max(0.0, pT + pI - 1), min(pT, pI)
        w = hi - lo
        ind, obs = pT * pI, v["joint_pass_pct"] / 100
        pos = (obs - lo) / w if w > 0 else float("nan")
        print(f"{arm:4} {100*pT:6.2f} {100*pI:6.2f} {100*lo:6.2f} {100*hi:6.2f} {100*w:6.2f} "
              f"{100*ind:7.2f} {100*obs:7.2f} {100*(obs-ind):9.3f} {pos:13.3f} "
              f"{'YES' if 100*w > 5 else 'no':>6}")
