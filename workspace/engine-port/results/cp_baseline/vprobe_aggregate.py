#!/usr/bin/env python3
"""Aggregate the V-probe: the untreated variance the 5th audit's H1 asked for.

Consumes every `vprobe_*/vprobe_scores.json` and reports, PER ARM:
  * the null Delta distribution (same arm, two boots, same job, same node)
  * its SD -- the number a future equivalence margin must clear
  * its MEAN, tested against zero.  A null that is not centred at zero is a
    within-job ORDER EFFECT, and it biases any A-then-B comparison run in one job.
  * the MDE arithmetic at n=4 for a paired one-sided/two-sided decision
  * the same, per phase (LO / HI) and across the 9-point SLO grid

It computes NO cross-arm quantity: arms are reported side by side because they are
different populations, not because they are being compared (PREREG_VPROBE sec 0).

Usage:  python3 vprobe_aggregate.py [dir]
"""
import json, math, statistics, sys
from pathlib import Path

# one-sided/two-sided Student-t, df = n-1 (same table as g1b_plateau_predicates)
T95_1S = {2: 6.314, 3: 2.920, 4: 2.353, 5: 2.132, 6: 2.015, 7: 1.943, 8: 1.895}
T975_2S = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}
MARGIN = 0.03          # CP-2 rev1's registered equivalence margin, under test here


def rel(a, b):
    return (b - a) / a


def summarize(name, deltas):
    n = len(deltas)
    if n < 2:
        return {"n": n, "note": "fewer than 2 cells: no SD"}
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(n)
    t2 = T975_2S.get(n)
    out = {
        "n": n, "mean": mean, "sd": sd, "se": se,
        "abs_max": max(abs(d) for d in deltas),
        "ci95": [mean - t2 * se, mean + t2 * se] if t2 else None,
        "mean_differs_from_zero": bool(t2 and abs(mean) > t2 * se),
    }
    return out


def mde(sd, n, margin=MARGIN):
    """What a paired comparison at this n could actually decide.

    A direction label needs the whole CI beyond the margin:  |Delta| > margin + t*SE.
    `equivalent` (TOST) needs the whole CI inside it:         t*SE < margin.
    """
    t2 = T975_2S.get(n)
    if not t2:
        return None
    se = sd / math.sqrt(n)
    return {
        "n": n, "sd": sd, "se": se, "half_width": t2 * se,
        "min_abs_delta_for_a_direction_label": margin + t2 * se,
        "equivalent_reachable": t2 * se < margin,
    }


def main(root="."):
    root = Path(root)
    cells = []
    for f in sorted(root.glob("vprobe_*/vprobe_scores.json")):
        d = json.loads(f.read_text())
        if d.get("boots_scored", 0) < 2:
            print(f"  [skip] {f.parent.name}: {d.get('boots_scored')} boot(s) scored "
                  f"-- no null Delta (this is a measurement outcome, not a zero)")
            continue
        b = d["per_boot"]
        cells.append({
            "job": f.parent.name, "arm": d["arm"], "seed": b[0]["seed"],
            "combined": rel(b[0]["combined_goodput_req_s"], b[1]["combined_goodput_req_s"]),
            "lo": rel(b[0]["lo_goodput_req_s"], b[1]["lo_goodput_req_s"]),
            "hi": rel(b[0]["hi_goodput_req_s"], b[1]["hi_goodput_req_s"]),
            "band": d.get("null_delta_rel_by_slo_point", {}),
            "g1": b[0]["combined_goodput_req_s"], "g2": b[1]["combined_goodput_req_s"],
            "hi_g1": b[0]["hi_goodput_req_s"], "hi_g2": b[1]["hi_goodput_req_s"],
        })

    arms = sorted({c["arm"] for c in cells})
    report = {
        "_what_this_is": "V-probe aggregate: untreated (same-arm, two-boot, one-job) "
                         "variance of the change-trace goodput estimator. No policy "
                         "verdict; no cross-arm quantity is computed.",
        "margin_under_test": MARGIN, "cells": cells, "per_arm": {},
    }

    for arm in arms:
        cs = [c for c in cells if c["arm"] == arm]
        a = {"cells": len(cs), "seeds": [c["seed"] for c in cs]}
        for key in ("combined", "lo", "hi"):
            a[key] = summarize(f"{arm}.{key}", [c[key] for c in cs])
            a[key]["mde_n4"] = mde(a[key]["sd"], 4) if a[key].get("sd") else None
        pts = sorted({k for c in cs for k in c["band"]},
                     key=lambda s: (float(s.split("/")[0]), float(s.split("/")[1])))
        a["band"] = {p: summarize(p, [c["band"][p] for c in cs if p in c["band"]])
                     for p in pts}
        report["per_arm"][arm] = a

    (root / "vprobe_aggregate.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

    print("=" * 78)
    print("V-PROBE AGGREGATE -- untreated variance of the paired goodput estimator")
    print(f"  margin under test: {MARGIN:.3f}   cells: {len(cells)}   arms: {arms}")
    print("=" * 78)
    for arm in arms:
        a = report["per_arm"][arm]
        print(f"\n### {arm}   (n={a['cells']} jobs, seeds {a['seeds']})")
        for key, label in (("combined", "COMBINED"), ("lo", "LO   (rate 3)"),
                           ("hi", "HI   (rate 12)")):
            s = a[key]
            if "sd" not in s:
                print(f"  {label}: {s.get('note')}")
                continue
            m = s["mde_n4"]
            print(f"  {label}:  null Δ mean {s['mean']:+.4f}  SD {s['sd']:.4f}  "
                  f"|max| {s['abs_max']:.4f}")
            print(f"      95% CI of the mean [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"
                  f"  -> mean differs from 0: {'YES' if s['mean_differs_from_zero'] else 'no'}")
            print(f"      at n=4: half-width {m['half_width']:.4f} | a direction label "
                  f"needs |Δ| > {m['min_abs_delta_for_a_direction_label']:.4f} | "
                  f"`equivalent` reachable: {'YES' if m['equivalent_reachable'] else 'NO'}")
    print(f"\n  wrote {root / 'vprobe_aggregate.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
