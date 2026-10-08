#!/usr/bin/env python3
"""Lag-aware re-analysis of the sensor-bandwidth probe (CPU only).

For each source (NVML power usage; energy counter -> power via finite difference over each update),
(1) step response at the continuous-load reference block: delay to 10/50/90 % of the swing after the
load-on edge and after the load-off edge; (2) per square-wave period P: modulation depth at the best
lag L in [0, 600] ms = (mean sensor in ON halves shifted by L - mean in OFF halves shifted by L) /
reference swing, and the lag that maximises it.
Usage: analyze_lag.py <job_dir>
"""
import bisect, csv, json, statistics as st, sys
from pathlib import Path

d = Path(sys.argv[1])
S = list(csv.DictReader(open(d / "samples.csv")))
E = list(csv.DictReader(open(d / "edges.csv")))
t = [float(r["t"]) for r in S]
pw = [float(r["usage_mW"]) / 1000 for r in S]
en = [float(r["energy_mJ"]) for r in S]

# counter -> power: at each counter change, power over the interval since the previous change
ct, cp = [], []
last_i = 0
for i in range(1, len(en)):
    if en[i] != en[last_i]:
        dt_ = t[i] - t[last_i]
        if dt_ > 0:
            ct.append(0.5 * (t[i] + t[last_i])); cp.append((en[i] - en[last_i]) / 1000 / dt_)
        last_i = i
# NVML power usage: keep only change points (value held until next update)
pt, pp = [t[0]], [pw[0]]
for i in range(1, len(pw)):
    if pw[i] != pw[i - 1]:
        pt.append(t[i]); pp.append(pw[i])
SRC = {"power_usage": (pt, pp, "hold"), "energy_counter": (ct, cp, "interval_mid")}
if (d / "buffered.csv").exists():   # v2: NVML buffered power samples (timeStamp us, CPU epoch)
    rows = [ln.strip().split(",") for ln in open(d / "buffered.csv") if ln[0].isdigit()]
    SRC["buffered_samples"] = ([int(x[0]) / 1e6 for x in rows], [int(x[1]) / 1000 for x in rows], "hold")


def val_at(src, x):
    ts, vs, _ = src
    i = bisect.bisect_right(ts, x) - 1
    return vs[i] if 0 <= i < len(vs) else None


def mean_over(src, a, b, step=0.001):
    xs = [a + k * step for k in range(int((b - a) / step) + 1)]
    v = [val_at(src, x) for x in xs]
    v = [x for x in v if x is not None]
    return st.fmean(v) if v else None


edges = {}
for r in E:
    edges.setdefault(int(r["P_ms"]), []).append((r["edge"], float(r["t"])))
ref = dict((k, x) for k, x in edges[0])
on0, off0, end0 = ref["on"], ref["off"], ref["end"]
out = {"reference_block_s": {"on": off0 - on0, "off": end0 - off0}, "sources": {}}
for name, src in SRC.items():
    hi = mean_over(src, on0 + 0.5 * (off0 - on0), off0 - 0.2)
    lo = mean_over(src, off0 + 0.5 * (end0 - off0), end0 - 0.1)
    sw = hi - lo
    def cross(level, start, rising):
        x = start
        while x < start + 2.0:
            v = val_at(src, x)
            if v is not None and ((v >= level) if rising else (v <= level)):
                return 1000 * (x - start)
            x += 0.001
        return None
    step = {f"on_{int(f*100)}pct_ms": cross(lo + f * sw, on0, True) for f in (0.1, 0.5, 0.9)}
    step |= {f"off_{int(f*100)}pct_ms": cross(hi - f * sw, off0, False) for f in (0.1, 0.5, 0.9)}
    per = {}
    for P in sorted(k for k in edges if k):
        ed = edges[P]
        halves = [(k, a, b) for (k, a), (_, b) in zip(ed, ed[1:])]
        halves = halves[len(halves) // 4: 3 * len(halves) // 4]
        best = (None, None)
        curve = {}
        for L in range(-100, 601, 5):
            Ls = L / 1000
            on = [mean_over(src, a + Ls, b + Ls) for k, a, b in halves if k == "on"]
            off = [mean_over(src, a + Ls, b + Ls) for k, a, b in halves if k == "off"]
            on = [x for x in on if x is not None]; off = [x for x in off if x is not None]
            if on and off:
                dep = (st.fmean(on) - st.fmean(off)) / sw
                curve[L] = dep
                if best[0] is None or dep > best[0]:
                    best = (dep, L)
        per[str(P)] = {"best_depth": best[0], "best_lag_ms": best[1], "depth_at_lag0": curve.get(0)}
    out["sources"][name] = {"n_updates": len(src[0]), "swing_W": sw, "high_W": hi, "low_W": lo,
                            "step": step, "periods": per}
json.dump(out, open(d / "lag_analysis.json", "w"), indent=1)
for name, s in out["sources"].items():
    print(f"## {name}: swing {s['swing_W']:.0f} W ({s['low_W']:.0f} -> {s['high_W']:.0f}), updates {s['n_updates']}")
    print("   step:", {k: (round(v) if v is not None else None) for k, v in s["step"].items()})
    for P, r in s["periods"].items():
        print(f"   P={P:>3} ms  best depth {r['best_depth']:.2f} at lag {r['best_lag_ms']} ms  (lag0 {r['depth_at_lag0']:.2f})")
