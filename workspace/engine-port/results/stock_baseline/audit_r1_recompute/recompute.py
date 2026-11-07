# Independent recompute: no project imports. numpy percentile + own rules.
import json, glob, os, sys, re
import numpy as np
from collections import defaultdict
B = "/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/stock_baseline/runs"
J1 = B + "/pdmux-stock-baseline-7899d1b-20261102085423/raw/runs"
JD = B + "/pdmux-stock-baseline-0e9c967-20261103192557/raw/runs"
JB = B + "/pdmux-stock-baseline-6481369-20261103192607/raw/runs"
DG = B + "/pdmux-stock-baseline-diag-4d1b9da-20261103112203/raw"
TTFT, ITL = 3000.0, 60.0

def load(d):
    out = []
    for p in sorted(glob.glob(d + "/*.json")):
        d = json.load(open(p))
        if "segments" in d: out.append(d)
    return out

def itl_from_mono(r):
    t = r.get("token_t_mono") or []
    return list(np.diff(np.array(t)) * 1000.0)

def good(r, ttft=TTFT, itl=ITL, use_mono=False):
    if r.get("error"): return False
    if r.get("ttft_ms") is None: return False
    if r["ttft_ms"] > ttft: return False
    x = itl_from_mono(r) if use_mono else r.get("token_itl_ms") or []
    if len(x) == 0: return True
    return np.percentile(np.array(x), 95) <= itl

def reqs(run): return [q for s in run["segments"] for q in s["requests"]]

def att(run, ttft=TTFT, itl=ITL, use_mono=False):
    rq = reqs(run); return sum(good(q, ttft, itl, use_mono) for q in rq) / len(rq)

def pick(points, thr=0.90):
    # rung passes iff every seed >= thr; walk ascending; lambda = last passing rung before first failing rung
    by = defaultdict(list)
    for r, a in points: by[r].append(a)
    rungs = sorted(by)
    last = None
    for r in rungs:
        if min(by[r]) >= thr: last = r
        else: break
    if last is None: return None, "BOTTOM_FAIL"
    if last == rungs[-1]: return last, "TOP_PASS"
    return last, "BRACKETED"

res = {}
j1, jd, jb = load(J1), load(JD), load(JB)
# --- structure checks
def struct(runs, name):
    nseg = set(len(r["segments"]) for r in runs)
    gpus = set(r["gpu_uuid"] for r in runs)
    jobs = set(r["job_name"] for r in runs)
    errs = sum(1 for r in runs for q in reqs(r) if q.get("error"))
    # duration vs load window
    dd = [abs(r["segments"][0]["duration_s"] - (r["segments"][0]["t_load_end_mono"] - r["segments"][0]["t_load_start_mono"])) for r in runs]
    # itl consistency
    mism = 0; tot = 0; ttftmis = []
    for r in runs:
        for q in reqs(r):
            a = np.array(q.get("token_itl_ms") or []); b = np.array(itl_from_mono(q))
            tot += 1
            if len(a) != len(b) or (len(a) and np.max(np.abs(a - b)) > 0.01): mism += 1
            if q.get("token_t_mono"):
                ttftmis.append(abs((q["token_t_mono"][0] - q["t_sent_mono"]) * 1000 - q["ttft_ms"]))
    # last completion after load end?
    over = max((max(q["completion_s"] for q in reqs(r)) - r["segments"][0]["duration_s"]) for r in runs)
    print(f"{name}: runs={len(runs)} nseg={nseg} gpus={gpus} jobs={len(jobs)} req_errors={errs} max|dur-loadwin|={max(dd):.4f} itl_vs_mono_mismatch={mism}/{tot} max|ttft-mono|={max(ttftmis):.3f}ms max(completion-dur)={over:.3f}s")
struct(j1, "J1"); struct(jd, "J1down"); struct(jb, "J1b")
# trace hash per (rate,seed) across arms
for runs, nm in ((j1, "J1"), (jd, "J1down")):
    th = defaultdict(set)
    for r in runs: th[(r["spec"]["rate"], r["seed"])].add(r["trace_hash"])
    print(nm, "cells", len(th), "max distinct trace hashes per cell", max(len(v) for v in th.values()))

# --- ladder attainments
pts = defaultdict(list)
for r in j1 + jd:
    if r.get("cell") in ("LADDER", "LADDER_DOWN"):
        pts[r["arm"]].append((r["spec"]["rate"], att(r), r["seed"], r["gpu_uuid"][:12], att(r, use_mono=True)))
print("\n== ladder (rate, seed: att) & pick")
for arm in sorted(pts):
    P = sorted(pts[arm])
    print(arm, " ".join(f"{p[0]:.4g}/s{p[2]}:{p[1]:.3f}" for p in P))
    print("   mono-ITL path identical:", all(abs(p[1]-p[4])<1e-12 for p in P))
    j1only = [(p[0], p[1]) for p in P if p[0] >= 5.516 - 1e-9]
    print("   J1-only pick:", pick(j1only), " J1+down pick:", pick([(p[0], p[1]) for p in P]))
# SLO perturbation for SG-stock / SG-off at 3.153791 (diagnostic only)
print("\n== ITL SLO perturbation (diag) on the deciding rung 3.153791 & 5.516")
for arm in ("SG-stock", "SG-off"):
    for r in jd + j1:
        if r["arm"] == arm and r["spec"]["rate"] in (3.153791, 3.62686, 5.516):
            p95 = np.array([np.percentile(q["token_itl_ms"], 95) if q["token_itl_ms"] else 0 for q in reqs(r)])
            band = np.mean((p95 >= 57) & (p95 <= 63))
            print(arm, r["spec"]["rate"], r["seed"], " ".join(f"{s}:{att(r, itl=s):.3f}" for s in (54, 57, 60, 63, 66)), f"band57-63={band:.2f}",
                  f"reqp95_med={np.median(p95):.1f}", f"ttft_fail={sum(q['ttft_ms']>TTFT for q in reqs(r))}")
    # pick with ITL 57 / 63
    P = [(p[0], p[1]) for p in pts[arm]]
    for s in (57, 63):
        PP = [(r["spec"]["rate"], att(r, itl=s)) for r in j1 + jd if r["arm"] == arm and r.get("cell") in ("LADDER", "LADDER_DOWN")]
        print(f"   {arm} pick @ITL{s}:", pick(PP))
# goodput with sum durations
print("\n== goodput check (sum durations) J1down SG-stock 3.15")
for r in jd:
    if r["arm"] == "SG-stock" and r["spec"]["rate"] == 3.153791:
        g = sum(good(q) for q in reqs(r)); D = sum(s["duration_s"] for s in r["segments"])
        print(r["seed"], g, D, g / D)

json.dump({a: [(p[0], p[2], p[1]) for p in v] for a, v in pts.items()}, open(os.path.dirname(__file__) + "/ladder.json", "w"))
