import json, glob, numpy as np
R = "/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/stock_baseline/runs"
def good(q, itl=60, tt=3000):
    if q.get("error") or q["ttft_ms"] > tt: return False
    x = q["token_itl_ms"]; return len(x) == 0 or np.percentile(x, 95) <= itl
def load(p): return json.load(open(p))
allr = [load(p) for p in glob.glob(R + "/pdmux-stock-baseline-*/raw/runs/*.json")]
allr = [r for r in allr if "segments" in r]
# TTFT cliff on the capacity-deciding rungs
print("== TTFT cliff (TTFT within 3000+-5%) on deciding rungs")
for r in sorted(allr, key=lambda r: (r["arm"], r["spec"]["rate"], r["seed"])):
    if (r["arm"], r["spec"]["rate"]) in {("PM-ref", 7.29491), ("PM-ref", 8.389146), ("VL-stock", 8.389146), ("VL-stock", 9.647518), ("SG-stock", 3.153791)}:
        Q = r["segments"][0]["requests"]; t = np.array([q["ttft_ms"] for q in Q])
        a = np.mean([good(q) for q in Q]); a95 = np.mean([good(q, tt=2850) for q in Q]); a105 = np.mean([good(q, tt=3150) for q in Q])
        print(r["run_id"], r["gpu_uuid"][4:12], f"att={a:.3f} cliff={np.mean((t>=2850)&(t<=3150)):.3f} att@TTFT2850={a95:.3f} att@3150={a105:.3f}")
# SG-off vs SG-stock paired per (rate, seed)
print("== SG-off minus SG-stock paired attainment")
idx = {(r["arm"], r["spec"]["rate"], r["seed"], r.get("cell")): r for r in allr}
d = []
for (arm, rate, seed, cell), r in idx.items():
    if arm == "SG-stock" and cell in ("LADDER", "LADDER_DOWN"):
        o = idx.get(("SG-off", rate, seed, cell))
        a1 = np.mean([good(q) for q in r["segments"][0]["requests"]]); a2 = np.mean([good(q) for q in o["segments"][0]["requests"]])
        assert r["trace_hash"] == o["trace_hash"]
        d.append((rate, seed, a2 - a1))
d.sort(); dd = np.array([x[2] for x in d])
print(f"n={len(dd)} mean={dd.mean():+.4f} SD={dd.std(ddof=1):.4f} max|d|={np.abs(dd).max():.3f} at", [x for x in d if abs(x[2]) > 0.05])
low = np.array([x[2] for x in d if x[0] < 5])
print(f"  J1-down rungs only: n={len(low)} mean={low.mean():+.4f} SD={low.std(ddof=1):.4f}")
# J1b REF gap p50 (same GPU as diag)
print("== PM-ref gap on GPU 27097514 (J1b REF) vs diag B5")
for r in allr:
    if r["run_id"].startswith("J1b_REF") or (r["arm"] == "PM-ref" and r["spec"]["rate"] == 7.29491 and r.get("cell") == "LADDER"):
        x = np.concatenate([q["token_itl_ms"] for q in r["segments"][0]["requests"] if q["token_itl_ms"]])
        print(r["run_id"], r["gpu_uuid"][4:12], f"tok>40={np.mean(x>40):.3f} gap p50={np.median(x[x>40]):.1f} base p50={np.median(x[x<=40]):.1f}")
