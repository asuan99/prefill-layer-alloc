import json, glob, numpy as np
B = "/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/stock_baseline/runs/pdmux-stock-baseline-6481369-20261103192607/raw/runs"
def good(q):
    if q.get("error") or q.get("ttft_ms") is None or q["ttft_ms"] > 3000: return False
    x = q.get("token_itl_ms") or []
    return len(x) == 0 or np.percentile(x, 95) <= 60
res = {}
for p in sorted(glob.glob(B + "/*.json")):
    r = json.load(open(p)); s = r["segments"][0]; Q = s["requests"]
    rate = r["spec"]["rate"]
    print(r["run_id"], r["arm"], rate, "n", len(Q), "dur", round(s["duration_s"], 2), "loadwin", round(s["t_load_end_mono"] - s["t_load_start_mono"], 2),
          "last_sched", round(max(q["scheduled_arrival_s"] for q in Q), 1), "last_compl", round(max(q["completion_s"] for q in Q), 1))
    if r.get("cell") != "SUSTAIN":
        print("   attainment", round(np.mean([good(q) for q in Q]), 3)); continue
    win = {w: [] for w in range(5)}; late = 0
    for q in Q:
        a = q["scheduled_arrival_s"]
        if a >= 600: late += 1; continue
        win[int(a // 120)].append(q)
    comp = []; tt = []
    for w in range(5):
        L = win[w]
        done = [q for q in L if not q.get("error") and q.get("completion_s") is not None and q["completion_s"] - (q["sent_s"] if q.get("sent_s") is not None else q["scheduled_arrival_s"]) <= 60]
        comp.append(len(done) / len(L)); tt.append(np.median([q["ttft_ms"] for q in L if q.get("ttft_ms") is not None]))
    growth = tt[-1] / tt[0]
    ok = min(comp) >= 0.95 and growth <= 2
    thr = sum(1 for q in Q if not q.get("error")) / (max(q["completion_s"] for q in Q) - min(q["sent_s"] for q in Q))
    print(f"   late={late} comp={[round(c,3) for c in comp]} ttft_p50_w={[round(t) for t in tt]} growth={growth:.2f} PASS={ok} att={np.mean([good(q) for q in Q]):.3f} "
          f"TTFT p50/95/99={np.percentile([q['ttft_ms'] for q in Q],[50,95,99]).round()} compl_thru={thr:.2f}")
    res.setdefault(r["arm"], []).append((rate, ok))
for a, v in res.items():
    ok = [x for x, y in v if y]
    print(a, "sustained lambda* =", max(ok) if ok else None)
lam = {a: max(x for x, y in v if y) for a, v in res.items()}
print("ref = min", min(lam.values()), " ratios sust/burst:", lam["PM-ref"] / 7.29491, lam["VL-stock"] / 8.389146)
print("cells R1 R2", round(0.5 * min(lam.values()), 6), round(0.8 * min(lam.values()), 6))
