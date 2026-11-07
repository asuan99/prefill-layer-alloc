import json, glob, numpy as np
R = "/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/stock_baseline/runs"
pats = ["/pdmux-stock-baseline-7899d1b-*/raw/runs/J1_*_lad_r5.516_s20*.json", "/pdmux-stock-baseline-0e9c967-*/raw/runs/J1-down_SG-*_lad_r3.153791_s20*.json",
        "/pdmux-stock-baseline-diag-*/raw/B*/runs/*.json"]
for pat in pats:
  for p in sorted(glob.glob(R + pat)):
    r = json.load(open(p)); Q = r["segments"][0]["requests"]
    tag = ("DIAG_" + p.split("/raw/")[1].split("/")[0]) if "diag" in p else r["run_id"]
    rows = []
    for j in Q:
        s = j["t_sent_mono"]; F = s + j["ttft_ms"] / 1000
        if any(o is not j and -0.15 < o["t_sent_mono"] - s < F - s + 0.15 for o in Q): continue  # no other arrival during j's prefill +-150ms
        ex = []
        for i in Q:
            if i is j: continue
            t = np.array(i["token_t_mono"])
            if len(t) < 5 or t[0] >= s - 0.3 or t[-1] <= F + 0.05: continue
            m = (t > s) & (t <= F + 0.1)
            idx = np.where(m)[0]
            if len(idx) == 0: continue
            d = [(t[k] - t[k-1]) * 1000 for k in idx if k > 0]
            pre = np.diff(t[(t >= s - 0.5) & (t <= s)]) * 1000; pre = pre[pre <= 40]
            if len(pre) < 3 or not d: continue
            ex.append(max(d) - np.median(pre))
        if ex: rows.append((j["input_tokens"], np.median(ex)))
    rows = np.array(rows)
    out = []
    for lo, hi in [(0, 32), (32, 128), (128, 512), (512, 1025), (1025, 10**6)]:
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        out.append(f"[{lo},{hi}):{np.median(rows[m,1]):.1f}(n={m.sum()})" if m.sum() else f"[{lo},{hi}):-")
    short = rows[rows[:, 0] < 512]
    rk = lambda x: np.argsort(np.argsort(x))
    sp = np.corrcoef(rk(short[:, 0]), rk(short[:, 1]))[0, 1] if len(short) > 3 else float('nan')
    print(f"{tag} {r['arm']}: n_iso={len(rows)} " + " ".join(out) + f" | <512 tok: median={np.median(short[:,1]):.1f} IQR=({np.percentile(short[:,1],25):.1f},{np.percentile(short[:,1],75):.1f}) spearman(len,excess)={sp:.2f}")
