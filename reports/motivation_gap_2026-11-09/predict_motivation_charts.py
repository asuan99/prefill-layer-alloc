#!/usr/bin/env python3
"""Prior values for the charts of every motivation measurement (2026-11-10).  PREDICTION ONLY, NOT CITABLE, not measured.

Inputs: analytic_terms.py (analytic-v1), dstar_model.py (proposal r0, decision 61, not canon), the real length pools of
results/motivation/workloads, hybrid_term_cells.py (D* cells).  Measurements are named by what they observe.
Usage: python3 predict_motivation_charts.py OUT.json   (CPU)
"""
from __future__ import annotations
import json, math, statistics as st, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid_term_cells as H  # adds costmodel and workloads to sys.path
dm, at, L = H.dm, H.at, H.L

GiB = 2**30
AVAIL = 78.74 * GiB
W_NANO = 16.58 * GiB            # boot log, as predict.py
QWEN_LSTAR = 1572               # linear_models.html (analytic-v1 s,k with public config)
nano = at.nano9b_v2()
a31 = at.nano9b_v2(31, 0)
llama = at.Arch(d=4096, n_q=32, n_kv=8, d_h=128, n_h=128, p=80, G=8, N=128, K=4, Q=128, d_ff=14336, V=128256, n_A=32, n_M=0, n_F=32, mlp_gated=True)
pool = {w: L.load_pool(w) for w in L.WORKLOADS}
out = {"status": "prediction only, not citable", "analytic": at.VERSION, "dstar": dm.VERSION}

def wbytes(a):
    return W_NANO if a is nano else at.params_total(a) * a.beta

def n_max(a, c, mem):
    return (mem * AVAIL - wbytes(a)) / (at.state_bytes_per_request(a) + c * at.kv_bytes_per_token(a))

# concurrency cap and decode memory traffic: regime map of the fixed-batch grid
ctxs = [2**i for i in range(7, 19)]
regime = []
for name, a in (("Nano", nano), ("a31", a31)):
    for c in ctxs:
        for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            ds = at.decode_step(a, b, c)
            comp = {"가중치": ds["bytes_weights"], "상태": ds["bytes_state_rw"], "KV": ds["bytes_kv_read"]}
            top = max(comp, key=comp.get)
            fit = wbytes(a) + b * at.state_bytes_per_request(a) + b * c * at.kv_bytes_per_token(a) <= 0.82 * AVAIL
            regime.append({"m": name, "c": c, "b": b, "top": top, "fit": fit})
out["regime"] = regime
out["regime_fit"] = {m: sum(r["fit"] for r in regime if r["m"] == m) for m in ("Nano", "a31")}

# binding resource: admissible concurrency against cap
cs = [int(round(256 * 2 ** (i / 2))) for i in range(0, 15)]
nm = [{"mem": m, "c": c, "n": round(n_max(nano, c, m), 1)} for m in (0.6, 0.75, 0.82) for c in cs]
out["nmax"] = nm
out["nmax_cross"] = [{"mem": m, "cap": cap, "c": round(((m * AVAIL - W_NANO) / cap - at.state_bytes_per_request(nano)) / at.kv_bytes_per_token(nano))} for m in (0.6, 0.75, 0.82) for cap in (48, 128, 256)]

# workload contexts
wl_stats = {}
for w, rows in pool.items():
    ins = [r[1] for r in rows]; outs = [r[2] for r in rows]
    wl_stats[w] = {"in_mean": st.fmean(ins), "out_mean": st.fmean(outs), "ctx": st.fmean(ins) + st.fmean(outs) / 2,
                   "over_nano": sum(i > at.L_star(nano) for i in ins) / len(ins), "over_qwen": sum(i > QWEN_LSTAR for i in ins) / len(ins)}
out["workloads"] = wl_stats
out["nano_Lstar"] = at.L_star(nano)

# arrival gap: share of inter-token intervals that contain an arrival gap
def gap_frac(lam, g, ER, c):
    t = dm.t_dec("108u", 1, c); p = 0.0
    for _ in range(300):
        b = min(48.0, lam * ER * t / 1e3)
        tb = dm.t_dec("108u", b, c)
        p = 1 - math.exp(-lam * t / 1e3)
        t = 0.5 * t + 0.5 * (tb + p * g)
    return p, b, t
sg = wl_stats["sharegpt_first"]
rates = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12]
out["gap"] = [{"g": g, "rate": r, "pct": round(100 * gap_frac(r, g, sg["out_mean"], sg["ctx"])[0], 2)} for g in (45, 64) for r in rates]
cross = {}
for g in (45, 64):
    lam = 0.01
    while gap_frac(lam, g, sg["out_mean"], sg["ctx"])[0] < 0.05: lam += 0.01
    cross[str(g)] = round(lam, 2)
out["gap_cross_5pct"] = cross

# layer-type decode: u(D) and traffic share
hw = dm.hardware_layer()
u108 = next(r["u"] for r in hw["rows"] if r["D"] == 108)
out["uD"] = [{"D": r["D"], "rel": round(r["u"] / u108, 3)} for r in hw["rows"]]
out["uD_knee"] = hw.get("D_knee", hw.get("dk"))
def layer_share(a, b, c):
    ds = at.decode_step(a, b, c)
    attn = a.n_A * at.params_attn(a) * a.beta + ds["bytes_kv_read"]
    mamba = a.n_M * at.params_mamba_matmul(a) * a.beta + ds["bytes_state_rw"]
    other = ds["bytes_total"] - attn - mamba
    return attn, mamba, other, ds["bytes_total"]
rows = []
for c in (256, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
    at_, m_, o_, t_ = layer_share(nano, 48, c)
    rows += [{"c": c, "s": n, "pct": round(100 * v / t_, 1)} for n, v in (("attention 층", at_), ("Mamba 층", m_), ("MLP·LM head", o_))]
out["layer_share"] = rows

# split dwell, split choice, arXiv output condition
dd, b0, u = H.b0_by_mode()
cells = {c["id"]: c for c in H.cells()}
def wl_of(k): return cells[k]["wl"]
named = {"사용자 첫 발화 ShareGPT": "sharegpt_first|whole", "openchat 라운드": "openchat_rounds|whole", "arXiv 요약(초록 길이)": "arxiv_summ|whole, output = abstract"}
dwell, capped = [], {}
for name, k in named.items():
    wl = wl_of(k); mom = dm.moments(wl["Ls"], 44)
    dm.CAP = 48
    r = dm.lam_cap(44, wl, 1.16, mom, step=0.01)
    capped[name] = {"lam_cap": r["lam_cap"], "f": round(r["at_cap"]["f"], 3) if r["at_cap"] else None, "binding": r["binding"]}
    lc = max(r["lam_cap"], 0.02)
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0, 1.25):
        lam = lc * frac
        dwell.append({"w": name, "x": frac, "pct": round(100 * min(1.0, lam * mom["ES"]), 1)})
out["dwell"] = dwell; out["dwell_cap"] = capped
choice = []
for name in ("사용자 첫 발화 ShareGPT", "openchat 라운드"):
    wl = wl_of(named[name])
    f = H.caps_for(wl, 1.16, 48, "full", dd); n = H.caps_for(wl, 1.16, 48, "corrected", dd)
    for D in dm.D_GRID:
        choice.append({"w": name, "D": D, "v": "포함", "lam": round(f[D][0], 2)})
        choice.append({"w": name, "D": D, "v": "제외", "lam": round(n[D][0], 2)})
out["choice"] = choice
ar = pool["arxiv_summ"]
arx = []
for name, er in (("초록 길이 출력(주 조건)", None), ("1,024 고정 출력(변형)", 1024.0)):
    wl = H.make_wl("arxiv_summ", [r[1] for r in ar], [r[2] for r in ar], er=er, ttft=2814.84, seed_key="charts")
    f = H.caps_for(wl, 1.16, 48, "full", dd)
    arx += [{"o": name, "D": D, "lam": round(f[D][0], 2), "bind": "+".join(f[D][1])} for D in dm.D_GRID]
out["arxiv"] = arx

# LooGLE interference
lg = [r[1] for r in pool["loogle_longdep"]]
lg.sort()
pts = {"p5": lg[int(.05 * len(lg))], "p50": lg[len(lg) // 2], "p95": lg[int(.95 * len(lg))]}
out["loogle_pct"] = pts
out["loogle_time"] = [{"P": lab, "L": l, "s": round(fn(l) / 1e3, 2)} for lab, fn in (("108 SM", dm.t2_108), ("64 SM (D44)", lambda x: dm.s_pre(x, 44)), ("44 SM (D64)", lambda x: dm.s_pre(x, 64))) for l in (13000, 20000, 30000, 40000, 50000, 60000, 68000)]
util, ucross = [], {}
for lab, fn in (("108 SM", dm.t2_108), ("64 SM (D44)", lambda x: dm.s_pre(x, 44)), ("44 SM (D64)", lambda x: dm.s_pre(x, 64))):
    es = st.fmean(fn(l) / 1e3 for l in lg)
    ucross[lab] = {"ES": round(es, 2), "rate_rho1": round(0.95 / 0.05 / es, 2)}
    util += [{"P": lab, "rate": r, "pct": round(100 * (0.05 / 0.95) * r * es, 1)} for r in (0.5, 1, 1.5, 2, 3, 4, 5)]
out["loogle_util"] = util; out["loogle_util_cross"] = ucross

# attention contrast
def tdec(a, b, c):
    co = at.decode_coefficients(a, dm.HBM_BW)
    return (co["a0_s"] + co["b0_s_per_req"] * b + co["c0_s_per_req_token"] * b * c) * 1e3
cc = [256, 512, 1000, 2000, 4000, 8000, 16000, 32000]
out["contrast"] = [{"m": n, "c": c, "ms": round(tdec(a, 48, c), 2)} for n, a in (("Nano-9B-v2", nano), ("a31", a31)) for c in cc]
lo, hi = 2.0, 1e6
for _ in range(80):
    mid = math.sqrt(lo * hi)
    if tdec(nano, 48, mid) > tdec(a31, 48, mid): lo = mid
    else: hi = mid
out["contrast_cross_ctx"] = round(lo)
out["contrast_nmax"] = [{"m": n, "c": c, "n": round(n_max(a, c, 0.75), 1)} for n, a in (("Nano-9B-v2", nano), ("a31", a31)) for c in cc]
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
for k in ("regime_fit", "nmax_cross", "workloads", "gap_cross_5pct", "uD", "uD_knee", "layer_share", "dwell_cap", "dwell", "choice", "arxiv", "loogle_pct", "loogle_time", "loogle_util_cross", "contrast", "contrast_cross_ctx", "contrast_nmax", "gap"):
    print(k, json.dumps(out[k], ensure_ascii=False))
print(wbytes(a31) / GiB, at.params_total(nano) * 2 / GiB)
