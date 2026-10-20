"""ITL p95 / lambda SD forecast for G-gamma observer TOST (CPU re-aggregation, 2026-10-20).

Inputs: J1 + J2/J4/J8 N=1 anchor raw cells (manual_retrieval), shape A, arms agnostic / fused_sym
(+ J8 gpu_dev block, arm fused, as device-to-device reference).
Per cell: achieved (mg_label.load_cell, canonical), pooled token-ITL p95, median of per-request p95.
Variance components on log scale; pair-diff SD ~= sqrt(2)*SD_rep (assumption).
Forecast: half-width of 95% (alpha-split look) and 90% t-CI, P(CI inside +-ln1.015 | effect 0) by MC.
FORECAST ONLY - substrate differs from the gate (mem 0.82, old digest, power-cap exposure).
"""
import sys, os, glob, json, math
import numpy as np
from scipy import stats

BASE = "/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/multigpu"
sys.path.insert(0, BASE)
import mg_label as L  # noqa: E402

JOBS = {"J1": "pdmux-multigpu-2106f42-20261018074950", "J2": "pdmux-multigpu-e559c64-20261018114257",
        "J4": "pdmux-multigpu-213df1b-20261018114306", "J8": "pdmux-multigpu-3167f74-20261018114314"}
DELTA = math.log(1.015)
RNG = np.random.default_rng(1)


def cell_dirs():
    for j, name in JOBS.items():
        root = f"{BASE}/manual_retrieval/{name}/results/multigpu/m3_{name}/cells"
        for d in sorted(glob.glob(root + "/m3_*__A_*")):
            b = os.path.basename(d)
            if b.startswith("m3_N1_") or b.startswith("m3_gpudev_"):
                yield j, d


def boot_sampling_sd(reqs, B=2000):
    """request-bootstrap SD of log(pooled p95) and log(median per-req p95) - heuristic."""
    arrs = [np.asarray(r["token_itl_ms"]) for r in reqs]
    p95r = np.array([np.percentile(a, 95) for a in arrs])
    n = len(arrs)
    lp, lm = [], []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        lp.append(np.log(np.percentile(np.concatenate([arrs[i] for i in idx]), 95)))
        lm.append(np.log(np.median(p95r[idx])))
    return float(np.std(lp, ddof=1)), float(np.std(lm, ddof=1))


def cliff(allitl):
    """bimodal gap diagnostics: stall fraction (>30ms), margin to 5% (pct-points), local slope of log Q at q95 (%/pt)."""
    f = float(np.mean(allitl > 30.0))
    q = lambda x: float(np.percentile(allitl, x))
    return dict(stall_frac=f, margin_pts=100 * f - 5.0,
                slope_pct_per_pt=100 * (math.log(q(95.5)) - math.log(q(94.5))),
                q95=q(95), q_below_gap=q(100 * (1 - f) - 0.5), q_above_gap=q(min(99.9, 100 * (1 - f) + 0.5)))


rows = []
for job, d in cell_dirs():
    c = L.load_cell(d)
    m = c["meta"]
    if m["arm"] not in ("agnostic", "fused_sym", "fused"):
        continue
    if m["arm"] == "fused" and m.get("block") != "gpu_dev":
        continue
    reqs = [r for r in L.load_requests(os.path.join(d, "requests.jsonl"))
            if not L.is_error(r) and r.get("token_itl_ms")]
    allitl = np.concatenate([r["token_itl_ms"] for r in reqs])
    p95r = [np.percentile(r["token_itl_ms"], 95) for r in reqs]
    qs = {q: float(np.percentile(allitl, q)) for q in (50, 90, 93, 94, 95, 96, 97, 99)}
    bs_p, bs_m = boot_sampling_sd(reqs, B=400 if m["rung"] == "HI" else 1000)
    rows.append(dict(job=job, cell=m["cell_id"], arm=m["arm"], rung=m["rung"], seed=m["seed"],
                     block=m.get("block"), gpu=m.get("gpu"), host=m.get("host"), offered=m["offered"],
                     achieved=c["achieved"], status=c["status"], g=c.get("g"), n_ok=len(reqs),
                     itl_pool_p95=qs[95], itl_rq95_med=float(np.median(p95r)), itl_q=qs,
                     frac_gt_30ms=float(np.mean(allitl > 30.0)), cliff=cliff(allitl),
                     bsd_pool=bs_p, bsd_rq=bs_m))

METRICS = ["achieved", "itl_pool_p95", "itl_rq95_med"]


def pooled_sd(groups):
    """pooled within-group SD of log values: sqrt(sum SS / sum df)."""
    ss, df = 0.0, 0
    for g in groups:
        g = np.log(np.asarray(g, float))
        if len(g) >= 2:
            ss += float(np.sum((g - g.mean()) ** 2)); df += len(g) - 1
    return (math.sqrt(ss / df) if df else None), df


comp = {}
for arm in ("agnostic", "fused_sym"):
    for rung in ("HI", "LO"):
        sub = [r for r in rows if r["arm"] == arm and r["rung"] == rung and r["block"] != "gpu_dev"]
        if not sub:
            continue
        seeds = sorted({r["seed"] for r in sub})
        jobs = sorted({r["job"] for r in sub})
        for met in METRICS:
            # between-Job (same arm, rung, seed; different Job/host)
            bj, dfj = pooled_sd([[r[met] for r in sub if r["seed"] == s] for s in seeds])
            # between-seed within Job (cancels in pair; context / upper bound)
            bs, dfs = pooled_sd([[r[met] for r in sub if r["job"] == j] for j in jobs])
            comp[(arm, rung, met)] = dict(between_job=bj, df_job=dfj, between_seed=bs, df_seed=dfs,
                                          n_cells=len(sub))
        for met in METRICS:
            # two-way additive (seed + Job) residual: removes Job(host+boot) offset -> cell-level noise (lower bound)
            if len(jobs) >= 2 and len(seeds) >= 2:
                Y = np.array([[math.log([r[met] for r in sub if r["job"] == j and r["seed"] == s0][0]) for s0 in seeds] for j in jobs])
                R = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
                dfr = (len(jobs) - 1) * (len(seeds) - 1)
                comp[(arm, rung, met)]["twoway_resid"] = math.sqrt(float((R ** 2).sum()) / dfr)
                comp[(arm, rung, met)]["job_offsets"] = {j: float(o) for j, o in zip(jobs, Y.mean(1) - Y.mean())}
        for met, key in (("itl_pool_p95", "bsd_pool"), ("itl_rq95_med", "bsd_rq")):
            comp[(arm, rung, met)]["req_bootstrap"] = float(np.sqrt(np.mean([r[key] ** 2 for r in sub])))

gd = [r for r in rows if r["block"] == "gpu_dev"]
gpudev = {met: float(np.std(np.log([r[met] for r in gd]), ddof=1)) for met in METRICS} if gd else {}
gpudev_bs = {"itl_pool_p95": float(np.sqrt(np.mean([r["bsd_pool"] ** 2 for r in gd]))),
             "itl_rq95_med": float(np.sqrt(np.mean([r["bsd_rq"] ** 2 for r in gd])))} if gd else {}


def halfwidth(sd_rep, n, conf):
    sd_d = math.sqrt(2) * sd_rep
    return stats.t.ppf(1 - (1 - conf) / 2, n - 1) * sd_d / math.sqrt(n)


def p_pass(sd_rep, n, conf, mu=0.0, M=40000):
    sd_d = math.sqrt(2) * sd_rep
    x = RNG.normal(mu, sd_d, size=(M, n))
    mean, s = x.mean(1), x.std(1, ddof=1)
    h = stats.t.ppf(1 - (1 - conf) / 2, n - 1) * s / math.sqrt(n)
    return float(np.mean((mean - h > -DELTA) & (mean + h < DELTA)))


def p_pass_two_stage(sd_rep, n1=10, n2=16, conf=0.95, mu=0.0, M=40000):
    """look1 at n1: PASS / OBSERVER_EFFECT(CI excl 0) stop; UNRESOLVED_POWER -> extend to n2."""
    sd_d = math.sqrt(2) * sd_rep
    x = RNG.normal(mu, sd_d, size=(M, n2))
    def look(k):
        m, s = x[:, :k].mean(1), x[:, :k].std(1, ddof=1)
        h = stats.t.ppf(1 - (1 - conf) / 2, k - 1) * s / math.sqrt(k)
        return m, h
    m1, h1 = look(n1)
    pass1 = (m1 - h1 > -DELTA) & (m1 + h1 < DELTA)
    eff1 = (m1 - h1 > 0) | (m1 + h1 < 0)
    m2, h2 = look(n2)
    pass2 = (~pass1) & (~eff1) & (m2 - h2 > -DELTA) & (m2 + h2 < DELTA)
    return float(np.mean(pass1)), float(np.mean(pass1 | pass2)), float(np.mean(eff1 & ~pass1))


def sd_needed(n, conf, target=0.80):
    lo, hi = 1e-4, 0.05
    for _ in range(30):
        mid = (lo + hi) / 2
        if p_pass(mid, n, conf, M=20000) >= target:
            lo = mid
        else:
            hi = mid
    return lo


forecast = {}
for k, v in comp.items():
    for src in ("between_job", "twoway_resid", "req_bootstrap"):
        sd = v.get(src)
        if sd is None:
            continue
        fk = f"{k[0]}|{k[1]}|{k[2]}|{src}"
        forecast[fk] = dict(sd_rep=sd, sd_pair=math.sqrt(2) * sd,
                            hw95={n: halfwidth(sd, n, 0.95) for n in (8, 10, 16)},
                            hw90={n: halfwidth(sd, n, 0.90) for n in (8, 10, 16)},
                            ppass95={n: p_pass(sd, n, 0.95) for n in (8, 10, 16)},
                            ppass90={n: p_pass(sd, n, 0.90) for n in (8, 10, 16)},
                            two_stage=p_pass_two_stage(sd))
for k, sd in gpudev.items():
    forecast[f"fused(gpudev)|HI|{k}|between_gpu_boot"] = dict(
        sd_rep=sd, sd_pair=math.sqrt(2) * sd,
        hw95={n: halfwidth(sd, n, 0.95) for n in (8, 10, 16)},
        ppass95={n: p_pass(sd, n, 0.95) for n in (8, 10, 16)}, two_stage=p_pass_two_stage(sd))

need = {n: {"95": sd_needed(n, 0.95), "90": sd_needed(n, 0.90)} for n in (8, 10, 16)}

out = dict(rows=rows, components={"|".join(k): v for k, v in comp.items()}, gpudev_sd=gpudev,
           gpudev_req_bootstrap=gpudev_bs, forecast=forecast, sd_rep_needed_for_80pct_pass=need,
           delta=DELTA, note="FORECAST ONLY; substrate J1-J8 (mem 0.82, old digest) != gate (mem 0.6, new digest)")
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "itl_sd_forecast.json")
json.dump(out, open(dst, "w"), indent=1, default=str)

pct = lambda x: f"{100 * x:.2f}%" if x is not None else "NA"
print("== cells")
for r in sorted(rows, key=lambda r: (r["arm"], r["rung"], r["seed"], r["job"], str(r["gpu"]))):
    print(f"{r['job']} {r['arm']:9s} {r['rung']} s{r['seed']:<5d} g{r['gpu']} ach={r['achieved']:.4f} "
          f"pool95={r['itl_pool_p95']:.2f} rq95med={r['itl_rq95_med']:.2f} q93/94/96/97="
          f"{r['itl_q'][93]:.1f}/{r['itl_q'][94]:.1f}/{r['itl_q'][96]:.1f}/{r['itl_q'][97]:.1f} "
          f">30ms={100*r['frac_gt_30ms']:.2f}% bsd={pct(r['bsd_pool'])}/{pct(r['bsd_rq'])} {r['status']}")
print("== LO cliff")
for r in rows:
    if r["rung"] == "LO":
        print(r["job"], r["arm"], r["seed"], "ach", round(r["achieved"], 3), {k: round(v, 3) for k, v in r["cliff"].items()})
print("== components (SD of log)")
for k, v in comp.items():
    print(k, {kk: (pct(vv) if isinstance(vv, float) else vv) for kk, vv in v.items()})
print("gpudev", {k: pct(v) for k, v in gpudev.items()}, "req-boot", {k: pct(v) for k, v in gpudev_bs.items()})
print("== forecast")
for k, f in forecast.items():
    print(k, "sd_rep", pct(f["sd_rep"]), "hw95", {n: pct(h) for n, h in f["hw95"].items()},
          "P95", {n: round(p, 3) for n, p in f["ppass95"].items()},
          "P90", {n: round(p, 3) for n, p in f.get("ppass90", {}).items()},
          "2stage(P1,Ptot,Peff1)", tuple(round(x, 3) for x in f["two_stage"]))
print("== sd_rep needed for 80% pass at effect 0", {n: {c: pct(v) for c, v in d.items()} for n, d in need.items()})
