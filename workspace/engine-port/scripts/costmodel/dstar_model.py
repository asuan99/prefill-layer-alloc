#!/usr/bin/env python3
"""D* model: static decode-SM split predicted from hardware + model terms + measured cost terms (PROPOSAL, 2026-10-30).

Status: proposal (decision 61).  Not canon.  It does not change analytic-v1; it imports it as the prior and stacks the
measured cost terms (T1, T2, T2', T3) and a lockstep PD-mux serving model on top -- the "PD-mux SM split" extension row
of ANALYTIC_COST_FORMULAS.md section 8.  Document: reports/cost_model/PROPOSAL_dstar_model_2026-10-30.md.

Three layers (each number in the output carries one of these labels):
  [derived]    closed form from analytic_terms.py (hardware BW, model architecture)
  [measured]   coefficient copied from a cost-term result (source pointer next to it)
  [model]      output of the serving model below (assumptions A1-A9 in the document)

Serving model (non-sticky lockstep PD-mux, SGLang public loop; decision 49):
  P = 108 - D.  While a prefill is in flight and decode is non-empty, decode runs on D SMs (slowed by rho) and prefill
  on P SMs; otherwise decode runs on all 108 SMs ("108u").
    S_pre(L)  = kappa * T2(L; P=108) * 108 / P            prefill service time of one request
    f         = min(1, lam * E[S_pre])                    co-location fraction (prefill busy fraction)
    T_co(b)   = rho * T_dec(D, b, c)                      decode step during co-location
    T_solo(b) = T_dec(108u, b, c)                         decode step otherwise
    T_eff     = f * T_co + (1 - f) * T_solo
    b         = lam * E[R] * T_eff                        decode occupancy (Little), fixed point
  Feasible at lam iff
    ITL : step that sets the request-internal p95 (T_co if f > 0.05 else T_solo), at b95 = b + 1.645 sqrt(b) (<= cap),
          is <= S_ITL
    SLOT: lam * (E[R] * T_eff + E[S_pre] + W) <= cap      (cap = max_running_requests)
    TTFT: S_pre(L_q90) + W <= S_TTFT,  W = lam E[S^2] / (2 (1 - lam E[S]))  (M/G/1 mean wait)
  lam_cap(D) = largest lam with every lam' <= lam feasible.  D* = argmax_D lam_cap(D).

Usage:
  python3 dstar_model.py summarize --out <summary.json>   # workload lengths from local raw traces (needs raw/)
  python3 dstar_model.py run --summary <summary.json> --out <result.json>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analytic_terms as at  # noqa: E402

VERSION = "dstar-model proposal r0 (2026-10-30)"
TOTAL_SM = 108
HBM_BW = 2.039e12            # A100 SXM 80GB [derived input, same value as ANALYTIC_COST_FORMULAS.md section 6]
D_GRID = (16, 24, 34, 44, 64)

# ---- [measured] T1 decode step, ms (results/cap_char/costterms_analysis/supplement_costterms_2026-10-27.json,
#      S2_round_diffs/T1_chosen; J-A1 r1 / J-A2 r2; adopted form per mode; RESULT_costterms section 2) -------------
#   F1: a + b*bs + c*bs*ctx ; F2: F1 + d*bs^2 ; F3: a + b*bs^beta + c*bs*ctx
T1 = {
    "108u": ("F1", [13.08333519, 0.2724744, 9.513807e-06], [13.03390978, 0.27494173, 9.592544e-06], None),
    16: ("F2", [36.81974154, 0.82199215, 2.025e-05, 0.00669418], [36.76381074, 0.82511334, 2.04e-05, 0.00670425], None),
    24: ("F3", [25.26419708, 0.28647175, 1.486e-05], [25.38148938, 0.30183705, 1.492e-05], (1.27, 1.26)),
    34: ("F2", [20.82011176, 0.43112856, 1.157e-05, 0.00282905], [20.87240742, 0.43606538, 1.16e-05, 0.00280929], None),
    44: ("F2", [17.08673005, 0.38845655, 1.132e-05, 0.00210504], [17.12273838, 0.39864394, 1.151e-05, 0.00205105], None),
    64: ("F2", [16.02091841, 0.24180685, 1.149e-05, 0.00146535], [16.04869696, 0.24545118, 1.135e-05, 0.00147861], None),
}
# ---- [measured] T2 single-request prefill TTFT at P=108, ms: u + v L + w L^2 (RESULT_costterms section 3, J-A1/J-A2 108v)
T2_108 = ([11.83, 0.1076, 1.880e-7], [14.08, 0.1059, 1.941e-7])
# ---- [measured] T2' at L=32768 with one decode co-running, ms, r1/r2 (RESULT_costterms section 4) -> kappa(D)
T2P_32K = {16: (4484, 4420), 24: (4933, 4823), 34: (5531, 5487), 44: (6363, 6320), 64: (9284, 9275)}
# ---- [measured] rho: co-location total effect (T3, CM-4; co-location + 400 W clock drop + observer cost, unseparated)
RHO = (1.13, 1.16, 1.19)

# ---- serving setup of the policy campaign (PREREG_HD_HE0cell rev1.7) ------------------------------------------------
CAP = 48                     # max_running_requests (TUNE log: bs 47/48 vs 0.85 x 48)
SLO = {"W1": {"ttft_ms": 3000.0, "itl_ms": 60.0}, "W3": {"ttft_ms": 2814.84, "itl_ms": 60.0}}
EVAL_RATES = {"LO": 2.7356, "HI": 10.9424}


def _avg(r1, r2):
    return [(x + y) / 2 for x, y in zip(r1, r2)]


def t_dec(mode, b, c):
    """Decode step (ms) at batch b and mean context c for a T1 mode."""
    form, r1, r2, beta = T1[mode]
    p = _avg(r1, r2)
    if form == "F1":
        return p[0] + p[1] * b + p[2] * b * c
    if form == "F2":
        return p[0] + p[1] * b + p[2] * b * c + p[3] * b * b
    bet = sum(beta) / 2
    return p[0] + p[1] * b ** bet + p[2] * b * c


def t2_108(L):
    u, v, w = _avg(*T2_108)
    return u + v * L + w * L * L


def kappa(D):
    """T2'(32k; P) * P / (T2(32k; 108) * 108): how far P-scaled T2 is from the co-running measurement."""
    return (sum(T2P_32K[D]) / 2) * (TOTAL_SM - D) / (t2_108(32768) * TOTAL_SM)


def s_pre(L, D):
    return kappa(D) * t2_108(L) * TOTAL_SM / (TOTAL_SM - D)


# ---- hardware layer: effective bandwidth fraction u(D) = a0 / a(D) ------------------------------------------------
def hardware_layer():
    arch = at.nano9b_v2()
    co = at.decode_coefficients(arch, HBM_BW)
    a0_ms = co["a0_s"] * 1e3
    rows = []
    for mode in (16, 24, 34, 44, 64, "108u"):
        a_ms = _avg(T1[mode][1], T1[mode][2])[0]
        D = TOTAL_SM if mode == "108u" else mode
        rows.append({"D": D, "a_ms": a_ms, "u": a0_ms / a_ms, "u_per_sm": a0_ms / a_ms / D})
    # Two-segment roofline through the origin: u = beta*D (D <= D_k), u = beta*D_k + gamma*(D - D_k) (D > D_k).
    # Descriptive only: a(D) also holds per-step fixed costs that are not bandwidth (launch, sampling) -- assumption A2.
    best = None
    for k in range(160, 1081):
        dk = k / 10
        xs1 = [min(r["D"], dk) for r in rows]
        xs2 = [max(r["D"] - dk, 0.0) for r in rows]
        ys = [r["u"] for r in rows]
        s11 = sum(x * x for x in xs1); s22 = sum(x * x for x in xs2); s12 = sum(x * y for x, y in zip(xs1, xs2))
        t1 = sum(x * y for x, y in zip(xs1, ys)); t2 = sum(x * y for x, y in zip(xs2, ys))
        det = s11 * s22 - s12 * s12
        if det <= 1e-12:
            continue
        beta = (t1 * s22 - t2 * s12) / det
        gamma = (s11 * t2 - s12 * t1) / det
        sse = sum((beta * a + gamma * b - y) ** 2 for a, b, y in zip(xs1, xs2, ys))
        if best is None or sse < best[0]:
            best = (sse, dk, beta, gamma)
    sse, dk, beta, gamma = best
    return {"a0_ms_derived": a0_ms, "hbm_bw": HBM_BW, "rows": rows,
            "fit": {"form": "two-segment through origin", "D_knee": dk, "beta_per_sm": beta, "gamma_per_sm": gamma,
                    "slope_ratio": gamma / beta, "sse": sse,
                    "per_sm_bw_GBps_below_knee": beta * HBM_BW / 1e9},
            "kappa": {D: kappa(D) for D in D_GRID}}


# ---- serving layer ---------------------------------------------------------------------------------------------------
def moments(Ls, D):
    s = [s_pre(L, D) / 1e3 for L in Ls]                    # seconds
    s.sort()
    return {"ES": statistics.fmean(s), "ES2": statistics.fmean(x * x for x in s), "S_q90": s[int(0.9 * (len(s) - 1))]}


def state(lam, D, wl, rho, mom):
    ER, c = wl["ER"], wl["ctx_decode"]
    f = min(1.0, lam * mom["ES"])
    T_eff = t_dec("108u", 1, c)
    b = 0.0
    for _ in range(400):
        b_new = lam * ER * T_eff / 1e3
        b = 0.5 * b + 0.5 * b_new
        T_eff = f * rho * t_dec(D, b, c) + (1 - f) * t_dec("108u", b, c)
    util = lam * mom["ES"]
    W = lam * mom["ES2"] / (2 * (1 - util)) if util < 1 else math.inf
    b95 = min(CAP, b + 1.645 * math.sqrt(max(b, 0.0)))
    step_p95 = rho * t_dec(D, b95, c) if f > 0.05 else t_dec("108u", b95, c)
    slots = lam * (ER * T_eff / 1e3 + mom["ES"] + (W if math.isfinite(W) else 0)) if math.isfinite(W) else math.inf
    ttft = (mom["S_q90"] + W) * 1e3
    ok = {"ITL": step_p95 <= wl["slo"]["itl_ms"], "SLOT": slots <= CAP, "TTFT": ttft <= wl["slo"]["ttft_ms"]}
    return {"lam": lam, "f": f, "b": b, "b95": b95, "T_eff_ms": T_eff, "step_p95_ms": step_p95, "slots": slots,
            "W_s": W, "ttft_q90_ms": ttft, "ok": ok}


def lam_cap(D, wl, rho, mom, step=0.005, lam_max=20.0):
    last = None
    lam = step
    while lam <= lam_max:
        st = state(lam, D, wl, rho, mom)
        if not all(st["ok"].values()):
            return {"lam_cap": last["lam"] if last else 0.0, "binding": [k for k, v in st["ok"].items() if not v],
                    "at_cap": last, "first_fail": st}
        last = st
        lam = round(lam + step, 6)
    return {"lam_cap": lam_max, "binding": ["none<=lam_max"], "at_cap": last, "first_fail": None}


def workload(summary, name):
    w = summary[name]
    return {"ER": w["out_mean"], "ctx_decode": w["in_mean"] + w["out_mean"] / 2, "slo": SLO[name], "Ls": w["in_sample"]}


def run(summary):
    out = {"version": VERSION, "analytic": at.VERSION, "hardware": hardware_layer(), "cap": CAP, "slo": SLO,
           "rho": RHO, "workloads": {}}
    for name in ("W1", "W3"):
        wl = workload(summary, name)
        per = {}
        for D in D_GRID:
            mom = moments(wl["Ls"], D)
            per[D] = {"moments": mom, "rho": {}}
            for rho in RHO:
                per[D]["rho"][str(rho)] = lam_cap(D, wl, rho, mom)
            # EVAL-phase diagnostics (W1 only): state at the LO rate, step at full cap for the HI burst
            if name == "W1":
                st_lo = state(EVAL_RATES["LO"], D, wl, RHO[1], mom)
                per[D]["eval"] = {
                    "LO_state": st_lo,
                    "HI_step_at_cap_ms": {str(r): r * t_dec(D, CAP, wl["ctx_decode"]) for r in RHO},
                    "HI_solo_step_at_cap_ms": t_dec("108u", CAP, wl["ctx_decode"]),
                }
        caps = {D: per[D]["rho"][str(RHO[1])]["lam_cap"] for D in D_GRID}
        out["workloads"][name] = {"ER": wl["ER"], "ctx_decode": wl["ctx_decode"], "n_sample": len(wl["Ls"]),
                                  "per_D": per, "lam_cap_rho_mid": caps,
                                  "D_star_rho_mid": max(D_GRID, key=lambda d: (caps[d], -d))}
    out["eval_prediction"] = {str(rho): eval_prediction(summary, rho) for rho in RHO}
    out["windows"] = {name: d_window(summary, name) for name in ("W1", "W3")}
    return out


def d_window(summary, name):
    """Closed-form feasibility window of D, before any queueing (the two walls of the formula).

    lower wall (grid): smallest D with rho * T_dec(D, cap, c) <= S_ITL  -- decode keeps the ITL SLO at a full slot cap
    upper wall (continuous): S_pre(L_q90; P) <= S_TTFT with zero wait  ->  D <= 108 - kappa * T2(L_q90) * 108 / S_TTFT
    """
    wl = workload(summary, name)
    Ls = sorted(wl["Ls"])
    Lq90 = Ls[int(0.9 * (len(Ls) - 1))]
    kap = statistics.fmean(kappa(D) for D in D_GRID)
    upper = TOTAL_SM - kap * t2_108(Lq90) * TOTAL_SM / wl["slo"]["ttft_ms"]
    lower = {}
    for rho in RHO:
        ok = [D for D in D_GRID if rho * t_dec(D, CAP, wl["ctx_decode"]) <= wl["slo"]["itl_ms"]]
        lower[str(rho)] = min(ok) if ok else None
    return {"L_q90": Lq90, "kappa_mean": kap, "upper_wall_D": upper,
            "lower_wall_D_at_cap": lower,
            "co_step_at_cap_ms": {D: RHO[1] * t_dec(D, CAP, wl["ctx_decode"]) for D in D_GRID}}


# ---- EVAL (HE0 cell, W1) phase model: LO = steady state, HI = burst into the slot cap ----------------------------
def _poisson_cdf(k, mu):
    if k < 0:
        return 0.0
    term = math.exp(-mu)
    acc = term
    for i in range(1, int(k) + 1):
        term *= mu / i
        acc += term
    return min(1.0, acc)


def eval_prediction(summary, rho, n_seg=200):
    """Per-D phase goodput for the HE0 cell (assumptions A7-A9 in the document).

    LO (2.7356 req/s): steady-state `state()`.  If the ITL rule holds, every request is good (TTFT is far below SLO
    in this regime).  If it fails, a request is good only when its co-located step count stays <= 5% of its tokens:
    co-located steps ~ Poisson(f * R_i), so P(good_i) = P(X <= 0.05 R_i).
    HI (10.9424 req/s, 200 requests, above every lam_cap): the slot cap is full; completions run at
    mu = cap / (E[R] T_eff + E[S]) with f = mu E[S]; a request is good if it is admitted within S_TTFT
    (the first `cap` plus mu * S_TTFT more) AND the co-located step at b = cap meets the ITL SLO (else 0 good).
    Segment duration: LO = n / lam_LO; HI = max(n / lam_HI, n / mu).  Total = sum good / sum duration.
    """
    wl = workload(summary, "W1")
    outs = summary["W1"]["out_sample"]
    res = {}
    for D in D_GRID:
        mom = moments(wl["Ls"], D)
        lo = state(EVAL_RATES["LO"], D, wl, rho, mom)
        if all(lo["ok"].values()):
            good_lo_frac = 1.0
        elif not lo["ok"]["ITL"]:
            good_lo_frac = statistics.fmean(_poisson_cdf(math.floor(0.05 * r), lo["f"] * r) for r in outs)
        else:
            good_lo_frac = 0.0
        c = wl["ctx_decode"]
        mu = cap_f = 0.0
        T_eff = t_dec("108u", CAP, c)
        for _ in range(200):
            mu = CAP / (wl["ER"] * T_eff / 1e3 + mom["ES"])
            cap_f = min(1.0, mu * mom["ES"])
            T_eff = cap_f * rho * t_dec(D, CAP, c) + (1 - cap_f) * t_dec("108u", CAP, c)
        itl_ok_hi = rho * t_dec(D, CAP, c) <= SLO["W1"]["itl_ms"]
        good_hi = min(n_seg, CAP + mu * SLO["W1"]["ttft_ms"] / 1e3) if itl_ok_hi else 0.0
        dur_lo = n_seg / EVAL_RATES["LO"]
        dur_hi = max(n_seg / EVAL_RATES["HI"], n_seg / mu)
        good_lo = good_lo_frac * n_seg
        res[D] = {"LO_good_frac": good_lo_frac, "LO_goodput": good_lo / dur_lo, "HI_mu": mu, "HI_f": cap_f,
                  "HI_itl_ok": itl_ok_hi, "HI_good": good_hi, "HI_goodput": good_hi / dur_hi,
                  "total_goodput": (good_lo + good_hi) / (dur_lo + dur_hi)}
    ref = res[44]["total_goodput"]
    for D in D_GRID:
        res[D]["total_ratio_to_D44"] = res[D]["total_goodput"] / ref if ref else math.nan
        res[D]["HI_ratio_to_D44"] = res[D]["HI_goodput"] / res[44]["HI_goodput"] if res[44]["HI_goodput"] else math.nan
    return res


# ---- workload summary from local raw traces (inputs are gitignored; the summary is what gets tracked) ------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(root, cap2_job, sample_n=4000, seed=7):
    import random
    rng = random.Random(seed)
    w1_path = os.path.join(root, cap2_job, "raw", "w1_lengths.json")
    pool = json.load(open(w1_path))
    ins = [a for a, _ in pool]
    outs = [b for _, b in pool]
    idx = sorted(rng.sample(range(len(ins)), sample_n))
    w1_sample = [ins[i] for i in idx]
    w1_out_sample = [outs[i] for i in idx]
    w3_in, w3_out, files = [], [], []
    for f in sorted(glob.glob(os.path.join(root, cap2_job, "raw", "segments", "*", "*.jsonl"))):
        files.append(os.path.relpath(f, root))
        for line in open(f):
            r = json.loads(line)
            if r.get("error"):
                continue
            w3_in.append(r["input_tokens"])
            w3_out.append(r.get("observed_output_tokens", 0))
    return {
        "W1": {"source": os.path.relpath(w1_path, root), "source_sha256": sha256(w1_path), "pool_n": len(pool),
               "in_mean": statistics.fmean(ins), "out_mean": statistics.fmean(outs),
               "note": "ShareGPT pool (ctx 4000), ignore_eos fixed output; in_sample = seeded subsample of inputs",
               "in_sample": w1_sample, "out_sample": w1_out_sample},
        "W3": {"source": f"{cap2_job}/raw/segments/*/*.jsonl ({len(files)} files)", "n": len(w3_in),
               "in_mean": statistics.fmean(w3_in), "out_mean": statistics.fmean(w3_out),
               "out_frac_at_1024": sum(o >= 1024 for o in w3_out) / len(w3_out),
               "note": "arXiv-Sum <=16k, natural output capped at 1024; output lengths read from CAP2 outcomes "
                       "(workload property, arm-independent by design; disclosed)",
               "in_sample": w3_in},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--root", default="workspace/engine-port/results/cap_char/policy_runs")
    s.add_argument("--cap2-job", default="pdmux-policy-cap2-e90e233-20261029023336")
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--summary", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "summarize":
        json.dump(summarize(a.root, a.cap2_job), open(a.out, "w"), indent=1)
        return 0
    res = run(json.load(open(a.summary)))
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    hw = res["hardware"]
    print(VERSION)
    print(f"a0 (derived) = {hw['a0_ms_derived']:.3f} ms; u(D) knee D_k={hw['fit']['D_knee']:.1f} beta={hw['fit']['beta_per_sm']:.4f}/SM gamma/beta={hw['fit']['slope_ratio']:.2f} "
          f"({hw['fit']['per_sm_bw_GBps_below_knee']:.1f} GB/s per SM below knee)")
    for row in hw["rows"]:
        print(f"  D={row['D']:>3}  a={row['a_ms']:.2f} ms  u={row['u']:.3f}  u/SM={row['u_per_sm']:.4f}")
    print("  kappa:", {d: round(k, 3) for d, k in hw["kappa"].items()})
    for name, w in res["workloads"].items():
        print(f"{name}: E[R]={w['ER']:.0f} ctx={w['ctx_decode']:.0f} D*={w['D_star_rho_mid']}")
        for D in D_GRID:
            pr = w["per_D"][D]["rho"]
            caps = " / ".join(f"{pr[str(rh)]['lam_cap']:.3f}" for rh in RHO)
            print(f"  D{D:<3} lam_cap(rho {RHO[0]}/{RHO[1]}/{RHO[2]}) = {caps}  binding={pr[str(RHO[1])]['binding']}"
                  f"  E[S]={w['per_D'][D]['moments']['ES']*1e3:.0f} ms  S_q90={w['per_D'][D]['moments']['S_q90']*1e3:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
