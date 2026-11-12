#!/usr/bin/env python3
"""Seed-to-seed variance of static-D goodput / attainment (Nano-9B-v2), for the motivation preregistration (CPU only, 2026-11-11).

Inputs (read only; sealed harness files are imported, never modified; no bytecode is written next to them):
  R  = policy_implfix Job ...131429 (static D16/24/34/44/64, one boot per D, seeds 1-4, W1 EVAL cell 3 rounds x LO/HI x 200 req)
  H  = policy_implfix Job ...131425 (s44 and s44-rep: two boots of D44, seeds 1-4, same cell and traces as R) -> boot-to-boot placebo
  CAP1R (W1, 120 req/run, 10 rungs x 2 seeds) and CAP2 (W3 long input, 48 req/run, 6 rungs x 2 seeds): reference only (n=2)
goodput = sum(good)/sum(segment durations); good = no error AND TTFT <= SLO AND request-internal token-ITL p95 <= SLO (ph_verdict).
Scope: the ratios between D are NOT citable (the HI segment is above capacity); this script is about dispersion only.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import math
import statistics
from pathlib import Path

from scipy import stats as SS

ROOT = Path("/home/wonho/Experiments/KISTI")
MAIN = ROOT / "prefill-layer-alloc/workspace/engine-port"
IMPL = ROOT / "prefill-layer-alloc-implfix/workspace/engine-port/results/policy_implfix"
R_RAW = IMPL / "runs/pdmux-policy-implfix-cb071cf-20261031131429/raw"
H_RAW = IMPL / "runs/pdmux-policy-implfix-cb071cf-20261031131425/raw"
CAP1R = MAIN / "results/cap_char/policy_runs/pdmux-policy-cap1r-4bef3cf-20261029003539/raw"
CAP2 = MAIN / "results/cap_char/policy_runs/pdmux-policy-cap2-e90e233-20261029023336/raw"
OUT = Path(__file__).resolve().parent

sys.path.insert(0, str(IMPL))
sys.path.insert(0, str(MAIN / "benchmarks"))
sys.path.insert(0, str(MAIN / "results/motivation"))
sys.path.insert(0, str(MAIN / "results/motivation/workloads"))
import pi_analyze as PA  # noqa: E402  (sealed; import only)
import ph_verdict as V  # noqa: E402  (sealed; import only)
from pdmux_eval import analyze as AN  # noqa: E402  (canonical bootstrap)
import mo_spec as MS  # noqa: E402
import mo_analyze as MA  # noqa: E402

TT, ITL = 3000.0, 60.0
DS = (16, 24, 34, 44, 64)
PH = (None, "LO", "HI")
PHN = {None: "ALL", "LO": "LO", "HI": "HI"}
NS = (4, 6, 8)
LN3 = math.log(1.03)


def tq(p, df):
    return float(SS.t.ppf(p, df))


def load(raw, prefix):
    out = {}
    for p in sorted((raw / "runs").glob(f"{prefix}*.json")):
        r = json.loads(p.read_text())
        out.setdefault(r["arm"]["name"], {})[r["seed"]] = r
    return out


def reqs(run, phase=None):
    return [q for s in run["segments"] if phase is None or s["phase"] == phase for q in s["requests"]]


def att(rs, tt=TT, itl=ITL):
    return sum(1 for q in rs if V.req_good(q, tt, itl)) / len(rs)


def ttft_ok(rs, tt=TT):
    return sum(1 for q in rs if q.get("ttft_ms") is not None and q["ttft_ms"] <= tt) / len(rs)


def msd(xs):
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else math.nan
    return {"mean": m, "sd": sd, "cv": sd / m if m else math.nan, "values": xs}


def tci_ln(lns):
    n = len(lns)
    m = statistics.fmean(lns)
    sd = statistics.stdev(lns)
    h = tq(0.975, n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean_ln": m, "sd_ln": sd, "t_half_ln": h, "ratio": math.exp(m), "t_ci_ratio": [math.exp(m - h), math.exp(m + h)]}


def boot_ln(base_ln, prop_ln):
    b = AN.paired_bootstrap_ci({str(k): v for k, v in base_ln.items()}, {str(k): v for k, v in prop_ln.items()})
    return {"ci_ln": [b["ci95_low"], b["ci95_high"]], "half_ln": (b["ci95_high"] - b["ci95_low"]) / 2,
            "ci_ratio": [math.exp(b["ci95_low"]), math.exp(b["ci95_high"])]}


def sd_upper(sd, df, conf=0.95):
    """one-sided 95 % upper bound of sigma from a sample sd with df degrees of freedom (chi-square)."""
    return sd * math.sqrt(df / SS.chi2.ppf(1 - conf, df))


def thresholds(sd_ln):
    """observed mean ln difference needed to exclude 0 / to exclude +-3 %, and the 80 %-power minimum detectable difference."""
    out = {}
    for n in NS:
        se = sd_ln / math.sqrt(n)
        h = tq(0.975, n - 1) * se
        mde = (tq(0.975, n - 1) + tq(0.80, n - 1)) * se
        out[str(n)] = {"excl0_pct": 100 * (math.exp(h) - 1), "excl3_pct": 100 * (math.exp(LN3 + h) - 1),
                       "mde80_vs0_pct": 100 * (math.exp(mde) - 1), "mde80_vs3_pct": 100 * (math.exp(LN3 + mde) - 1)}
    return out


def main():
    res = {"scope": ("Dispersion only. Ratios between D in HI/ALL are from an above-capacity segment (TTFT<=SLO fraction < 0.5) and are "
                     "NOT citable as magnitudes; they appear only to check that the right runs/definition are used."),
           "inputs": {"R": str(R_RAW), "H": str(H_RAW), "CAP1R": str(CAP1R), "CAP2": str(CAP2)}}
    R = {k: v for k, v in load(R_RAW, "PI_R_").items()}
    H = load(H_RAW, "PI_H_s44")
    seeds = sorted(set.intersection(*[set(R[f"s{d}"]) for d in DS]))
    # ---- integrity: same trace per seed across D, one boot per arm, one GPU
    trace = {s: {R[f"s{d}"][s]["trace_hash"] for d in DS} for s in seeds}
    res["integrity"] = {"seeds": seeds, "trace_hash_unique_per_seed": all(len(v) == 1 for v in trace.values()),
                        "gpu": sorted({r["gpu_uuid"] for a in R.values() for r in a.values()}),
                        "boots": {a: sorted({r["boot_id"] for r in v.values()}) for a, v in R.items()},
                        "H_same_trace_as_R": all(H["s44"][s]["trace_hash"] == R["s44"][s]["trace_hash"] for s in seeds),
                        "H_gpu": sorted({r["gpu_uuid"] for a in H.values() for r in a.values()}),
                        "segments_per_run": sorted({len(r["segments"]) for a in R.values() for r in a.values()}),
                        "max_vs_sum_duration_check": {a: [sum(s["duration_s"] for s in r["segments"]) / max(s["duration_s"] for s in r["segments"])
                                                          for r in v.values()] for a, v in R.items() if a == "s44"}}
    # ---- 1. reproduction through the sealed analyzer
    st = PA.static_table({f"s{d}": R[f"s{d}"] for d in DS})
    g, hi, lo = st["G_all"], st["hi_good"], st["lo_frac"]
    hi_ln = [math.log(PA.goodput(R["s64"][s], "HI") / PA.goodput(R["s44"][s], "HI")) for s in seeds]
    m, l, h_, n = V.paired_t_ci(hi_ln)
    res["reproduction"] = {
        "G_all": g, "hi_good": hi, "lo_frac": lo, "score": PA.score_dstar(g, lo, hi),
        "G24/G44": g[24] / g[44], "G16/G44": g[16] / g[44], "G34/G44": g[34] / g[44], "G64/G44": g[64] / g[44],
        "HI G64/G44": hi[64] / hi[44], "HI D64/D44 paired t CI": [math.exp(l), math.exp(h_)], "LO D16 good fraction": lo[16],
        "reported": {"G24/G44": 0.733, "G16/G44": 0.395, "G34/G44": 0.985, "G64/G44": 0.995, "HI G64/G44": 0.983,
                     "HI CI": [0.937, 1.028], "LO D16": 0.562}}
    # ---- 2. per D, per seed goodput; TTFT<=SLO fraction (overload flag)
    gp = {d: {ph: {s: PA.goodput(R[f"s{d}"][s], ph) for s in seeds} for ph in PH} for d in DS}
    res["goodput"] = {f"D{d}": {PHN[ph]: msd([gp[d][ph][s] for s in seeds]) for ph in PH} for d in DS}
    ovl = {d: {PHN[ph]: ttft_ok([q for s in seeds for q in reqs(R[f"s{d}"][s], ph)]) for ph in PH} for d in DS}
    res["ttft_le_slo_fraction"] = ovl
    blocked = {PHN[ph]: any(ovl[d][PHN[ph]] < 0.5 for d in DS) for ph in PH}
    blocked["ALL"] = blocked["ALL"] or blocked["HI"]          # gate 4: a mix containing an overloaded segment is blocked too
    res["magnitude_blocked"] = blocked
    # ---- 3. ratios vs D44 (same seed)
    rat = {}
    for d in (16, 24, 34, 64):
        rat[f"D{d}"] = {}
        for ph in PH:
            pos = [s for s in seeds if gp[d][ph][s] > 0 and gp[44][ph][s] > 0]
            if len(pos) < 2:
                rat[f"D{d}"][PHN[ph]] = {"note": "goodput 0 in some seed; ln ratio undefined"}
                continue
            lns = {s: math.log(gp[d][ph][s] / gp[44][ph][s]) for s in pos}
            row = tci_ln(list(lns.values()))
            row.update({"seed_ratio": {s: math.exp(v) for s, v in lns.items()},
                        "boot": boot_ln({s: math.log(gp[44][ph][s]) for s in pos}, {s: math.log(gp[d][ph][s]) for s in pos}),
                        "sign": "+" if row["mean_ln"] > 0 else "-", "magnitude_blocked": blocked[PHN[ph]]})
            rat[f"D{d}"][PHN[ph]] = row
    res["ratio_vs_D44"] = rat
    # ---- 4. all D pairs: sd of the seed-level ln ratio, grouped; thresholds for n=4/6/8
    import itertools
    pairs = {}
    for a, b in itertools.combinations(DS, 2):
        for ph in PH:
            if all(gp[a][ph][s] > 0 and gp[b][ph][s] > 0 for s in seeds):
                pairs.setdefault(PHN[ph], {})[f"D{a}-D{b}"] = statistics.stdev([math.log(gp[a][ph][s] / gp[b][ph][s]) for s in seeds])
    groups = {"plateau (34,44,64)": [("D34-D44"), ("D44-D64"), ("D34-D64")],
              "off-plateau vs plateau (16/24 vs 34/44/64)": ["D16-D34", "D16-D44", "D16-D64", "D24-D34", "D24-D44", "D24-D64"],
              "D16-D24": ["D16-D24"]}
    grp = {}
    for ph in ("ALL", "LO", "HI"):
        grp[ph] = {}
        for gname, keys in groups.items():
            v = [pairs[ph][k] for k in keys if k in pairs.get(ph, {})]
            if not v:
                continue
            pooled = math.sqrt(statistics.fmean(x * x for x in v))
            grp[ph][gname] = {"pair_sd_ln": {k: pairs[ph][k] for k in keys if k in pairs[ph]}, "pooled_sd_ln": pooled,
                              "max_sd_ln": max(v), "upper95_sd_ln_df3": sd_upper(max(v), 3),
                              "thresholds_pooled": thresholds(pooled), "thresholds_max": thresholds(max(v)),
                              "thresholds_upper95": thresholds(sd_upper(max(v), 3))}
    res["pair_sd"] = pairs
    res["pair_sd_groups"] = grp
    # placebo: two boots of D44, same seeds and traces (H Job)
    plac = {}
    for ph in PH:
        lns = [math.log(PA.goodput(H["s44-rep"][s], ph) / PA.goodput(H["s44"][s], ph)) for s in seeds]
        plac[PHN[ph]] = tci_ln(lns)
        plac[PHN[ph]]["thresholds"] = thresholds(statistics.stdev(lns))
    res["placebo_boot_D44_H"] = plac
    res["user_proxy_P_e1ty_vs_e1t"] = {"sd_ln": 0.0304, "note": "P cell (PM5 2k single-request pool, D44, PC_ABOVE_BAND), not the static-D sweep"}
    # ---- 5. attainment (primary SLO 3000/60): per seed, per phase; per segment (200 req); paired D diffs; reading-rule false-sign check
    A = {d: {ph: {s: att(reqs(R[f"s{d}"][s], ph)) for s in seeds} for ph in PH} for d in DS}
    res["attainment_seed"] = {f"D{d}": {PHN[ph]: msd([A[d][ph][s] for s in seeds]) for ph in PH} for d in DS}
    seg = {}
    for d in DS:
        for ph in ("LO", "HI"):
            vals = [att(sg["requests"]) for s in seeds for sg in R[f"s{d}"][s]["segments"] if sg["phase"] == ph]
            seg[f"D{d}|{ph}"] = {"n_segments": len(vals), "req_per_segment": 200, **msd(vals),
                                 "binomial_sd": math.sqrt(max(statistics.fmean(vals) * (1 - statistics.fmean(vals)), 0) / 200)}
    res["attainment_segment"] = seg
    ad = {}
    for d in (16, 24, 34, 64):
        for ph in PH:
            diffs = [A[d][ph][s] - A[44][ph][s] for s in seeds]
            sd = statistics.stdev(diffs)
            ad[f"D{d}-D44|{PHN[ph]}"] = {"mean": statistics.fmean(diffs), "sd": sd,
                                         "sd_of_seed_mean": {str(n): sd / math.sqrt(n) for n in NS},
                                         "p_false_sign_at_0.03": {str(n): (2 * (1 - SS.norm.cdf(MS.ATT_MARGIN / (sd / math.sqrt(n)))) if sd > 0 else 0.0) for n in NS}}
    res["attainment_paired_diff"] = ad
    # placebo attainment over the whole registered SLO grid, plus the registered reading rule
    pg = {}
    for ph in PH:
        diffs_grid = {}
        for p in MA.SLO_PAIRS:
            dd = [att(reqs(H["s44-rep"][s], ph), *p) - att(reqs(H["s44"][s], ph), *p) for s in seeds]
            a_rep = att([q for s in seeds for q in reqs(H["s44-rep"][s], ph)], *p)
            a_ref = att([q for s in seeds for q in reqs(H["s44"][s], ph)], *p)
            diffs_grid[f"{p[0]:g}/{p[1]:g}"] = {"pooled_diff": a_rep - a_ref, "seed_sd": statistics.stdev(dd), "rep": a_rep, "ref": a_ref}
        ga = {p: att([q for s in seeds for q in reqs(H["s44-rep"][s], ph)], *p) for p in MA.SLO_PAIRS}
        gb = {p: att([q for s in seeds for q in reqs(H["s44"][s], ph)], *p) for p in MA.SLO_PAIRS}
        lab = MA.read_label(ga, gb)
        inf = [v for k, v in diffs_grid.items() if not ((v["rep"] >= MS.SAT_HI and v["ref"] >= MS.SAT_HI) or (v["rep"] <= MS.SAT_LO and v["ref"] <= MS.SAT_LO))]
        pg[PHN[ph]] = {"grid": diffs_grid, "reading_rule_label": lab[0], "n_informative": lab[1],
                       "max_abs_pooled_diff_informative": max((abs(v["pooled_diff"]) for v in inf), default=None),
                       "max_seed_sd_informative": max((v["seed_sd"] for v in inf), default=None)}
    res["placebo_attainment_grid_H"] = pg
    # all D pairs over the SLO grid: seed sd of paired attainment differences at informative grid points (R)
    gsd = {}
    for a, b in itertools.combinations(DS, 2):
        for ph in PH:
            v = []
            for p in MA.SLO_PAIRS:
                xa = [att(reqs(R[f"s{a}"][s], ph), *p) for s in seeds]
                xb = [att(reqs(R[f"s{b}"][s], ph), *p) for s in seeds]
                ma, mb = statistics.fmean(xa), statistics.fmean(xb)
                if (ma >= MS.SAT_HI and mb >= MS.SAT_HI) or (ma <= MS.SAT_LO and mb <= MS.SAT_LO):
                    continue
                v.append(statistics.stdev([x - y for x, y in zip(xa, xb)]))
            if v:
                gsd[f"D{a}-D{b}|{PHN[ph]}"] = {"n_informative": len(v), "median_sd": statistics.median(v), "max_sd": max(v)}
    res["attainment_grid_pair_sd_R"] = gsd
    # ---- CAP1R (W1, 120 req) and CAP2 (W3, 48 req): 2-seed attainment sd per rung (reference only)
    def cap_rungs(raw, prefix, tt):
        runs = {}
        for p in sorted((raw / "runs").glob(f"{prefix}_D*.json")):
            r = json.loads(p.read_text())
            runs.setdefault((r["arm"]["D"], r["spec"]["rate"]), {})[r["seed"]] = r
        rows, sq, sqd = [], [], []
        for (d, rt), v in sorted(runs.items()):
            if len(v) != 2:
                continue
            a = [att(reqs(v[s]), tt, ITL) for s in sorted(v)]
            m_ = statistics.fmean(a)
            info = 0.02 < m_ < 0.98
            rows.append({"D": d, "rate": rt, "att": a, "n_req": len(reqs(v[sorted(v)[0]])), "informative": info})
            if info:
                sq.append((a[0] - a[1]) ** 2 / 2)
        # paired D-vs-D44 difference sd at the same rung
        for (d, rt), v in sorted(runs.items()):
            if d == 44 or (44, rt) not in runs:
                continue
            w = runs[(44, rt)]
            if len(v) != 2 or len(w) != 2:
                continue
            dd = [att(reqs(v[s]), tt, ITL) - att(reqs(w[s]), tt, ITL) for s in sorted(v)]
            ms_ = [statistics.fmean([att(reqs(x[s]), tt, ITL) for s in sorted(x)]) for x in (v, w)]
            if all(m >= 0.98 for m in ms_) or all(m <= 0.02 for m in ms_):
                continue
            sqd.append((dd[0] - dd[1]) ** 2 / 2)
        return {"rows": rows, "n_informative_cells": len(sq), "pooled_seed_sd": math.sqrt(statistics.fmean(sq)) if sq else None,
                "n_informative_pairs_vs_D44": len(sqd), "pooled_paired_diff_sd_vs_D44": math.sqrt(statistics.fmean(sqd)) if sqd else None}
    w3_tt = 3.0 * json.loads((CAP2 / "floor.json").read_text())["floor_ms"]
    res["cap1r_W1"] = cap_rungs(CAP1R, "CAP1R", TT)
    res["cap2_W3"] = {**cap_rungs(CAP2, "CAP2", w3_tt), "ttft_slo_ms": w3_tt}
    for key in ("cap1r_W1", "cap2_W3"):
        sd = res[key]["pooled_paired_diff_sd_vs_D44"]
        dfp = res[key]["n_informative_pairs_vs_D44"]          # one df per (D, rung) pair; cells share a boot -> optimistic df
        res[key]["reading_rule_check"] = {
            str(n): {"sd_of_seed_mean_diff": sd / math.sqrt(n),
                     "p_abs_diff_gt_margin_if_true0": 2 * (1 - SS.norm.cdf(MS.ATT_MARGIN / (sd / math.sqrt(n)))),
                     "margin_over_2se": MS.ATT_MARGIN / (2 * sd / math.sqrt(n)),
                     "true_diff_for_80pct_sign": MS.ATT_MARGIN + 0.8416 * sd / math.sqrt(n)} for n in NS}
        res[key]["upper95_paired_sd"] = sd_upper(sd, dfp)

    # ---- 6. request class counts (mo_spec.CLASS_EDGES)
    cls = {}
    for d in (44,):
        for s in seeds:
            for ph in PH:
                c = {}
                for q in reqs(R[f"s{d}"][s], ph):
                    k = MS.request_class(q["input_tokens"])
                    c[k] = c.get(k, 0) + 1
                cls[f"R s{d} seed{s} {PHN[ph]}"] = c
    c1 = {}
    for p in sorted((CAP1R / "runs").glob("CAP1R_D44_*.json")):
        r = json.loads(p.read_text())
        c = {}
        for q in reqs(r):
            k = MS.request_class(q["input_tokens"])
            c[k] = c.get(k, 0) + 1
        c1[r["run_id"]] = c
    res["class_counts_R"] = cls
    res["class_counts_CAP1R_D44"] = c1
    try:
        import gen_traces as GT
        import mw_lib as L
        pools = {w: L.load_pool(w) for w in ("sharegpt_first", "loogle_longdep")}
        ic = {}
        for s in L.SEEDS:
            for rate in GT.PROVISIONAL_SHORT_RATES:
                segs, _ = GT.interference(s, pools["sharegpt_first"], pools["loogle_longdep"], rate, GT.INTERFERENCE_DURATION, 1)
                c = {}
                for q in segs[0]["requests"]:
                    k = MS.request_class(q["input_tokens"]) + ("|long" if q.get("request_class") == "long" else "")
                    c[k] = c.get(k, 0) + 1
                ic[f"seed{s} short_rate{rate}"] = c
        res["class_counts_interference_provisional"] = ic
    except Exception as e:  # noqa: BLE001
        res["class_counts_interference_provisional"] = {"error": repr(e)}
    (OUT / "vc_variance.json").write_text(json.dumps(res, indent=1, sort_keys=False, default=str))
    print(json.dumps(res["reproduction"], indent=1, default=str))


if __name__ == "__main__":
    main()
