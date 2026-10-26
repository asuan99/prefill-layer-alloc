#!/usr/bin/env python3
"""Cost-term campaign -- result-analyst supplement (2026-10-27).  REGISTERED RULES ONLY (decision 38): no new metric.

Post-fetch code (disclosed: PREREG §8-6 asks analysis code before fetch; this file only re-applies sealed rules the
harness / aj_analysis did not emit, using the harness's own ct_fit / ct_extract functions read-only):

  S-1  §1-5 CLOCK_SENSITIVE for J-A1 / J-A2 T1 (harness A-22 deferred it to "the analyser after time alignment").
       [해석] alignment: the T3 cells of the SAME Job carry a (wall, monotonic) clock_pair; CLOCK_MONOTONIC is system-wide,
       so wall - mono of that pair aligns every T1 window t_range (engine monotonic) to clk.csv (UTC).  Exposure > 20 %
       windows are dropped, the T1 fit (ct_fit.fit_t1) is redone, the label is set if any coefficient of the
       CHOSEN form (and, separately reported, F1) moves by > 3 %.  Jobs without a T3 boot have no clock pair -> NOT_COMPUTED.
  S-2  §1-1 (iv) / §1-2 two-round coefficient differences (J-A1 vs J-A2; J-C r1 vs r2) -- descriptive.
  S-3  J-C N=1 anchor T2 / T2' fit and q: the harness keyed T2 / T2' without tp, so the TP2 boot overwrote the
       anchor in tables.t2 / fits.t2 / t2prime (key collision).  Re-applies ct_fit.fit_t2 / q_index / q_label to the anchor.
  S-4  Job-level SW-power-cap (0x4) exposure from meta/clk.csv (descriptive substrate disclosure).
  S-5  window-status / label / INFEASIBLE_BY_RULE census per Job.
  S-6  measured / analytic-v1 prior ratios (priors only through analytic_terms.py, decision 27).
  S-7  T2' leave-one-internal-L-out errors (the harness strips them from analysis.json): ct_fit.fit_t2 on CONFIRMED points.

  python3 supplement_costterms_2026-10-27.py --out supplement_costterms_2026-10-27.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(os.path.abspath(__file__)).parent
CAP = HERE.parent                                    # results/cap_char
ENG = CAP.parents[1]                                 # workspace/engine-port
sys.path.insert(0, str(ENG / "scripts" / "costmodel"))
sys.path.insert(0, str(CAP / "costterms_harness"))
import analytic_terms as AT  # noqa: E402
import ct_extract as X  # noqa: E402
import ct_fit as F  # noqa: E402

RUNS = CAP / "costterms_runs"
JOBS = {
    "J-A1": "pdmux-costterms-ja1-af28dfe-20261026083233",
    "J-A2": "pdmux-costterms-ja2-7d4b142-20261026121455",
    "J-H1": "pdmux-costterms-jh1-6180cfd-20261026121507",
    "J-H2": "pdmux-costterms-jh2-6673ac8-20261026121517",
    "J-B2": "pdmux-costterms-jb2-aff07e3-20261026121528",
    "J-C": "pdmux-costterms-jc-f2fdf8c-20261026121538",
}
META = {k: ENG / "results" / f"costterms_{k.replace('-', '').lower()}" / "vessl" / v / "meta" for k, v in JOBS.items()}
EXPOSURE_MAX = 0.20          # §1-5 [제안 봉인]
COEF_CHANGE_MAX = 0.03       # §1-5
PEAK_FLOPS = 312e12          # AJ-4 sealed constant (prefill prior time)
HBM = 2.039e12


def load(job):
    return json.loads((RUNS / JOBS[job] / "analysis.json").read_text())


def median(xs):
    return X.median(xs)


def fit_points(ws):
    vals, ctx = {}, {}
    for w in ws:
        if w.get("status") == "VALID" and w.get("S_eng_ms") is not None:
            vals.setdefault((w["bs"], w["c"]), []).append(w["S_eng_ms"])
            ctx.setdefault((w["bs"], w["c"]), []).append(X_fit_ctx(w))
    return [(b, median(ctx[(b, c)]), median(v), c) for (b, c), v in sorted(vals.items())]


def X_fit_ctx(w):
    return w.get("ctx_mean")


def rel_change(a, b):
    return [abs(y / x - 1) if x else None for x, y in zip(a, b)]


# ---------------------------------------------------------------- S-1
def clock_offset(A):
    offs = []
    for t in (A["tables"].get("t3") or {}).values():
        for c in t["cells"].values():
            for w in c["reps"]:
                if w.get("clock_pair"):
                    offs.append(w["clock_pair"][0] - w["clock_pair"][1])
    return (median(offs), max(offs) - min(offs), len(offs)) if offs else (None, None, 0)


def clock_sensitive(job):
    A = load(job)
    off, spread, n = clock_offset(A)
    if off is None:
        return {"status": "NOT_COMPUTED (no clock pair in the Job summary; raw on cluster volume)"}
    clk = X.read_clk(META[job] / "clk.csv")
    out = {"offset_wall_minus_mono_s": off, "offset_spread_s": spread, "n_pairs": n, "modes": {}}
    for bid, b in A["boots"].items():
        if b["kind"] != "T1T2" or not b.get("t1_windows") or b["identity"]["cap"] != 128:
            continue
        ws = b["t1_windows"]
        expo = {}
        for w in ws:
            if w.get("status") != "VALID":
                continue
            cov = X.clock_covariates(clk, (off, 0.0), [tuple(w["t_range"])])
            expo[w["cell_id"]] = cov
        keep = [w for w in ws if w.get("status") == "VALID" and (expo.get(w["cell_id"], {}).get("sw_power_cap_share") or 0) <= EXPOSURE_MAX]
        base = F.fit_t1(fit_points(ws), b["identity"]["mode"])
        alt = F.fit_t1(fit_points(keep), b["identity"]["mode"]) if len(keep) >= 4 else None
        shares = [c.get("sw_power_cap_share") for c in expo.values() if c.get("status") == "OK"]
        row = {"boot": bid, "n_valid": len(expo), "n_exposed_gt20": len(expo) - len(keep),
               "share_median": median(shares) if shares else None, "share_max": max(shares) if shares else None,
               "sm_mhz_median": median([c["sm_mhz_median"] for c in expo.values() if c.get("status") == "OK"]) if shares else None}
        if len(keep) == len(expo):
            row.update({"label": None, "note": "no window above 20 % -> refit identical"})
        elif alt is None:
            row.update({"label": "NA (fewer than 4 cells left)"})
        else:
            ch_f1 = rel_change(base["F1"]["coef"], alt["F1"]["coef"])
            cb, ca = base["chosen"], alt["chosen"]
            ch_chosen = rel_change(cb["coef"], ca["coef"]) if cb and ca and cb["form"] == ca["form"] else None
            sens = max(ch_f1) > COEF_CHANGE_MAX or (ch_chosen is not None and max(ch_chosen) > COEF_CHANGE_MAX)
            row.update({"coef_change_F1": ch_f1, "chosen_form_base": cb and cb["form"], "chosen_form_alt": ca and ca["form"],
                        "coef_change_chosen": ch_chosen, "label": "CLOCK_SENSITIVE" if sens else None})
        out["modes"][b["identity"]["mode"]] = row
    return out


# ---------------------------------------------------------------- S-2
def round_diffs():
    a1, a2 = load("J-A1"), load("J-A2")
    out = {"T1_F1": {}, "T1_chosen": {}, "T2": {}, "T2prime": {}, "gamma_j": {}, "theta_prime": {}, "T3_ln_rho": {},
           "T3_ln_sigma": {}}
    for k1, f1 in a1["fits"]["t1"].items():
        mode = k1.split("|")[3]
        k2 = k1.replace("J-A1|r1", "J-A2|r2")
        f2 = a2["fits"]["t1"].get(k2)
        if not f2:
            continue
        out["T1_F1"][mode] = {"r1": f1["F1"]["coef"], "r2": f2["F1"]["coef"],
                              "rel_diff": [abs(y / x - 1) for x, y in zip(f1["F1"]["coef"], f2["F1"]["coef"])]}
        c1, c2 = f1["chosen"], f2["chosen"]
        if c1 and c2 and c1["form"] == c2["form"]:
            out["T1_chosen"][mode] = {"form": c1["form"], "r1": c1["coef"], "r2": c2["coef"], "beta": [c1.get("beta"), c2.get("beta")],
                                      "rel_diff": [abs(y / x - 1) for x, y in zip(c1["coef"], c2["coef"])]}
    for tab in ("t2", "t2prime"):
        for k1, f1 in a1["fits"][tab].items():
            f2 = a2["fits"][tab].get(k1.replace("J-A1|r1", "J-A2|r2"))
            if f1.get("coef") and f2 and f2.get("coef"):
                out["T2" if tab == "t2" else "T2prime"][k1.split("|")[3]] = {
                    "r1": f1["coef"], "r2": f2["coef"], "rel_diff": [abs(y / x - 1) for x, y in zip(f1["coef"], f2["coef"])],
                    "q": [f1["q"].get("q"), f2["q"].get("q")]}
    g1, g2 = a1["fits"]["gamma_j"]["J-A1|r1"], a2["fits"]["gamma_j"]["J-A2|r2"]
    out["gamma_j"] = {j: [g1[j], g2[j]] for j in g1}
    out["theta_prime_harness"] = [a1["fits"]["theta_prime"]["J-A1|r1|real_a4"]["theta_prime"],
                                  a2["fits"]["theta_prime"]["J-A2|r2|real_a4"]["theta_prime"]]
    for d in ("D24", "D44"):
        t1 = a1["tables"]["t3"][f"J-A1|r1|real_a4|{d}"]["cells"]
        t2 = a2["tables"]["t3"][f"J-A2|r2|real_a4|{d}"]["cells"]
        for ck in sorted(t1):
            out["T3_ln_rho"][f"{d}|{ck}"] = [t1[ck]["ln_rho_mean"], t2[ck]["ln_rho_mean"]]
            out["T3_ln_sigma"][f"{d}|{ck}"] = [t1[ck]["ln_sigma_prime_mean"], t2[ck]["ln_sigma_prime_mean"]]
    # J-C r1 vs r2 (same Job)
    jc = load("J-C")
    out["J-C_T1_F1"] = {}
    for k, f in jc["fits"]["t1"].items():
        if "|r1|" in k:
            g = jc["fits"]["t1"].get(k.replace("|r1|", "|r2|"))
            if g:
                out["J-C_T1_F1"][k.replace("J-C|r1|real_a4|", "")] = {
                    "r1": f["F1"]["coef"], "r2": g["F1"]["coef"],
                    "rel_diff": [abs(y / x - 1) for x, y in zip(f["F1"]["coef"], g["F1"]["coef"])]}
    return out


# ---------------------------------------------------------------- S-3
def jc_anchor_t2():
    A = load("J-C")
    out = {"note": "harness tables.t2 / t2prime key = job|round|tag|mode (no tp): the TP2 boot overwrote the N=1 anchor; "
                   "this re-applies ct_fit.fit_t2 / q_index / q_label / j1_slope_label to the anchor boots"}
    qs = []
    for bid in ("jc1_n1_108v", "jc2_n1_108v"):
        b = A["boots"][bid]
        pts = [(x["L"], x["ttft_ms"]) for x in b["t2"] if x["status"] == "VALID"]
        ft = F.fit_t2(pts) if len(pts) >= 3 else None
        qi = F.q_index(pts, ft)
        slope, slab = F.j1_slope_label(pts)
        qs.append(qi.get("q"))
        out[bid] = {"points": {str(L): t for L, t in sorted(pts)}, "coef": ft and ft["coef"],
                    "loo_internal": ft and ft["loo_internal"], "accepted": bool(ft and ft["accepted"]),
                    "label": None if (ft and ft["accepted"]) else "FORM_INADEQUATE_T2_P108", "q": qi,
                    "j1_slope": slope, "j1_label": slab}
    out["q_label_anchor"] = F.q_label(qs)
    for bid in ("jc1_n1_D44", "jc2_n1_D44"):
        b = A["boots"][bid]
        pts = [(x["L"], x["ttft_ms"]) for x in b["t2p"] if x["status"] == "VALID" and (x.get("coexist_engine") or {}).get("confirmed")]
        ft = F.fit_t2(pts) if len(pts) >= 3 else None
        out[bid] = {"points_confirmed": {str(L): t for L, t in sorted(pts)}, "coef": ft and ft["coef"],
                    "loo_internal": ft and ft["loo_internal"], "accepted": bool(ft and ft["accepted"]),
                    "label": None if (ft and ft["accepted"]) else "FORM_INADEQUATE_T2P_P64", "q": F.q_index(pts, ft)}
    return out


# ---------------------------------------------------------------- S-4
def job_exposure():
    out = {}
    for job in JOBS:
        clk = X.read_clk(META[job] / "clk.csv")
        per = {}
        for g in sorted({c[3] for c in clk}):
            xs = [c for c in clk if c[3] == g]
            act = [c for c in xs if not (c[2] & 0x1)]           # 0x1 = GPU idle event reason
            per[str(g)] = {"n": len(xs), "share_0x4_all": sum(1 for c in xs if c[2] & 0x4) / len(xs),
                           "n_nonidle": len(act),
                           "share_0x4_nonidle": (sum(1 for c in act if c[2] & 0x4) / len(act)) if act else None,
                           "sm_mhz_median_nonidle": median([c[1] for c in act]) if act else None}
        out[job] = per
    return out


# ---------------------------------------------------------------- S-5
def census():
    out = {}
    for job in JOBS:
        A = load(job)
        st, lab, inf, t3st = {}, {}, [], {}
        for bid, b in A["boots"].items():
            for w in b.get("t1_windows", []):
                st[w["status"]] = st.get(w["status"], 0) + 1
                for l in w.get("labels", []):
                    lab[l] = lab.get(l, 0) + 1
            for x in b.get("feasibility", []) or []:
                if x.get("label") == "INFEASIBLE_BY_RULE":
                    inf.append(f"{bid}:({x['bs']},{x['c']})")
            for x in b.get("feasibility_t3", []) or []:
                if x.get("label") == "INFEASIBLE_BY_RULE":
                    inf.append(f"{bid}:T3({x.get('bs')},{x.get('L')})")
            for w in b.get("t3_cells", []):
                t3st[w.get("status")] = t3st.get(w.get("status"), 0) + 1
            for x in b.get("t2", []) + b.get("t2p", []):
                st["T2/T2'_" + x["status"]] = st.get("T2/T2'_" + x["status"], 0) + 1
            if b.get("not_run_after_death"):
                inf.append(f"{bid}:NOT_RUN_AFTER_SERVER_DEATH {b['not_run_after_death']}")
            for l in b.get("labels", []):
                lab["boot:" + l] = lab.get("boot:" + l, 0) + 1
        out[job] = {"t1_and_t2_status": st, "window_labels": lab, "t3_status": t3st, "infeasible_or_not_run": inf,
                    "boots": len(A["boots"]), "usable": sum(1 for b in A["boots"].values() if b.get("usable"))}
    return out


# ---------------------------------------------------------------- S-6
def prior_ratios():
    a4 = AT.nano9b_v2(4, 27)
    dc = AT.decode_coefficients(a4, HBM)
    pf = AT.prefill_flops(a4, 65536)
    pri = {"a0_ms": dc["a0_s"] * 1e3, "b0_ms": dc["b0_s_per_req"] * 1e3, "c0_ms": dc["c0_s_per_req_token"] * 1e3,
           "v_ms_per_tok": pf["linear_per_token"] / PEAK_FLOPS * 1e3, "w_ms_per_tok2": pf["quad_coeff"] / PEAK_FLOPS * 1e3,
           "s_bytes": AT.state_bytes_per_request(a4), "k_bytes": AT.kv_bytes_per_token(a4), "version": AT.VERSION}
    out = {"prior_a4": pri, "T1_F1_over_prior": {}, "T2_over_prior": {}}
    for job in ("J-A1", "J-A2"):
        A = load(job)
        for k, f in A["fits"]["t1"].items():
            a, b, c = f["F1"]["coef"]
            out["T1_F1_over_prior"][f"{job}|{k.split('|')[3]}"] = [a / pri["a0_ms"], b / pri["b0_ms"], c / pri["c0_ms"]]
        for k, f in A["fits"]["t2"].items():
            u, v, w = f["coef"]
            out["T2_over_prior"][f"{job}|{k.split('|')[3]}"] = [v / pri["v_ms_per_tok"], w / pri["w_ms_per_tok2"]]
    return out


# ---------------------------------------------------------------- S-7
def t2p_loo():
    out = {}
    for job in ("J-A1", "J-A2", "J-C"):
        A = load(job)
        for bid, b in A["boots"].items():
            if b["kind"] != "T1T2" or not b.get("t2p"):
                continue
            pts = [(x["L"], x["ttft_ms"]) for x in b["t2p"]
                   if x["status"] == "VALID" and (x.get("coexist_engine") or {}).get("confirmed")]
            ft = F.fit_t2(pts) if len(pts) >= 3 else None
            out[f"{job}|{bid}"] = {"n_confirmed": len(pts), "loo_internal": ft and ft["loo_internal"],
                                   "loo_max": ft and ft["loo_max"], "accepted": bool(ft and ft["accepted"])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"version": "costterms-supplement-v1 (2026-10-27, result-analyst; registered rules only)",
           "S1_clock_sensitive": {j: clock_sensitive(j) for j in ("J-A1", "J-A2")},
           "S2_round_diffs": round_diffs(), "S3_jc_anchor_t2": jc_anchor_t2(), "S4_job_exposure": job_exposure(),
           "S5_census": census(), "S6_prior_ratios": prior_ratios(),
           "S7_t2prime_loo": t2p_loo()}
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
