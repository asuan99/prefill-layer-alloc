#!/usr/bin/env python3
"""Where does including the hybrid (per-request state) decode term change the predicted best decode SM count D?  PREDICTION ONLY, NOT CITABLE.

Refines the synthetic search in predict.py (15 synthetic cells) with the real length pools of workspace/engine-port/results/motivation/workloads
(sharegpt_first, openchat_rounds, arxiv_summ, loogle_longdep): whole workloads, plus cells cut by input-length band.

Model: scripts/costmodel/dstar_model.py (proposal, decision 61, not canon) with analytic_terms.py.  The term removed from T1 is the
derived decode state-traffic term  b0(D) = decode_coefficients(...)['b0_s_per_req'] / u(D)  [derived / u from T1 a(D)].
Two state-removal variants are computed so the earlier method can be checked:
  corrected   T_nostate(b) = T(b) - b * b0(D), floored at the mode's a(D); applied to every T1 mode, including the F3 form of D24
  predict.py  the exact predict.py construction (b coefficient minus b0(D), clamped at 0; F3-form D24 left unchanged, so D24 keeps its state term)

Usage: python3 -I hybrid_term_cells.py --out OUT.json       (CPU; about a minute)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workspace" / "engine-port" / "scripts" / "costmodel"))
sys.path.insert(0, str(ROOT / "workspace" / "engine-port" / "results" / "motivation" / "workloads"))
import dstar_model as dm  # noqa: E402
import analytic_terms as at  # noqa: E402
import mw_lib as L  # noqa: E402

BANDS = [(0, 256), (256, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16385)]
CAPS = (48, 128)
RHOS = dm.RHO
SEED = 20261110
SAMPLE = 4000
ARXIV_OBS_OUT = 1007.2510416666667   # CAP2 observed natural output mean (reports/cost_model/dstar/workload_summary.json W3 out_mean)


def b0_by_mode():
    co = at.decode_coefficients(at.nano9b_v2(), dm.HBM_BW)
    b0 = co["b0_s_per_req"] * 1e3
    hw = dm.hardware_layer()
    u = {r["D"]: r["u"] for r in hw["rows"]}
    return {m: b0 / u[108 if m == "108u" else m] for m in dm.T1}, b0, u


def make_nostate_predictpy(dd):
    ns = {}
    for m, (form, r1, r2, beta) in copy.deepcopy(dm.T1).items():
        if form == "F3":
            ns[m] = (form, r1, r2, beta)
            continue
        r1, r2 = list(r1), list(r2)
        r1[1], r2[1] = max(r1[1] - dd[m], 0), max(r2[1] - dd[m], 0)
        ns[m] = (form, r1, r2, beta)
    return ns


ORIG_T_DEC = dm.t_dec
ORIG_T1 = copy.deepcopy(dm.T1)


def caps_for(wl, rho, cap, variant, dd):
    """lam_cap per D under one variant: 'full', 'corrected', 'predictpy'."""
    dm.CAP = cap
    dm.T1 = ORIG_T1
    dm.t_dec = ORIG_T_DEC
    if variant == "predictpy":
        dm.T1 = make_nostate_predictpy(dd)
    elif variant == "corrected":
        def td(mode, b, c):
            a = dm._avg(ORIG_T1[mode][1], ORIG_T1[mode][2])[0]
            return max(a, ORIG_T_DEC(mode, b, c) - dd[mode] * b)
        dm.t_dec = td
    out = {}
    for D in dm.D_GRID:
        mom = dm.moments(wl["Ls"], D)
        r = dm.lam_cap(D, wl, rho, mom, step=0.01)
        out[D] = (r["lam_cap"], r["binding"])
    dm.T1 = ORIG_T1
    dm.t_dec = ORIG_T_DEC
    return out


def dstar(caps_map):
    best = max(dm.D_GRID, key=lambda d: (caps_map[d][0], -d))
    return best


def make_wl(name, ins, outs, er=None, ttft=3000.0, itl=60.0, seed_key=""):
    rng = random.Random(f"{SEED}|{name}|{seed_key}")
    ls = list(ins)
    if len(ls) > SAMPLE:
        ls = rng.sample(ls, SAMPLE)
    r = float(st.fmean(outs)) if er is None else er
    return {"ER": r, "ctx_decode": float(st.fmean(ins)) + r / 2, "slo": {"ttft_ms": ttft, "itl_ms": itl}, "Ls": ls}


def cells():
    pool = {w: L.load_pool(w) for w in L.WORKLOADS}
    c = []
    for w, ttft, er, lab in (("sharegpt_first", 3000.0, None, "whole"), ("openchat_rounds", 3000.0, None, "whole"),
                             ("arxiv_summ", 2814.84, None, "whole, output = abstract"),
                             ("arxiv_summ", 2814.84, ARXIV_OBS_OUT, "whole, output = CAP2 observed natural"),
                             ("loogle_longdep", 3000.0, None, "whole (long stream alone)")):
        rows = pool[w]
        c.append({"id": f"{w}|{lab}", "workload": w, "band": lab, "n": len(rows),
                  "wl": make_wl(w, [r[1] for r in rows], [r[2] for r in rows], er=er, ttft=ttft, seed_key=lab)})
    for w, ttft in (("sharegpt_first", 3000.0), ("openchat_rounds", 3000.0), ("arxiv_summ", 2814.84)):
        for lo, hi in BANDS:
            rows = [r for r in pool[w] if lo <= r[1] < hi]
            if len(rows) < 200:
                continue
            lab = f"input {lo}-{hi - 1}"
            c.append({"id": f"{w}|{lab}", "workload": w, "band": lab, "n": len(rows),
                      "wl": make_wl(w, [r[1] for r in rows], [r[2] for r in rows], ttft=ttft, seed_key=lab)})
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dd, b0, u = b0_by_mode()
    grid, rows = cells(), []
    for c in grid:
        for cap in CAPS:
            for rho in RHOS:
                f = caps_for(c["wl"], rho, cap, "full", dd)
                n = caps_for(c["wl"], rho, cap, "corrected", dd)
                p = caps_for(c["wl"], rho, cap, "predictpy", dd)
                df, dn, dp = dstar(f), dstar(n), dstar(p)
                tied_f = sum(1 for d in dm.D_GRID if f[d][0] == f[df][0]) > 1
                rows.append({"id": c["id"], "workload": c["workload"], "band": c["band"], "n": c["n"], "cap": cap, "rho": rho,
                             "ER": round(c["wl"]["ER"], 1), "ctx_decode": round(c["wl"]["ctx_decode"], 1),
                             "D_full": df, "lam_full": f[df][0], "binding_full": f[df][1],
                             "lam_full_by_D": {str(d): f[d][0] for d in dm.D_GRID},
                             "D_nostate_corrected": dn, "lam_nostate_corrected": n[dn][0],
                             "D_nostate_predictpy": dp, "lam_nostate_predictpy": p[dp][0],
                             "differs_corrected": df != dn, "differs_predictpy": df != dp,
                             "degenerate_all_zero": f[df][0] == 0.0 and n[dn][0] == 0.0, "tied_full": tied_f})
    summary = {"n_cells": len(rows), "n_differs_corrected": sum(r["differs_corrected"] for r in rows),
               "n_differs_predictpy": sum(r["differs_predictpy"] for r in rows),
               "predictpy_differ_cells_not_in_corrected": sum(r["differs_predictpy"] and not r["differs_corrected"] for r in rows),
               "degenerate_cells": sum(r["degenerate_all_zero"] for r in rows)}
    out = {"status": "prediction only, not citable; dstar_model proposal r0 (decision 61), analytic-v1 prior",
           "version": dm.VERSION, "analytic": at.VERSION, "b0_ms_per_req_derived": b0,
           "dd_ms_per_req_by_mode": {str(k): v for k, v in dd.items()}, "u_by_D": {str(k): v for k, v in u.items()},
           "summary": summary, "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))
    print(json.dumps({"dd": out["dd_ms_per_req_by_mode"], "b0": b0}, indent=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
