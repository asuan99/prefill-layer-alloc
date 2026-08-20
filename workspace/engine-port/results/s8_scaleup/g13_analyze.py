#!/usr/bin/env python3
"""Pre-registered analyzer for the gate #13 job-axis campaign (2026-08-20).

WRITTEN BEFORE ANY CAMPAIGN DATA EXISTS.  Its SHA-256 is recorded in
`PREREG_G13_2026-08-20.md`; a later change to this file invalidates that record.

It implements `DESIGN_G13_JOB_BATCH_REV3_2026-08-19.md` sec2.1 VERBATIM and adds
nothing:

    1. one-way random effects: sigma_job^2_hat = (MS_B - MS_W) / m
    2. bound: registered per arm BEFORE data (M8 Satterthwaite, Ha8 exact-F --
       the output of design_g13_stats._eval_plan at the REGISTERED plan, which
       depends only on (k, m, sigma_boot prior, prior band), all pre-data)
    3. guard, each bound cut at ITS OWN degeneracy point:
           Satterthwaite : MS_B <= MS_W          -> MEASUREMENT_FAILURE
           exact-F       : MS_B <= F_.05 * MS_W  -> MEASUREMENT_FAILURE
       MEASUREMENT_FAILURE is never counted as PASS.
    4. verdict: gap >= 3 * UB95  -> PASS, else `underpowered`
    5. MEASUREMENT_FAILURE -> UNDETERMINED and STOP.  No extra blocks. (sec2.4)

BLINDING.  Decision quantities are withheld unless `--unblind` is passed, and
`--unblind` is refused while `PREREG_G13_2026-08-20.md` still carries the
`AUDIT_STATUS: PENDING` line.  Data collection does not depend on the decision
rule, so the campaign may run while the rules audit is in flight -- but nobody
may look at r until that audit closes.  This flag is the mechanical enforcement
of that, in the spirit of methodology gate #34.

★WHAT THIS CANNOT SAY, whatever the numbers are (rev3 sec5-0 / sec5-1):
    - NOT "gate #13 is closed".  The b-matched contrast canon asks for is a
      batch (x) regime alias, and this campaign wires only the job axis: node
      and day are recorded, never decomposed.
    - the registerable sentence stops at "the magnitudes of Delta_batch and
      sigma_alloc were each measured".
    - no performance claim, no policy implication, HE0 untouched.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HERE, "PREREG_G13_2026-08-20.md")
BOOT_ID = re.compile(r"^(?P<arm>\w+)/(?P<cell>d\d+)/(?P<job>\d+)/blk(?P<blk>\d+)$")

# Registered, pre-data (design rev3 sec2.2.1 / sec3).
REGISTERED = {
    "M8":  {"k": 16, "m": 3, "window_s": 60, "bound": "satterthwaite_registered"},
    "Ha8": {"k": 16, "m": 6, "window_s": 60, "bound": "exact_F_pivot"},
}


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load("design_g13_stats", "design_g13_stats.py")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- ratios
def ratios_by_job(score_json):
    """-> {arm: {job: {blk: r}}} from the canonical scorer's own output."""
    d = json.load(open(score_json))
    legs = collections.defaultdict(dict)      # (arm, job, blk) -> {cell: median}
    for key, cell in d["cells"].items():
        arm, cellname, _sm, _b = key.split("/")
        for boot_id, med in cell["per_boot_median"].items():
            m = BOOT_ID.match(boot_id)
            if not m or m.group("arm") != arm:
                continue
            legs[(arm, m.group("job"), int(m.group("blk")))][cellname] = med
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for (arm, job, blk), cells in legs.items():
        if "d16" in cells and "d92" in cells:
            out[arm][job][blk] = cells["d16"] / cells["d92"]
    return {a: {j: dict(sorted(b.items())) for j, b in sorted(v.items())}
            for a, v in out.items()}


def anova(per_job):
    """One-way random effects on r, BALANCED design (m boot-pairs per job).

    Unbalanced input is refused rather than silently harmonic-meaned: the
    registered operating characteristic was simulated for the balanced case,
    so an unbalanced analysis would not be the registered rule.
    """
    jobs = sorted(per_job)
    ms = [len(per_job[j]) for j in jobs]
    if len(set(ms)) != 1:
        return {"balanced": False, "per_job_m": dict(zip(jobs, ms))}
    m, k = ms[0], len(jobs)
    if k < 2 or m < 2:
        return {"balanced": True, "k": k, "m": m, "insufficient": True}
    means = {j: st.fmean(list(per_job[j].values())) for j in jobs}
    grand = st.fmean(list(means.values()))
    # percent units throughout, matching the design inputs
    to_pct = lambda x: x / grand * 100.0                      # noqa: E731
    ss_b = m * sum((to_pct(means[j]) - to_pct(grand)) ** 2 for j in jobs)
    ss_w = sum((to_pct(v) - to_pct(means[j])) ** 2
               for j in jobs for v in per_job[j].values())
    return {"balanced": True, "k": k, "m": m, "grand_mean_r": grand,
            "MS_B": ss_b / (k - 1), "MS_W": ss_w / (k * (m - 1)),
            "df_B": k - 1, "df_W": k * (m - 1),
            "job_means_r": means}


def decide(arm, a):
    """rev3 sec2.1 steps 2-5, verbatim."""
    reg = REGISTERED[arm]
    gap = G.bslope()["total_gaps_to_explain"][arm]["pct_gap"]
    msb, msw, df_b, df_w, m = a["MS_B"], a["MS_W"], a["df_B"], a["df_W"], a["m"]
    fq = G._f_quantile(0.05, df_b, df_w)
    if reg["bound"] == "satterthwaite_registered":
        degenerate = msb <= msw
        de = max(1.0, 2 * G.REGISTERED_SAT_PRIOR ** 4 / ((2.0 / m ** 2) * (
            (m * G.REGISTERED_SAT_PRIOR ** 2 + msw) ** 2 / df_b + msw ** 2 / df_w)))
        ub = math.sqrt(max(0.0, (msb - msw) / m)) * math.sqrt(de / G._chi2_lower(0.05, de))
    else:
        degenerate = msb <= fq * msw
        ub = G._ub_exact_f(msb, msw, m, fq, G._chi2_lower(0.05, df_w), df_w)
    if degenerate:
        return {"bound": reg["bound"], "guard_threshold_ratio": fq,
                "MS_B_over_MS_W": msb / msw if msw else None,
                "verdict": "UNDETERMINED",
                "reason": "MEASUREMENT_FAILURE (own degeneracy point) -- sec2.4: "
                          "register UNDETERMINED and stop, add no blocks"}
    sigma_job = math.sqrt(max(0.0, (msb - msw) / m))
    return {"bound": reg["bound"], "guard_threshold_ratio": fq,
            "MS_B_over_MS_W": msb / msw,
            "sigma_job_hat_pct": sigma_job, "sigma_boot_hat_pct": math.sqrt(max(0.0, msw)),
            "UB95_sigma_job_pct": ub, "gap_pct": gap, "three_UB95": 3 * ub,
            "verdict": "PASS" if gap >= 3 * ub else "underpowered"}


# ---------------------------------------------------------------- controls
def controls(score_json_c2r):
    """Falsifiable checks that do NOT depend on the new campaign.

    PC-A  With the six committed C2-R boot-pairs of ONE job, MS_W is exactly the
          between-boot variance of r, so sqrt(MS_W) must reproduce
          design_g13_stats.pairsd()'s sigma_boot.  Fails if the ratio assembly
          or the percent normalisation here differs from the design inputs.
    PC-B  The bound helpers must reproduce external chi-square/F table values --
          delegated to design_g13_stats' own control set, whose result is
          reported here rather than re-derived (re-deriving would be a new,
          uncontrolled copy).
    Deliberately NOT counted: "MS_B >= 0" and "UB >= 0" (identities).
    """
    out = []
    try:
        rb = ratios_by_job(score_json_c2r)
        for arm in ("M8", "Ha8"):
            jobs = rb.get(arm, {})
            job = max(jobs, key=lambda j: len(jobs[j])) if jobs else None
            if not job or len(jobs[job]) < 2:
                out.append({"id": f"PC-A/{arm}", "ok": False,
                            "why": "no single job with >=2 boot-pairs in the C2-R json"})
                continue
            vals = list(jobs[job].values())
            mu = st.fmean(vals)
            sd_pct = st.stdev(vals) / mu * 100.0
            ref = G.pairsd()["arms"][arm]["sigma_boot_paired_pct"]
            out.append({"id": f"PC-A/{arm}",
                        "what": "sqrt(MS_W) on the committed C2-R job reproduces pairsd() sigma_boot",
                        "recomputed_pct": sd_pct, "design_input_pct": ref,
                        "abs_diff": abs(sd_pct - ref), "tolerance": 1e-6,
                        "ok": abs(sd_pct - ref) <= 1e-6})
    except Exception as exc:                                   # noqa: BLE001
        out.append({"id": "PC-A", "ok": False, "why": repr(exc)})
    try:
        c = G._rev2_positive_controls()
        out.append({"id": "PC-B", "what": "design_g13_stats control set (bound helpers)",
                    "all_pass": c.get("all_pass"), "n": len(c.get("checks", [])),
                    "ok": bool(c.get("all_pass"))})
    except Exception as exc:                                   # noqa: BLE001
        out.append({"id": "PC-B", "ok": False, "why": repr(exc)})
    return out


def _audit_pending():
    try:
        return "AUDIT_STATUS: PENDING" in open(PREREG).read()
    except OSError:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-json", required=True,
                    help="output of `s8_c2r_score.py c2r --output ...` over the campaign jobs")
    ap.add_argument("--c2r-json", default=os.path.join(HERE, "C2R_RESULTS_2026-08-16.json"),
                    help="committed C2-R scorer output, for PC-A")
    ap.add_argument("--unblind", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    res = {"analyzer_sha256": _sha256(os.path.abspath(__file__)),
           "registered_plan": REGISTERED,
           "registered_sigma_boot_band_point": G.REGISTERED_SIGMA_BOOT_BAND_POINT,
           "rule": "DESIGN_G13_JOB_BATCH_REV3_2026-08-19.md sec2.1, verbatim",
           "forbidden": ["gate #13 is closed", "performance claim", "policy implication",
                         "arm ranking", "node/day decomposition"],
           "positive_controls": controls(a.c2r_json)}
    res["CONTROLS"] = "PASS" if all(c.get("ok") for c in res["positive_controls"]) else "FAIL"

    rb = ratios_by_job(a.score_json)
    res["inventory"] = {arm: {"n_jobs": len(v), "boot_pairs_per_job":
                              {j: len(b) for j, b in v.items()}} for arm, v in rb.items()}
    if not a.unblind:
        res["BLINDED"] = ("decision quantities withheld; pass --unblind after the rules "
                          "audit closes and PREREG_G13_2026-08-20.md no longer says "
                          "AUDIT_STATUS: PENDING")
    elif _audit_pending():
        res["BLINDED"] = "REFUSED: --unblind requested but PREREG still says AUDIT_STATUS: PENDING"
    else:
        res["arms"] = {}
        for arm in ("M8", "Ha8"):
            an = anova(rb.get(arm, {}))
            res["arms"][arm] = {"anova": an}
            if an.get("balanced") and not an.get("insufficient"):
                res["arms"][arm]["decision"] = decide(arm, an)
                reg = REGISTERED[arm]
                if (an["k"], an["m"]) != (reg["k"], reg["m"]):
                    res["arms"][arm]["PLAN_DEVIATION"] = (
                        f"realised (k,m)=({an['k']},{an['m']}) != registered "
                        f"({reg['k']},{reg['m']}); the registered operating "
                        f"characteristic does not describe this run")
    txt = json.dumps(res, indent=2, default=str)
    if a.out:
        dest = os.path.join(HERE, a.out)
        if os.path.exists(dest):
            sys.stderr.write("REFUSING to overwrite: %s\n" % dest)
            raise SystemExit(2)
        open(dest, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
