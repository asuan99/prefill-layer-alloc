#!/usr/bin/env python3
"""
NSL-1 detectability precondition -- what boot-level noise can this design tolerate?

★ CPU ONLY. NO GPU. NO MEASUREMENT. NO IMPORTED CONSTANT.

Why this file exists
--------------------
`DESIGN_NSL1_ADMISSION_LEVER_2026-08-23.md` sec8-3 flags its own biggest hole:
"this document has no power calculation, and kernel_mech rev5 was rejected for
exactly that". The honest fix is NOT to import a variance from another
campaign -- lesson #31 (auxiliary numbers imported from another campaign/scale
must have their basis verified first, and this project has been burnt by that
twice). So this file computes the requirement in the OTHER direction:

    given the canon's own headline threshold (a difference below 3% is not a
    headline, methodology gate #3) and the registered n, WHAT BOOT-LEVEL SD
    would make that difference visible?

The answer becomes a Stage-0 PRECONDITION (probe P6): measure the boot-level SD
of goodput first, and only then decide whether the grid is affordable. That
converts an unknown into a gate instead of an assumption.

Cells are compared ACROSS boots (different servers, different jobs), so the
comparison is UNPAIRED -- pairing is unavailable by construction, which is the
same structural fact gate #13's job-axis campaign established for this
substrate.
"""
import argparse
import json
import math

from scipy.stats import nct, t as tdist

BOOT_S = 145.8          # gate #13 measured boot constant (s) -- cited as a
                        # COST constant only, not as a variance
ALPHA = 0.05
POWER = 0.80
REL_DELTA = 0.03        # methodology gate #3: below 3% is not a headline


def n_required(sd_pp, goodput_pp, rel_delta=REL_DELTA, alpha=ALPHA,
               power=POWER, n_max=200):
    """Smallest per-cell n for an unpaired two-sample t test to reach `power`.

    Exact non-central t, not the normal approximation: at n=4 the two differ
    by enough to change the decision.
    """
    delta = rel_delta * goodput_pp
    for n in range(2, n_max + 1):
        df = 2 * n - 2
        ncp = delta / (sd_pp * math.sqrt(2.0 / n))
        crit = tdist.ppf(1 - alpha / 2, df)
        got = 1 - nct.cdf(crit, df, ncp) + nct.cdf(-crit, df, ncp)
        if got >= power:
            return n, got
    return None, None


def detectable_sd(n, goodput_pp, **kw):
    """Largest boot-level SD (pp) at which `n` still reaches the target power."""
    lo, hi = 1e-4, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2
        need, _ = n_required(mid, goodput_pp, n_max=n, **kw)
        if need is not None and need <= n:
            lo = mid
        else:
            hi = mid
    return lo


# ===========================================================================
# rev2 -- PAIRED contrasts with multiplicity control (audit F5/F6/F7)
# ===========================================================================
# ★rev1 asserted that pairing was "structurally impossible". That is FALSE:
#   s8_frontier/batchcap.sbatch:96-99 runs several (cell, cap) combinations
#   inside ONE job and says so itself. The whole power gate stood on that
#   wrong assumption.
# ★rev1 also treated the decision as a single contrast while the registered
#   decision quantity was a 9-cell argmax. Under the corrected design there
#   are TWO contrasts against the incumbent cap, so alpha is split.

def n_required_paired(sd_diff_pp, goodput_pp, n_contrasts=2,
                      rel_delta=REL_DELTA, alpha=ALPHA, power=POWER,
                      n_max=400):
    """Per-arm n for a PAIRED t-test at Bonferroni-corrected alpha."""
    delta = rel_delta * goodput_pp
    a = alpha / max(1, n_contrasts)
    for n in range(2, n_max + 1):
        df = n - 1
        ncp = delta / (sd_diff_pp / math.sqrt(n))
        crit = tdist.ppf(1 - a / 2, df)
        got = 1 - nct.cdf(crit, df, ncp) + nct.cdf(-crit, df, ncp)
        if got >= power:
            return n, got
    return None, None


def detectable_sd_paired(n, goodput_pp, **kw):
    lo, hi = 1e-4, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2
        need, _ = n_required_paired(mid, goodput_pp, n_max=n, **kw)
        lo, hi = (mid, hi) if (need is not None and need <= n) else (lo, mid)
    return lo


def rev2_table():
    """What the corrected design needs, over the plausible paired-SD range.

    The BETWEEN-boot SD for this arm/SLO pair is on record in canon
    (reports/interactive_slo_retune_plan.md:153-158, n=4, chat 300/50). The
    PAIRED-difference SD is NOT on record and is bounded above by sqrt(2) x
    that value (zero correlation) and below by ~0 (perfectly shared boot
    effect). That bracket is the honest state of knowledge, so the table
    spans it and Stage 0 measures where inside it we actually are.
    """
    out = {"note": ("paired-difference SD is bracketed, not assumed: the "
                    "upper end is sqrt(2) x the canon between-boot SD (zero "
                    "correlation), the lower end is a strongly shared boot "
                    "effect. Stage 0 P6 measures it."),
           "n_contrasts": 2, "alpha_per_contrast": ALPHA / 2}
    for goodput_pp in (50.0, 70.0):
        key = f"goodput={goodput_pp:.0f}pp"
        out[key] = {"detectable_sd_diff_pp_at_n":
                    {f"n={n}": round(detectable_sd_paired(n, goodput_pp), 3)
                     for n in (4, 6, 8, 12, 20)},
                    "n_required_by_sd_diff": {}}
        for sd in (0.5, 1.0, 2.0, 3.4, 4.8, 6.8):
            n, got = n_required_paired(sd, goodput_pp)
            # 4 cells per job (3 caps + 1 anchor); boots are jobs
            cost = None if n is None else round(n * 4 * BOOT_S / 3600.0, 2)
            out[key]["n_required_by_sd_diff"][f"sd_diff={sd}pp"] = {
                "n_jobs": n, "achieved_power": None if got is None
                else round(got, 3), "boot_only_GPU_hr_4cells": cost}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="NSL1_POWER_2026-08-23.json")
    a = ap.parse_args()
    cells = 9
    res = {"kind": "nsl1_detectability_precondition",
           "assumptions": {
               "comparison": "unpaired across boots (pairing unavailable: "
                             "different servers/jobs)",
               "test": "two-sample t, exact non-central t power",
               "alpha": ALPHA, "power": POWER,
               "rel_delta": REL_DELTA,
               "boot_cost_s": BOOT_S,
               "note": "NO variance is imported. The design registers the "
                       "measured SD as a Stage-0 gate instead."},
           "detectable_sd_pp_at_n": {},
           "n_required_by_sd": {},
        "rev2_paired": None}

    for goodput_pp in (50.0, 70.0):
        key = f"goodput={goodput_pp:.0f}pp"
        res["detectable_sd_pp_at_n"][key] = {
            f"n={n}": round(detectable_sd(n, goodput_pp), 3)
            for n in (4, 6, 8, 12)}
        row = {}
        for sd in (0.5, 1.0, 2.0, 3.0, 4.8, 8.0):
            n, got = n_required(sd, goodput_pp)
            cost = (None if n is None
                    else round(cells * n * BOOT_S / 3600.0, 2))
            row[f"sd={sd}pp"] = {"n_per_cell": n,
                                 "achieved_power": None if got is None
                                 else round(got, 3),
                                 "boot_only_GPU_hr_9cells": cost}
        res["n_required_by_sd"][key] = row

    res["rev2_paired"] = rev2_table()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"\n-> {a.out}")
    print("\n★ boot_only_GPU_hr counts ONLY server boots (145.8 s each). The "
          "change-trace benchmark time is NOT included -- the design's cost "
          "table stays incomplete until that is measured from existing logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
