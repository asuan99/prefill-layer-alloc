#!/usr/bin/env python3
"""S-6 (telemetry OFF<->ON contrast) -- sample-size calculation.

Fills DESIGN_S6_TELEMETRY_CONTRAST_2026-08-17.md sec3, which registered the
rule "n comes from a power calculation, do NOT inherit B-2's n>=2" but left
the calculation itself unwritten.  Rules-layer artifact: GPU spend 0, no
performance verdict, no grade change.

Discipline notes (why this file looks the way it does):
  * The per-boot estimands are taken from ``g16_analyze.boot_estimands`` via
    ``load_campaign_grid`` -- the pre-registered sec4 definitions -- so nothing
    is re-derived here (design sec2: "G16 사전등록 §4 정의 그대로, 재유도 금지").
  * sigma priors are reported BOTH as point estimates and as one-sided 95%
    chi-square upper bounds.  gate #13 rev3 bought the ``hi_chi2_upper`` band
    point for exactly this reason: a point-estimate n is a coin flip on the
    prior, and prior underestimation loses the whole campaign.
  * The target delta is NOT hard-wired to 5.4%.  That number is imported from
    the S2/alpha cross-campaign comparison (CONSENSUS.md:1396-1397, per-token
    ITL p50, T8 sticky grid) and is basis-unverified for this campaign's
    estimands; lesson #31/gate #32 says import only after basis verification.
    So n is reported as a function of delta and the 5.4% row is labelled.
  * Which library computed the quantiles is recorded (gate #22: silent scipy
    fallbacks do not reach the artifact).
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g16_analyze as g16  # noqa: E402  (canonical, unmodified; sha recorded)

try:
    from scipy import stats as _st  # type: ignore
    QUANT_BACKEND = "scipy"
except Exception:                                       # pragma: no cover
    _st = None
    QUANT_BACKEND = "internal_tables"

# two-sided 97.5% t quantiles, df 1..40 (used only if scipy is absent)
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
         14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
         26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 35: 2.030,
         40: 2.021}
_CHI2_05 = {1: 0.00393, 2: 0.1026, 3: 0.3518, 4: 0.7107, 5: 1.1455, 6: 1.6354,
            7: 2.1673, 8: 2.7326, 9: 3.3251, 10: 3.9403, 15: 7.2609,
            20: 10.851, 21: 11.591, 25: 14.611, 28: 16.928, 30: 18.493}


def t975(df: int) -> float:
    if _st is not None:
        return float(_st.t.ppf(0.975, df))
    if df in _T975:
        return _T975[df]
    keys = sorted(_T975)
    lo = max(k for k in keys if k <= df)
    return _T975[lo]


def chi2_05(df: int) -> float:
    """5th percentile of chi-square(df) -- the tail used by the sigma UB."""
    if _st is not None:
        return float(_st.chi2.ppf(0.05, df))
    if df in _CHI2_05:
        return _CHI2_05[df]
    keys = sorted(_CHI2_05)
    lo = max(k for k in keys if k <= df)
    return _CHI2_05[lo]


def sigma_ub95(sd: float, df: int) -> float:
    """One-sided upper 95% bound on sigma from a sample SD with df."""
    return sd * math.sqrt(df / chi2_05(df))


METRICS = ("M_ttft", "M_itl")
ARM = "d44"


def collect(directory: Path):
    out = {}
    for phase in ("LO", "HI"):
        grid, meta = g16.load_campaign_grid(directory, phase)
        per_arm = {}
        for rec in grid:
            row = per_arm.setdefault(rec.arm, [])
            row.append({
                "block": rec.block, "boot": rec.boot, "node": rec.node,
                "git_commit": rec.git_commit, "manifest_sha": rec.manifest_sha,
                "boot_s": rec.boot_s,
                "duration_s": rec.est["duration_s"],
                **{m: rec.est[m] for m in METRICS},
            })
        out[phase] = {"per_arm": per_arm, "n_adopted": meta["n_adopted"]}
    return out


def cv_stats(values):
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "cv_pct": 100.0 * sd / mean,
            "df": n - 1}


def pooled_cv(per_arm, metric):
    """Pool the WITHIN-ARM relative dispersion across arms (df adds up).

    Assumption made explicit: constant coefficient of variation across arms.
    The heteroscedasticity ratio is reported so the auditor can reject it.
    """
    num = den = 0.0
    cvs = {}
    for arm, rows in sorted(per_arm.items()):
        vals = [r[metric] for r in rows]
        if len(vals) < 2:
            continue
        st = cv_stats(vals)
        cvs[arm] = st["cv_pct"]
        num += st["df"] * (st["cv_pct"] ** 2)
        den += st["df"]
    return {"cv_pct": math.sqrt(num / den), "df": int(den), "per_arm_cv_pct": cvs,
            "hetero_ratio_max_min": max(cvs.values()) / min(cvs.values())}


def boot_cost_seconds(directory: Path):
    """Per-boot GPU occupancy from this campaign's own sidecars.

    ``t_boot1 - t_boot0`` is the whole boot slot (spawn -> teardown), so it
    already contains warm-up, both phases and shutdown.  No cost constant is
    imported: gate #13's 144 s/boot belongs to the s8_scaleup workload
    (different model, different NP/ROUNDS) and lesson #31 forbids carrying it.
    """
    spans, jobs = [], {}
    for sc in sorted(directory.glob("g16_*_sidecar.json")):
        blob = json.loads(sc.read_text(encoding="utf-8"))
        if blob.get("mode") == "smoke" or blob.get("t_boot1") is None:
            continue
        span = blob["t_boot1"] - blob["t_boot0"]
        spans.append(span)
        job = sc.name.split("_")[-2]           # g16_<blk>_<arm>_boot<n>_<JOBID>_sidecar.json
        jobs.setdefault(job, []).append(span)
    # sacct 2026-08-21: 884336 01:01:14 / 884410 00:58:46 / 884411 00:58:45 /
    # 884412 00:57:26 (Elapsed, COMPLETED, 1 node each).  Recorded here because
    # the job-level overhead (setup + warm-up boot + teardown) is not visible
    # in the sidecars, and S-6 pays it once per job.
    elapsed = {"884336": 3674, "884410": 3526, "884411": 3525, "884412": 3446}
    overhead = {j: elapsed[j] - sum(v) for j, v in jobs.items() if j in elapsed}
    return {
        "n_boots": len(spans),
        "job_elapsed_s_sacct": elapsed,
        "job_overhead_s": overhead,
        "job_overhead_s_mean": (statistics.fmean(overhead.values())
                                if overhead else None),
        "boot_slot_s_mean": statistics.fmean(spans),
        "boot_slot_s_max": max(spans),
        "boot_slot_s_min": min(spans),
        "per_job_boot_sum_s": {j: sum(v) for j, v in sorted(jobs.items())},
    }


def variance_components(per_arm, metric):
    """Split within-arm dispersion into a BLOCK part and a residual part.

    Two-way additive model on log(metric) over the balanced arm x block grid,
    so the components read as relative dispersion.  Why this matters for S-6:
    the design pairs OFF/ON *within a boot pair*.  Pairing can only cancel a
    shared component, and the block (= one job, one node, one contiguous
    window) term is the only shared component this campaign can see.

      rho = var_block / (var_block + var_resid)
      SD(paired difference) = sqrt(2 * (1 - rho)) * sigma_boot

    rho is reported, never assumed: the headline n stays at rho = 0.
    """
    arms = sorted(per_arm)
    blocks = sorted({r["block"] for rows in per_arm.values() for r in rows})
    cell = {}
    for arm, rows in per_arm.items():
        for r in rows:
            cell[(arm, r["block"])] = math.log(r[metric])
    if len(cell) != len(arms) * len(blocks):
        return {"balanced": False, "rho": None}
    grand = statistics.fmean(cell.values())
    arm_mean = {a: statistics.fmean([cell[(a, b)] for b in blocks]) for a in arms}
    blk_mean = {b: statistics.fmean([cell[(a, b)] for a in arms]) for b in blocks}
    na, nb = len(arms), len(blocks)
    ss_blk = na * sum((blk_mean[b] - grand) ** 2 for b in blocks)
    ss_res = sum((cell[(a, b)] - arm_mean[a] - blk_mean[b] + grand) ** 2
                 for a in arms for b in blocks)
    ms_blk = ss_blk / (nb - 1)
    ms_res = ss_res / ((na - 1) * (nb - 1))
    var_blk = max(0.0, (ms_blk - ms_res) / na)      # negative -> clamp to 0
    rho = var_blk / (var_blk + ms_res) if (var_blk + ms_res) > 0 else 0.0
    return {
        "balanced": True, "n_arms": na, "n_blocks": nb,
        "sd_block_pct": 100.0 * math.sqrt(var_blk),
        "sd_resid_pct": 100.0 * math.sqrt(ms_res),
        "sd_total_within_arm_pct": 100.0 * math.sqrt(var_blk + ms_res),
        "rho_block_share": rho,
        "ms_blk_over_ms_res": ms_blk / ms_res if ms_res > 0 else None,
        "df_block": nb - 1, "df_resid": (na - 1) * (nb - 1),
        "note": ("var_block was clamped at 0 (MS_block <= MS_resid): the data "
                 "give no evidence of a shared block component, so pairing "
                 "buys nothing detectable here" if var_blk == 0.0 else
                 "a shared block component is present; pairing within a block "
                 "would cancel it, but only if the OFF/ON pair really is "
                 "adjacent on one node in one window"),
    }


def chi2_cdf(x: float, df: int) -> float:
    if _st is not None:
        return float(_st.chi2.cdf(x, df))
    raise RuntimeError("assurance needs scipy; rerun under the project venv")


def assurance(cv_true_pct: float, delta_pct: float, n: int,
              rho: float = 0.0) -> float:
    """P(realized 95% half-width <= delta) when the TRUE dispersion is cv_true.

    The realized half-width is t*S/sqrt(n) with S random, so "n from a point
    estimate" is a coin flip even when the prior is right -- gate #13 rev3's
    reason for buying the upper band point.  This is the same P(pass) quantity
    that campaign registered, computed for S-6 instead of guessed.
    """
    sd_d = math.sqrt(2.0 * (1.0 - rho)) * cv_true_pct
    df = n - 1
    crit = delta_pct * math.sqrt(n) / t975(df)          # need S_d <= crit
    return chi2_cdf(df * (crit / sd_d) ** 2, df)


def required_n(cv_pct: float, delta_pct: float, rho: float = 0.0,
               n_max: int = 400):
    """Smallest number of BOOT PAIRS whose 95% CI half-width <= delta.

    Paired design: pair i contributes d_i = metric(OFF,i) - metric(ON,i).
    SD(d) = sqrt(2*(1-rho)) * sigma_boot   (rho = 0 is the conservative case:
    no shared node/date component is credited to the pairing).
    half-width(n) = t_{.975, n-1} * SD(d) / sqrt(n)
    """
    sd_d = math.sqrt(2.0 * (1.0 - rho)) * cv_pct
    for n in range(2, n_max + 1):
        if t975(n - 1) * sd_d / math.sqrt(n) <= delta_pct:
            return n, t975(n - 1) * sd_d / math.sqrt(n)
    return None, None


def main() -> None:
    directory = HERE
    data = collect(directory)
    report = {
        "frame": ("S-6 sample-size calculation. Rules-layer input, NOT a "
                  "pre-registration and NOT a performance verdict. GPU 0."),
        "prior_source": {
            "campaign": "G16 (jobs 884336/884410/884411/884412), 4 blocks",
            "loader": "g16_analyze.load_campaign_grid (unmodified)",
            "analyzer_sha256": g16._self_sha256(),
            "estimands": "g16_analyze.boot_estimands -- prereg sec4 M_ttft/M_itl",
            "caveat": ("all G16 boots are telemetry=ON; the OFF leg's variance "
                       "is unobserved and assumed equal -- unverifiable pre-hoc"),
        },
        "quantile_backend": QUANT_BACKEND,
        "phases": {},
    }

    for phase, blob in data.items():
        per_arm = blob["per_arm"]
        entry = {"n_adopted_boots": blob["n_adopted"], "arms": sorted(per_arm)}
        for metric in METRICS:
            arm_rows = per_arm.get(ARM, [])
            st = cv_stats([r[metric] for r in arm_rows])
            pool = pooled_cv(per_arm, metric)
            cv_ub_arm = 100.0 * sigma_ub95(st["sd"], st["df"]) / st["mean"]
            cv_ub_pool = pool["cv_pct"] * math.sqrt(pool["df"] / chi2_05(pool["df"]))
            rows = {}
            for delta in (2.0, 3.0, 5.4, 8.0, 10.0):
                r = {}
                for tag, cv in (("arm_point", st["cv_pct"]),
                                ("arm_ub95", cv_ub_arm),
                                ("pooled_point", pool["cv_pct"]),
                                ("pooled_ub95", cv_ub_pool)):
                    n, hw = required_n(cv, delta)
                    r[tag] = {"n_pairs": n, "n_boots": None if n is None else 2 * n,
                              "half_width_pct": hw}
                rows[f"delta_{delta}pct"] = r
            assur = {}
            for delta in (3.0, 5.4):
                assur[f"delta_{delta}pct"] = {
                    f"n{n}": {
                        "sigma_true=point": assurance(st["cv_pct"], delta, n),
                        "sigma_true=ub95": assurance(cv_ub_arm, delta, n),
                        "sigma_true=pooled_point": assurance(pool["cv_pct"], delta, n),
                    } for n in (3, 4, 5, 6, 8, 10, 12, 16, 20)}
            entry[metric] = {
                "assurance_P_halfwidth_le_delta": assur,
                "variance_components": variance_components(per_arm, metric),
                "arm_d44": st,
                "arm_d44_cv_ub95_pct": cv_ub_arm,
                "pooled_within_arm": pool,
                "pooled_cv_ub95_pct": cv_ub_pool,
                "required_n": rows,
            }
        # cost basis, from this campaign's own boots (no imported constant)
        boot_s = [r["boot_s"] for rows in per_arm.values() for r in rows
                  if r["boot_s"] is not None]
        dur = [r["duration_s"] for rows in per_arm.values() for r in rows]
        entry["cost_basis_s"] = {
            "boot_s_mean": statistics.fmean(boot_s) if boot_s else None,
            "boot_s_max": max(boot_s) if boot_s else None,
            "bench_duration_s_mean": statistics.fmean(dur) if dur else None,
            "n": len(dur),
        }
        report["phases"][phase] = entry
    report["cost_basis"] = boot_cost_seconds(directory)

    out = HERE / "S6_POWER_2026-08-21.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    print(json.dumps(report, indent=2)[:1])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
