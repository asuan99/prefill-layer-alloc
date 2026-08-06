"""C-2 (paired-t vs percentile bootstrap on the canonical predicate) and
C-3 (threshold ladder violation counts / sign boundary).

All scoring goes through the canonical ``pdmux_eval.analyze`` predicate objects.
The only local statistics are the Student-t interval (df = n-1 = 4), whose critical
value is verified numerically against the t density in ``tcrit_check()`` rather than
copied from another agent's table.

Aggregation units (declared):
  scoring unit      = request
  goodput unit      = cell (model, arm, rate, rep) -> good/duration
  replication unit  = rep (n=5), paired on rep index (identical seed => identical
                      prompt set and Poisson arrival stream; verified by input_lens)
  interval          = over the 5 paired REP differences.  No request-level pooling.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/benchmarks")
sys.path.insert(0, str(Path(__file__).parent))
from pdmux_eval.analyze import paired_bootstrap_ci, percentile  # noqa: E402
from verify_load import ARMS, MODELS, REPS, TTFT_SLO_MS, load_cells  # noqa: E402

OUT = Path(__file__).parent
LADDER = (40.0, 48.0, 50.0, 60.0, 80.0, 100.0, 150.0, 300.0)
RATES = (2.0, 3.0, 4.0, 6.0)


# ---------------------------------------------------------------- t machinery
def t_pdf(x: float, df: int) -> float:
    return (
        math.gamma((df + 1) / 2)
        / (math.sqrt(df * math.pi) * math.gamma(df / 2))
        * (1 + x * x / df) ** (-(df + 1) / 2)
    )


def t_cdf(x: float, df: int, panels: int = 200000, lo: float = -60.0) -> float:
    """Simpson integration of the t density -- independent of any tabulated value."""
    h = (x - lo) / panels
    total = t_pdf(lo, df) + t_pdf(x, df)
    for i in range(1, panels):
        total += (4 if i % 2 else 2) * t_pdf(lo + i * h, df)
    return total * h / 3


def tcrit_check(df: int = 4) -> float:
    lo, hi = 1.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < 0.975:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


TCRIT4 = 2.7764451051977987


def paired_t(diffs):
    n = len(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    half = TCRIT4 * sd / math.sqrt(n)
    tstat = mean / (sd / math.sqrt(n)) if sd else math.inf
    return {"mean": mean, "sd": sd, "se": sd / math.sqrt(n), "t": tstat,
            "lo": mean - half, "hi": mean + half}


# ---------------------------------------------------------------- predicates
def passes(req, itl_slo_ms, missing_itl_is_violation=True):
    """Canonical predicate with the ITL threshold as a free knob (C-3 ladder).

    ``RequestResult.passes`` maps an empty ITL tuple to percentile()->nan, and
    ``nan <= T`` is False, i.e. missing ITL counts as a VIOLATION.  That is
    reproduced here exactly; the alternative convention is exposed for the
    C-2 sensitivity check.
    """
    if req.ttft_ms > TTFT_SLO_MS:
        return False
    if not req.token_itl_ms:
        return not missing_itl_is_violation
    return percentile(req.token_itl_ms, 0.95) <= itl_slo_ms


def goodput(cell, itl_slo_ms, missing_itl_is_violation=True):
    good = sum(passes(r, itl_slo_ms, missing_itl_is_violation) for r in cell["requests"])
    return good / cell["duration"]


def main():
    cells = load_cells()
    report = {}

    print(f"t_(0.975, df=4) numerically inverted = {tcrit_check():.9f}  "
          f"(hardcoded {TCRIT4:.9f})")

    # ---------------------------------------------------------------- C-2
    print("\n=== C-2: canonical predicate (TTFT<=3s AND per-request p95 ITL<=60ms) ===")
    print("unit: per-rep cell goodput (good/duration); n=5 paired reps\n")
    hdr = (f"{'model':22s} {'rate':>4s} {'plain mean+-SD':>18s} {'agn mean':>9s} "
           f"{'eff%':>8s} {'boot CI95':>22s} {'t CI95 (df=4)':>22s} {'t':>7s} {'boot0':>6s} {'t0':>4s}")
    print(hdr)
    c2 = []
    for model in sorted(MODELS):
        for rate in RATES:
            base = {rep: goodput(cells[(model, "plain", rate, rep)], 60.0) for rep in REPS}
            prop = {rep: goodput(cells[(model, "agnostic", rate, rep)], 60.0) for rep in REPS}
            boot = paired_bootstrap_ci(base, prop)
            diffs = [prop[r] - base[r] for r in REPS]
            t = paired_t(diffs)
            bmean = statistics.fmean(base.values())
            bsd = statistics.stdev(base.values())
            row = {
                "model": model, "rate": rate,
                "plain_mean": bmean, "plain_sd": bsd,
                "agn_mean": statistics.fmean(prop.values()),
                "agn_sd": statistics.stdev(prop.values()),
                "mean_effect": t["mean"], "effect_percent": 100 * t["mean"] / bmean,
                "sd_effect": t["sd"], "t_stat": t["t"],
                "boot_lo": boot["ci95_low"], "boot_hi": boot["ci95_high"],
                "t_lo": t["lo"], "t_hi": t["hi"],
                "boot_excludes_0": (boot["ci95_low"] > 0) or (boot["ci95_high"] < 0),
                "t_excludes_0": (t["lo"] > 0) or (t["hi"] < 0),
                "boot_width": boot["ci95_high"] - boot["ci95_low"],
                "t_width": t["hi"] - t["lo"],
                "per_rep_diffs": diffs,
            }
            c2.append(row)
            print(f"{model:22s} {rate:4.0f} {bmean:9.4f}+-{bsd:6.4f} {row['agn_mean']:9.4f} "
                  f"{row['effect_percent']:+8.2f} "
                  f"[{boot['ci95_low']:+8.4f},{boot['ci95_high']:+8.4f}] "
                  f"[{t['lo']:+8.4f},{t['hi']:+8.4f}] {t['t']:7.3f} "
                  f"{str(row['boot_excludes_0']):>6s} {str(row['t_excludes_0']):>4s}")
    report["c2"] = c2

    # ratio of widths (relevant to C-1's "0.58x" claim, measured on real data)
    ratios = [r["boot_width"] / r["t_width"] for r in c2]
    print(f"\nboot/t CI width ratio on these 8 real cells: "
          f"min {min(ratios):.4f} max {max(ratios):.4f} mean {statistics.fmean(ratios):.4f}")
    report["width_ratio_real_cells"] = ratios

    # ---------------------------------------------------------------- C-2b: missing-ITL convention
    print("\n=== C-2b: missing-ITL convention sensitivity (violation vs pass) ===")
    conv = []
    for model in sorted(MODELS):
        for rate in RATES:
            out = {}
            for flag in (True, False):
                base = {rep: goodput(cells[(model, "plain", rate, rep)], 60.0, flag) for rep in REPS}
                prop = {rep: goodput(cells[(model, "agnostic", rate, rep)], 60.0, flag) for rep in REPS}
                diffs = [prop[r] - base[r] for r in REPS]
                t = paired_t(diffs)
                out[flag] = {
                    "effect_percent": 100 * t["mean"] / statistics.fmean(base.values()),
                    "t_lo": t["lo"], "t_hi": t["hi"],
                    "boot": paired_bootstrap_ci(base, prop),
                }
            n_missing = {
                arm: sum(
                    sum(1 for r in cells[(model, arm, rate, rep)]["requests"] if not r.token_itl_ms)
                    for rep in REPS
                )
                for arm in ARMS
            }
            conv.append({"model": model, "rate": rate, "n_missing": n_missing,
                         "violation": out[True], "pass": out[False]})
            print(f"{model:22s} r{rate:.0f} missing-ITL reqs plain={n_missing['plain']:2d} "
                  f"agn={n_missing['agnostic']:2d} | eff% viol-conv {out[True]['effect_percent']:+7.2f} "
                  f"t[{out[True]['t_lo']:+.4f},{out[True]['t_hi']:+.4f}] | pass-conv "
                  f"{out[False]['effect_percent']:+7.2f} "
                  f"t[{out[False]['t_lo']:+.4f},{out[False]['t_hi']:+.4f}]")
    report["c2b_missing_itl"] = conv

    # ---------------------------------------------------------------- C-3 ladder
    print("\n=== C-3: threshold ladder -- violation counts (5 reps POOLED, 600 req/cell) ===")
    print(f"{'model':22s} {'rate':>4s} {'arm':>9s} " + " ".join(f"T={t:<5.0f}" for t in LADDER)
          + "  |  TTFT>3s  missingITL")
    ladder_counts = []
    for model in sorted(MODELS):
        for rate in RATES:
            for arm in ARMS:
                cs = [cells[(model, arm, rate, rep)] for rep in REPS]
                reqs = [r for c in cs for r in c["requests"]]
                counts = {T: sum(1 for r in reqs if not passes(r, T)) for T in LADDER}
                n_ttft = sum(1 for r in reqs if r.ttft_ms > TTFT_SLO_MS)
                n_missing = sum(1 for r in reqs if not r.token_itl_ms)
                # violations attributable purely to the ITL axis at each T
                counts_itl_only = {
                    T: sum(1 for r in reqs if r.token_itl_ms
                           and percentile(r.token_itl_ms, 0.95) > T)
                    for T in LADDER
                }
                ladder_counts.append({"model": model, "rate": rate, "arm": arm,
                                      "n_req": len(reqs), "violations": counts,
                                      "itl_only_violations": counts_itl_only,
                                      "n_ttft_violation": n_ttft,
                                      "n_missing_itl": n_missing})
                print(f"{model:22s} {rate:4.0f} {arm:>9s} "
                      + " ".join(f"{counts[T]:5d} " for T in LADDER)
                      + f"  |  {n_ttft:5d}   {n_missing:5d}")
    report["c3_ladder_counts"] = ladder_counts

    print("\n=== C-3: paired effect at each ladder threshold (n=5 reps) ===")
    print(f"{'model':22s} {'rate':>4s} {'T':>5s} {'plain':>8s} {'agn':>8s} {'eff%':>9s} "
          f"{'boot CI95':>22s} {'t CI95':>22s} {'sign':>5s} {'bootExcl0':>9s} {'tExcl0':>7s} "
          f"{'viol(p/a)':>11s}")
    ladder_eff = []
    for model in sorted(MODELS):
        for rate in RATES:
            for T in LADDER:
                base = {rep: goodput(cells[(model, "plain", rate, rep)], T) for rep in REPS}
                prop = {rep: goodput(cells[(model, "agnostic", rate, rep)], T) for rep in REPS}
                diffs = [prop[r] - base[r] for r in REPS]
                bmean = statistics.fmean(base.values())
                eff = statistics.fmean(diffs)
                if statistics.stdev(diffs) == 0.0:
                    t = {"lo": eff, "hi": eff, "sd": 0.0, "mean": eff, "t": math.nan,
                         "se": 0.0}
                    boot = {"ci95_low": eff, "ci95_high": eff}
                else:
                    t = paired_t(diffs)
                    boot = paired_bootstrap_ci(base, prop)
                vp = sum(1 for rep in REPS
                         for r in cells[(model, "plain", rate, rep)]["requests"]
                         if not passes(r, T))
                va = sum(1 for rep in REPS
                         for r in cells[(model, "agnostic", rate, rep)]["requests"]
                         if not passes(r, T))
                row = {"model": model, "rate": rate, "T": T,
                       "plain_mean": bmean, "agn_mean": statistics.fmean(prop.values()),
                       "mean_effect": eff,
                       "effect_percent": 100 * eff / bmean if bmean else math.nan,
                       "boot_lo": boot["ci95_low"], "boot_hi": boot["ci95_high"],
                       "t_lo": t["lo"], "t_hi": t["hi"],
                       "sign": (0 if eff == 0 else (1 if eff > 0 else -1)),
                       "boot_excludes_0": boot["ci95_low"] > 0 or boot["ci95_high"] < 0,
                       "t_excludes_0": t["lo"] > 0 or t["hi"] < 0,
                       "viol_plain": vp, "viol_agn": va,
                       "per_rep_diffs": diffs}
                ladder_eff.append(row)
                print(f"{model:22s} {rate:4.0f} {T:5.0f} {bmean:8.4f} {row['agn_mean']:8.4f} "
                      f"{row['effect_percent']:+9.2f} "
                      f"[{row['boot_lo']:+8.4f},{row['boot_hi']:+8.4f}] "
                      f"[{row['t_lo']:+8.4f},{row['t_hi']:+8.4f}] "
                      f"{'+' if row['sign'] > 0 else ('-' if row['sign'] < 0 else '0'):>5s} "
                      f"{str(row['boot_excludes_0']):>9s} {str(row['t_excludes_0']):>7s} "
                      f"{vp:5d}/{va:<5d}")
    report["c3_ladder_effects"] = ladder_eff

    (OUT / "c2_c3_results.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {OUT / 'c2_c3_results.json'}")


if __name__ == "__main__":
    main()
