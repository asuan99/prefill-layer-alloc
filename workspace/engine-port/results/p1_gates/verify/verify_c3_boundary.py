"""C-3 fine boundary: exact threshold where each cell (a) becomes vacuous
(zero ITL-axis violations in BOTH arms) and (b) where the paired sign flips.
Plus forensics on the missing-ITL requests (are they instrumentation gaps or
genuinely single-token generations?).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/benchmarks")
sys.path.insert(0, str(Path(__file__).parent))
from pdmux_eval.analyze import percentile  # noqa: E402
from verify_load import ARMS, MODELS, REPS, TTFT_SLO_MS, load_cells  # noqa: E402
from verify_c2_c3 import RATES, goodput, paired_t  # noqa: E402


def main():
    cells = load_cells()
    out = {}

    print("=== missing-ITL forensics (empty itls array) ===")
    print(f"{'model':22s} {'rate':>4s} {'arm':>9s} {'rep':>3s} {'idx':>4s} {'output_len':>10s} "
          f"{'input_len':>9s} {'ttft_ms':>9s}")
    missing = []
    for model in sorted(MODELS):
        for rate in RATES:
            for arm in ARMS:
                for rep in REPS:
                    c = cells[(model, arm, rate, rep)]
                    for i, r in enumerate(c["requests"]):
                        if not r.token_itl_ms:
                            missing.append({"model": model, "rate": rate, "arm": arm,
                                            "rep": rep, "index": i,
                                            "output_len": c["output_lens"][i],
                                            "input_len": c["input_lens"][i],
                                            "ttft_ms": r.ttft_ms})
                            print(f"{model:22s} {rate:4.0f} {arm:>9s} {rep:3d} {i:4d} "
                                  f"{c['output_lens'][i]:10d} {c['input_lens'][i]:9d} "
                                  f"{r.ttft_ms:9.1f}")
    out["missing_itl_requests"] = missing
    print(f"total missing-ITL requests: {len(missing)}")
    ol = sorted({m['output_len'] for m in missing})
    print(f"distinct output_len among them: {ol}")

    print("\n=== per-cell ITL-axis extremes and vacuity threshold ===")
    print(f"{'model':22s} {'rate':>4s} {'arm':>9s} {'max req-p95-ITL(ms)':>20s} "
          f"{'p100 of req p95':>16s} {'n_missing':>9s} {'n_ttft>3s':>9s}")
    vac = {}
    for model in sorted(MODELS):
        for rate in RATES:
            for arm in ARMS:
                reqs = [r for rep in REPS for r in cells[(model, arm, rate, rep)]["requests"]]
                p95s = [percentile(r.token_itl_ms, 0.95) for r in reqs if r.token_itl_ms]
                mx = max(p95s)
                vac[(model, rate, arm)] = mx
                print(f"{model:22s} {rate:4.0f} {arm:>9s} {mx:20.2f} {'':16s} "
                      f"{sum(1 for r in reqs if not r.token_itl_ms):9d} "
                      f"{sum(1 for r in reqs if r.ttft_ms > TTFT_SLO_MS):9d}")
    out["max_request_p95_itl_ms"] = {f"{k[0]}|r{k[1]:.0f}|{k[2]}": v for k, v in vac.items()}

    print("\n=== exact sign-flip / vacuity boundary per cell ===")
    print(f"{'model':22s} {'rate':>4s} {'T_vacuous(both arms)':>21s} {'T_signflip':>11s} "
          f"{'sign@40':>7s} {'sign@300':>8s} {'eff%@300':>9s} {'boot excl0@300':>14s} "
          f"{'t excl0@300':>11s}")
    boundary = []
    grid = [round(x, 1) for x in [40 + 0.5 * i for i in range(0, 521)]]  # 40..300 step .5
    for model in sorted(MODELS):
        for rate in RATES:
            t_vac = max(vac[(model, rate, "plain")], vac[(model, rate, "agnostic")])
            eff_by_T = {}
            for T in grid:
                b = statistics.fmean(goodput(cells[(model, "plain", rate, rep)], T) for rep in REPS)
                a = statistics.fmean(goodput(cells[(model, "agnostic", rate, rep)], T) for rep in REPS)
                eff_by_T[T] = a - b
            flip = None
            s0 = 1 if eff_by_T[grid[0]] > 0 else (-1 if eff_by_T[grid[0]] < 0 else 0)
            for T in grid:
                s = 1 if eff_by_T[T] > 0 else (-1 if eff_by_T[T] < 0 else 0)
                if s != s0:
                    flip = T
                    break
            diffs300 = [goodput(cells[(model, "agnostic", rate, rep)], 300.0)
                        - goodput(cells[(model, "plain", rate, rep)], 300.0) for rep in REPS]
            base300 = statistics.fmean(goodput(cells[(model, "plain", rate, rep)], 300.0)
                                       for rep in REPS)
            t300 = paired_t(diffs300)
            row = {"model": model, "rate": rate, "T_vacuous_ms": t_vac,
                   "T_signflip_ms": flip, "sign_at_40": s0,
                   "effect_percent_at_300": 100 * t300["mean"] / base300,
                   "t_lo_300": t300["lo"], "t_hi_300": t300["hi"],
                   "t_excl0_300": t300["lo"] > 0 or t300["hi"] < 0}
            boundary.append(row)
            print(f"{model:22s} {rate:4.0f} {t_vac:21.2f} "
                  f"{('%.1f' % flip) if flip else 'none':>11s} "
                  f"{'+' if s0 > 0 else '-':>7s} "
                  f"{'+' if t300['mean'] > 0 else '-':>8s} "
                  f"{row['effect_percent_at_300']:+9.2f} "
                  f"{'':14s} {str(row['t_excl0_300']):>11s}")
    out["boundary"] = boundary

    print("\n=== pairing integrity: input_lens identical across arms per (rate,rep)? ===")
    ok = True
    for model in sorted(MODELS):
        for rate in RATES:
            for rep in REPS:
                a = cells[(model, "plain", rate, rep)]["input_lens"]
                b = cells[(model, "agnostic", rate, rep)]["input_lens"]
                if a != b:
                    ok = False
                    print(f"MISMATCH {model} r{rate} rep{rep}")
    print("all 40 (model,rate,rep) pairs have identical input_lens:", ok)
    out["pairing_input_lens_identical"] = ok

    Path(Path(__file__).parent / "c3_boundary.json").write_text(json.dumps(out, indent=1, default=str))
    print("\nwrote c3_boundary.json")


if __name__ == "__main__":
    main()
