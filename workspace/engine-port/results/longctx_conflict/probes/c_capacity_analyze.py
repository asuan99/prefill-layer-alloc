#!/usr/bin/env python3
"""Per-cell analysis for PREREG_CAPACITY_2026-09-09.md (probe C).

PRIMARY estimator is CLIENT-SIDE and EXACT -- Little's law in integral form
over `bench_serving --output-details`:

    L_decode = sum_i sum(itls_i) / duration    # first token -> last token
    L_pre    = sum_i ttft_i      / duration    # arrival -> first token
    L_total  = L_pre + L_decode                # cross-checked against the
                                               # independently reported
                                               # `concurrency` field

It uses no arrival timestamps and no telemetry snapshots, so it is immune to
the sample-interval imputation artifact that this track has been bitten by
(CONSENSUS sec 3 item 120, methodology gate #103).  On the two existing P1
cells the cross-check closes to 2.1e-07 / 2.4e-08 relative error.

Telemetry quantities (f_time, f_count, decode_sms histogram) are SECONDARY /
diagnostic and are obtained by importing `analyze_probes.analyze_one`
verbatim -- which itself imports the canonical `compute_decode_realized` from
`s8_frontier/e1_pin_check.py` (methodology gate #9: do not reimplement the
canonical function).  They are reported next to, never in place of, the exact
client-side numbers, so that any disagreement between the two is visible
rather than silently resolved.  (On P1 d44 the count-weighted and
time-weighted telemetry means of `decode_running_batch_size` differ by 21x;
the exact client-side value is 1.336 and the time-weighted one 1.781.)

Usage:
  python3 c_capacity_analyze.py <bench.jsonl> --telemetry <t.jsonl> --expect-d D
                                --out out.json [--label L]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# P2 job 905712, ctx 8192, concurrency 1, uncontended.  Registered constants:
# PREREG_CAPACITY sec 2.  Do NOT recompute these from the probe's own runs --
# they are the pre-treatment reference and reusing run-internal values here
# would make R and s(D) circular.
FLOOR_REF_8192_S = 0.9104067257139832
ITL_SOLO_S = 0.01303689880296588


def percentile(xs, q):
    """Linear-interpolated percentile; `q` in [0,1]. Empty -> nan."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def client_side(bench_path):
    d = json.loads(Path(bench_path).read_text().strip().split("\n")[0])
    errs = d["errors"]
    ok = [i for i, e in enumerate(errs) if e == ""]
    ttfts = [d["ttfts"][i] for i in ok]
    itls = [d["itls"][i] for i in ok]
    dur = d["duration"]

    sum_dec = sum(sum(v) for v in itls)
    sum_pre = sum(ttfts)
    L_dec = sum_dec / dur
    L_pre = sum_pre / dur
    reported = d.get("concurrency", float("nan"))
    cross_err = (
        abs(L_pre + L_dec - reported) / reported
        if reported and reported == reported
        else float("nan")
    )

    flat_itls = [x for v in itls for x in v]
    out_len = d["random_output_len"]
    R = FLOOR_REF_8192_S / (FLOOR_REF_8192_S + out_len * ITL_SOLO_S)
    itl_p50 = percentile(flat_itls, 0.50)
    mu_solo = 1.0 / FLOOR_REF_8192_S
    mu_ach = d["request_throughput"]
    s_D = mu_solo / mu_ach if mu_ach > 0 else float("inf")
    pred = ((1 - R) / R) * (itl_p50 / ITL_SOLO_S) / s_D if s_D not in (0, float("inf")) else float("nan")

    offered = d["request_rate"]
    return dict(
        offered_rate=offered,
        achieved_rate=mu_ach,
        achieved_over_offered=(mu_ach / offered) if offered and offered == offered else float("nan"),
        duration_s=dur,
        completed=d["completed"],
        n_errors=len(errs) - len(ok),
        random_input_len=d["random_input_len"],
        random_output_len=out_len,
        ttft_p50_s=percentile(ttfts, 0.50),
        ttft_p95_s=percentile(ttfts, 0.95),
        ttft_p99_s=percentile(ttfts, 0.99),
        ttft_p50_over_floor_ref=percentile(ttfts, 0.50) / FLOOR_REF_8192_S,
        itl_p50_s=itl_p50,
        itl_p95_s=percentile(flat_itls, 0.95),
        itl_p50_over_itl_solo=itl_p50 / ITL_SOLO_S,
        L_total_exact=L_pre + L_dec,
        L_pre_exact=L_pre,
        L_decode_exact=L_dec,
        L_total_reported=reported,
        crosscheck_rel_err=cross_err,
        R=R,
        C_max_algebraic=(1 - R) / R,
        prefill_slowdown_s_D=s_D,
        L_decode_model_pred=pred,
        L_decode_obs_over_pred=(L_dec / pred) if pred and pred == pred and pred != 0 else float("nan"),
    )


def telemetry_side(telemetry_path, expect_d):
    try:
        import analyze_probes  # same directory
        t = analyze_probes.analyze_one(telemetry_path, expect_d)
    except Exception as exc:  # diagnostic only -- never fail the cell on this
        return {"telemetry_error": f"{type(exc).__name__}: {exc}"}
    return {
        "f_time": t.get("f_time"),
        "f_count": t.get("f_count"),
        "decode_hist": t.get("decode_hist"),
        "decode_hist_mode": t.get("decode_hist_mode"),
        "n_decode_active_snapshots": t.get("n_decode_active_snapshots"),
        "n_decode_active_at_target": t.get("n_decode_active_at_target"),
        "n_split_total": t.get("n_split_total"),
        "n_split_pab0": t.get("n_split_pab0"),
        "telemetry_audit": t.get("audit"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_path")
    ap.add_argument("--telemetry", default=None)
    ap.add_argument("--expect-d", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    res = {"probe": "C_CAPACITY", "expect_d": args.expect_d,
           "bench_path": str(args.bench_path)}
    if args.label:
        res["label"] = args.label
    res.update(client_side(args.bench_path))
    if args.telemetry and Path(args.telemetry).exists():
        res["telemetry"] = telemetry_side(args.telemetry, args.expect_d)
    else:
        res["telemetry"] = {"telemetry_error": "missing"}

    text = json.dumps(res, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
