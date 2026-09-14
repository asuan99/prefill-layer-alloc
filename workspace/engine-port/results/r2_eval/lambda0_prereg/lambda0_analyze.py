#!/usr/bin/env python3
"""Per-cell analyzer for campaign stage 0 (lambda*), rev4.

PRIMARY ESTIMATOR, AND WHAT IT ACTUALLY IS (rev3, forced by the rev2 audit):

    achieved_over_realized = achieved_rate / realized_offered_rate
                           = (N/(N-1)) * span / duration
                           = (N/(N-1)) * span / (span + L)

    span = sum of the replayed inter-arrivals     (arrival window)
    L    = duration - span                        (DRAIN: time after the last
                                                   arrival that the run still
                                                   needs to finish)

This is an IDENTITY, not a modelling choice, and it is exact on all 12 archived
probe C cells (`duration = max_i(arrival_i + ttft_i + sum itl_i)` reproduces the
recorded duration to <= 0.265 s).  ★Therefore an UNSATURATED cell does not read
1: it reads `kappa = (N/(N-1))/(1 + L/span)`.  rev2 registered this estimator
having validated it only on (8192, 96) cells where L/span is 0.7-1.7%; on
(256, 512) the drain is 511 tokens x ITL and rev2's fixed 170 s window put kappa
at 0.91-0.95, i.e. UNDER the low-side threshold it was being compared against.

rev3 therefore records, per cell and with no extra instrumentation:

    span_s, drain_s, drain_over_span        MEASURED (drain_s = duration - span)
    kappa_pred, drain_pred_s                REGISTERED before the run (plan)
    drain_model_ok                          drain_s <= 2.0 * drain_pred_s
    low_side_candidate                      registered: kappa_pred >= 0.97

`kappa_pred` is a CEILING on the reading, not a prediction of it.  A rung with
kappa_pred < 0.95 cannot pass the low-side test no matter how far below capacity
it sits, so only registered low-side candidates may supply R1's low side.
`drain_model_ok` is what keeps that certification honest: the ceiling is computed
from a step model, and if the measured drain blows past the model the cell's
certification is void (-> UNRESOLVED), rather than producing a confident label
from a wrong safety margin.

`drain_s` is NOT an independent estimate of kappa.  `(N/(N-1))*span/(span+drain_s)`
is ALGEBRAICALLY EQUAL to `achieved_over_realized`; dividing one by the other
gives 1 by construction.  It is recorded so the shortfall can be attributed, and
compared against the REGISTERED prediction -- never used to normalise the reading.

★D20 -- `--random-range-ratio 1.0` is a NECESSARY CONDITION for the offline
arrival replay, not merely for length exactness.  `benchmark/datasets/common.py:
56-64` draws `np.random.randint` twice per run before any interval is sampled;
only a degenerate range (rr = 1.0) consumes zero stream.  So this analyzer reads
`random_range_ratio` back out of the bench record and refuses the cell if it is
not 1.0.  (rev2's claim that "np.random has exactly two consumers" was true of
`bench_serving.py` as a FILE and false of the PATH.)

★The Ebar guard is kept, but rev2's claim about what it means is WITHDRAWN: Ebar
is computed from the replayed stream itself, so it cannot detect a case where the
replay stopped matching the run.  It only catches a cell run at a seed or
num-prompts other than the registered ones.  Detection of a broken replay is what
`range_ratio_ok` is for.

Everything here is CLIENT SIDE.  No telemetry is read, so the interval
attribution machinery this track keeps dying on (`CONSENSUS sec 3 item 120`)
cannot enter the judged quantity.

Usage:
  lambda0_analyze.py <bench.jsonl> --shape A|B --label NAME --num-prompts N
                     --seed S [--kappa-pred K --drain-pred-s L --low-side-candidate
                     0|1 --t-measure-s T] --out cell.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

EBAR_GUARD = (0.95, 1.05)     # 1/Ebar band; the plan's seed choice clears it 2x
DRAIN_MODEL_TOL = 2.0         # measured drain vs registered prediction
REQUIRED_RANGE_RATIO = 1.0    # D20: necessary for the arrival replay


def percentile(xs, q):
    """Linear-interpolated percentile; `q` in [0,1].  Empty -> nan.

    Copied verbatim from `c_capacity_analyze.py:47-59` so any number this stage
    prints beside a probe C number came from the same estimator.
    """
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def arrival_span(seed: int, num_prompts: int, rate: float) -> tuple:
    """Reproduce bench_serving's arrival window.  Returns (span_s, Ebar)."""
    if num_prompts < 2 or not rate or rate != rate or rate == float("inf"):
        return float("nan"), float("nan")
    np.random.seed(seed)
    iv = np.random.exponential(1.0 / rate, size=num_prompts - 1)
    return float(iv.sum()), float(iv.mean()) * rate


def analyze(bench_path, shape, label, num_prompts, seed, kappa_pred=None,
            drain_pred_s=None, low_side_candidate=False, t_measure_s=None):
    d = json.loads(Path(bench_path).read_text().strip().split("\n")[0])
    errs = d["errors"]
    ok = [i for i, e in enumerate(errs) if e == ""]
    ttfts = [d["ttfts"][i] for i in ok]
    itls = [d["itls"][i] for i in ok]
    flat_itls = [x for v in itls for x in v]
    dur = d["duration"]

    n_sent = num_prompts if num_prompts else len(errs)
    offered = d["request_rate"]
    achieved = d["request_throughput"]
    span, eb = arrival_span(seed, n_sent, offered)
    off_real = (n_sent - 1) / span if span == span and span else float("nan")
    drain = dur - span if span == span else float("nan")

    in_len, out_len = d["random_input_len"], d["random_output_len"]
    n_ok = len(ok)
    got_in, got_out = d.get("total_input_tokens"), d.get("total_output_tokens")
    rr = d.get("random_range_ratio")

    return {
        "stage": "LAMBDA0", "rev": 5,
        "prereg": "PREREG_LAMBDA0_REV5_2026-09-14.md",
        "shape": shape, "label": label,
        "bench_path": str(bench_path),
        "client_seed": seed,
        "num_prompts": n_sent,
        "completed": d["completed"],
        "n_errors": len(errs) - n_ok,
        "duration_s": dur,
        "offered_rate": offered,                        # NOMINAL (reported)
        "achieved_rate": achieved,
        "achieved_over_offered": (achieved / offered) if offered else float("nan"),
        "realized_offered_rate": off_real,              # PRIMARY denominator
        "achieved_over_realized": (achieved / off_real)
                                  if off_real == off_real and off_real else float("nan"),
        # ---- drain decomposition (measured).  See the module docstring: this
        # is an algebraic re-expression of the ratio above, never a normaliser.
        "span_s": span,
        "drain_s": drain,
        "drain_over_span": (drain / span) if span else float("nan"),
        # ---- registered ceiling (from the plan, fixed before the run)
        "t_measure_s": t_measure_s,
        "kappa_pred": kappa_pred,
        "drain_pred_s": drain_pred_s,
        "low_side_candidate": bool(low_side_candidate),
        "drain_model_tol": DRAIN_MODEL_TOL,
        "drain_model_ok": bool(
            drain_pred_s is None or (drain == drain
                                     and drain <= DRAIN_MODEL_TOL * drain_pred_s)),
        "headroom_pred": (None if kappa_pred is None else kappa_pred - 0.95),
        # ---- guards
        "ebar": eb,
        "inv_ebar": (1.0 / eb) if eb == eb and eb else float("nan"),
        "ebar_guard": list(EBAR_GUARD),
        "ebar_guard_ok": bool(eb == eb and EBAR_GUARD[0] <= 1.0 / eb <= EBAR_GUARD[1]),
        "random_range_ratio": rr,
        "range_ratio_ok": bool(rr is not None and abs(rr - REQUIRED_RANGE_RATIO) < 1e-12),
        # ---- length verification (D10)
        "random_input_len": in_len, "random_output_len": out_len,
        "total_input_tokens": got_in, "total_output_tokens": got_out,
        "expected_input_tokens": n_ok * in_len,
        "expected_output_tokens": n_ok * out_len,
        "input_len_exact": bool(got_in == n_ok * in_len),
        "output_len_exact": bool(got_out == n_ok * out_len),
        # ---- reported only
        "ttft_p50_s": percentile(ttfts, 0.50),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "ttft_p99_s": percentile(ttfts, 0.99),
        "ttft_max_s": max(ttfts) if ttfts else float("nan"),
        "itl_p50_s": percentile(flat_itls, 0.50),
        "itl_p95_s": percentile(flat_itls, 0.95),
        "concurrency_reported": d.get("concurrency"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_path")
    ap.add_argument("--shape", required=True, choices=("A", "B"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--kappa-pred", type=float, default=None)
    ap.add_argument("--drain-pred-s", type=float, default=None)
    ap.add_argument("--low-side-candidate", type=int, default=0)
    ap.add_argument("--t-measure-s", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = analyze(a.bench_path, a.shape, a.label, a.num_prompts, a.seed,
                  kappa_pred=a.kappa_pred, drain_pred_s=a.drain_pred_s,
                  low_side_candidate=bool(a.low_side_candidate),
                  t_measure_s=a.t_measure_s)
    text = json.dumps(res, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
