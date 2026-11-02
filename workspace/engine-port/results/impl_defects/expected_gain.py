#!/usr/bin/env python3
"""Expected latency effect of the loop changes E1/E2/E3 (+ budget 512, E4) -- derived, not measured (2026-11-02).

Answers "if we change the loop, what gets faster, where, and by how much" in milliseconds, so the axes-map §6b can show a
benefit (or a cost) next to each change instead of only a changed flow.

Inputs (all existing):
  * prefill time S_pre(L; P) and decode step T_dec(D, b, c): scripts/costmodel/dstar_model.py (T1/T2/T2' cost terms,
    analytic-v1 prior) -- same functions the D* proposal uses.
  * B2 frequency/duration: phase0_c1_c2.md table C2-B (N3 Jobs 3-4, N2-EA) [measured, upper bound].
  * B1 bounds: phase0_c1_c2.md table C1-3 (HE0, KISTI) [derived upper bound].
  * span rule of the current loop: forward_count = max(1, 65536 // L) layers (multiplexing_mixin.py:1416-1424).
Every output number is [derived]. Run: python3 expected_gain.py --out expected_gain.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "costmodel"))
import dstar_model as dm  # noqa: E402

N_LAYERS = 56          # Nemotron-Nano-9B-v2
BUDGET = 65536
RHO = 1.16
D = 44                 # registered static default; P = 64
P = dm.TOTAL_SM - D

# [measured] phase0_c1_c2.md table C2-B
B2 = {"N3": {"freq_of_spans": (0.448, 0.458), "dur_ms_p50": (849, 888), "dur_ms_p95": (1549, 1559)},
      "N2-EA": {"freq_of_spans": (0.029, 0.029), "dur_ms_max": 0.61}}
# [derived upper bound] phase0_c1_c2.md table C1-3 (HE0, KISTI Zamba2-2.7B -- other substrate, bound only)
B1 = {"bind+GATE": {"W_max_ms": 489, "HI_req_share_upper": 0.0072},
      "slo": {"W_max_ms": 538, "HI_req_share_upper": (0.102, 0.112)}}


def chunks(L):
    per = max(1, BUDGET // L)
    return math.ceil(N_LAYERS / per), per


def e2_b2(L=8192):
    """E2 on B2: after decode empties, only <= k issued spans stay on P SMs; the rest runs on 108 SMs."""
    W = dm.s_pre(L, D)
    n, per = chunks(L)
    span = W * per / N_LAYERS
    out = {"L": L, "W_ms": W, "n_spans": n, "span_ms": span, "cases": []}
    for label, R in (("p50", sum(B2["N3"]["dur_ms_p50"]) / 2), ("p95", sum(B2["N3"]["dur_ms_p95"]) / 2)):
        for k in (1, 2):
            moved = max(0.0, R - k * span)                    # P-SM time of work that E2 keeps unissued
            saved = moved * (1 - P / dm.TOTAL_SM)              # it now runs P/108 as long
            out["cases"].append({"B2_remaining_ms": R, "pct": label, "k": k, "saved_ms": saved,
                                 "saved_share_of_prefill": saved / W})
    # cost while decode is busy (accumulation regime, span > step): with k=1 the next span is issued only at the next
    # loop iteration after the running one ends -> expected idle ~ step/2 per span; with k>=2 one span is always queued.
    step = RHO * dm.t_dec(D, 8, L + 64)
    out["busy_cost_k1_ms"] = n * step / 2
    out["busy_cost_k2_ms"] = 0.0
    out["busy_step_ms"] = step
    return out


def e1_time(L, step_ms, factor=1.0):
    """E1 'time' policy (average layer cost): span = largest layer count with cumulative <= factor*step (>= 1).
    One span issued per loop iteration, iterations paced by the decode step. Baseline = budget rule."""
    W = dm.s_pre(L, D)
    t = W / N_LAYERS
    k = max(1, min(N_LAYERS, int(factor * step_ms // t)))
    n = math.ceil(N_LAYERS / k)
    t_e1 = max(W, (n - 1) * step_ms + k * t)               # GPU-bound if spans exceed the step, else step-paced
    nb, perb = chunks(L)
    t_base = max(W, (nb - 1) * step_ms + perb * t) if nb > 1 else W
    return {"L": L, "step_ms": step_ms, "factor": factor, "W_ms": W, "layer_ms": t, "layers_per_span": k,
            "iterations": n, "prefill_ms_E1": t_e1, "prefill_ms_budget": t_base, "ratio": t_e1 / t_base}


def budget512(L=2048, step_ms=None):
    step_ms = step_ms or RHO * dm.t_dec(D, 8, 2048 + 64)
    W = dm.s_pre(L, D)
    per = max(1, 512 // L)
    n = math.ceil(N_LAYERS / per)
    t_layer = W / N_LAYERS
    t = max(W, (n - 1) * step_ms + per * t_layer)
    return {"L": L, "step_ms": step_ms, "W_ms": W, "prefill_ms": t, "ratio": t / W}


def p_decode_empty(lam, R, ctx):
    """How often the decode batch is empty: decode occupancy ~ Poisson(b_mean) (M/G/inf), b_mean = lam * E[R] * T_dec
    (fixed point on the 108-SM decode step). B2 -- and with it every E2 gain -- needs decode to be empty while a prefill
    is in flight, so its weight scales with ~exp(-b_mean)."""
    T = dm.t_dec("108u", 1, ctx)
    b = 0.0
    for _ in range(200):
        b = 0.5 * b + 0.5 * lam * R * T / 1e3
        T = dm.t_dec("108u", max(b, 1.0), ctx)
    return {"lam": lam, "ER": R, "ctx": ctx, "b_mean": b, "p_empty": math.exp(-b)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    steps = {f"b{b}_c{c}": RHO * dm.t_dec(D, b, c) for b in (1, 8, 32) for c in (2048, 8192)}
    res = {
        "setup": {"D": D, "P": P, "rho": RHO, "model": "Nemotron-Nano-9B-v2", "substrate": "VESSL A100 (cost terms)",
                  "co_located_decode_step_ms": steps},
        "E2_on_B2": e2_b2(8192),
        "E1_time": [e1_time(L, s, f) for f in (1.0, 5.0) for L in (256, 2048, 8192, 32768) for s in (20.0, 40.0)],
        "budget512_2k": budget512(2048),
        "E4_hol_wait_upper_ms": {str(L): dm.s_pre(L, D) for L in (8192, 16384)},
        "B2_source": B2, "B1_source": B1,
        "p_decode_empty": {"N3_shapeB_r0.10": p_decode_empty(0.10, 64, 8224),
                           "N3_shapeB_r0.77": p_decode_empty(0.77, 64, 8224),
                           "W1_LO_2.7356": p_decode_empty(2.7356, 286, 373),
                           "W1_1.0": p_decode_empty(1.0, 286, 373),
                           "W3_lamstar_0.3375": p_decode_empty(0.3375, 1007, 7609)},
    }
    json.dump(res, open(a.out, "w"), indent=1)
    e2 = res["E2_on_B2"]
    print(f"S_pre(8k;P64)={e2['W_ms']:.0f} ms, spans {e2['n_spans']} x {e2['span_ms']:.0f} ms")
    for c in e2["cases"]:
        print(f"  E2 k={c['k']} B2 {c['pct']} {c['B2_remaining_ms']:.0f} ms -> saved {c['saved_ms']:.0f} ms "
              f"({100*c['saved_share_of_prefill']:.0f}% of prefill)")
    print(f"  E2 busy cost: k=1 ~{e2['busy_cost_k1_ms']:.0f} ms/prefill (step {e2['busy_step_ms']:.1f}), k=2 0")
    print("  E4 HOL wait upper (remaining long prefill):", {k: round(v) for k, v in res['E4_hol_wait_upper_ms'].items()})
    for k, v in res["p_decode_empty"].items():
        print(f"  P(decode empty) {k}: b_mean {v['b_mean']:.2f} -> {v['p_empty']:.3g}")
    print("co-located decode step (D44, rho 1.16):", {k: round(v, 1) for k, v in steps.items()})
    for r in res["E1_time"]:
        print(f"  E1 f={r['factor']:.0f} L={r['L']:>5} step={r['step_ms']:.0f}: layers/span {r['layers_per_span']:>2} "
              f"iters {r['iterations']:>2}  prefill {r['prefill_ms_budget']:.0f} -> {r['prefill_ms_E1']:.0f} ms (x{r['ratio']:.2f})")
    b = res["budget512_2k"]
    print(f"  budget 512, 2k: step {b['step_ms']:.1f} ms, prefill {b['W_ms']:.0f} -> {b['prefill_ms']:.0f} ms (x{b['ratio']:.2f})")


if __name__ == "__main__":
    main()
