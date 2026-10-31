#!/usr/bin/env python3
"""Phase A5 of PLAN_stock_baseline_metrics_2026-11-01: a-priori r_ser grid for stock SGLang on Nemotron-Nano-9B-v2.

  r_ser_hat(L, bs, ctx) = T2(L; 108) / T1(bs, ctx; 108v)

Inputs are the vanilla-engine (`108v`, P = 108, no PD-mux) cost-term fits of the 2026-10-26 campaign
(cap_char/RESULT_costterms_2026-10-26.md §2-2 F1 and §3 T2, J-A1 round 1; audit r1: T1 F1 at P = 108
CONFIRMED(scoped), T2 q label CONFIRMED):
  T1  F1: a + b*bs + c*bs*ctx            a = 13.00 ms, b = 0.2689 ms/req, c = 9.73e-6 ms/(req*token)
  T2     : u + v*L + w*L^2                u = 11.83 ms, v = 0.1076 ms/token, w = 1.880e-7 ms/token^2

Caveats carried with every number (printed in the output header):
  - T2 is client TTFT (includes host/queue overhead u); r_ser measured from HOLB/decode_probe is GPU time,
    so r_ser_hat is biased upward at small L.  L < 1k is outside the T2 grid (1k-64k): extrapolation.
  - With chunked_prefill_size C < L the prefill runs as back-to-back chunks with no decode in between
    (SGLang default scheduler), so the stall is still ~T2(L); per-chunk overhead is not modelled.
  - Forecast only. To be sealed in the Phase B PREREG before any stock-SGLang Nano run; no verdicts.
"""
import csv
import os

A, B, C = 13.00, 0.2689, 9.73e-6
U, V, W = 11.83, 0.1076, 1.880e-7
LS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
BSS = [1, 8, 32, 48]
CTXS = [2048, 8192, 16384]


def t1(bs, ctx):
    return A + B * bs + C * bs * ctx


def t2(L):
    return U + V * L + W * L * L


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "R_SER_FORECAST_NANO.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["L", "bs", "ctx", "T2_ms", "T1_ms", "r_ser_hat", "note"])
        for L in LS:
            for bs in BSS:
                for ctx in CTXS:
                    note = "L<1k extrapolated" if L < 1024 else ""
                    w.writerow([L, bs, ctx, f"{t2(L):.2f}", f"{t1(bs, ctx):.2f}", f"{t2(L) / t1(bs, ctx):.2f}", note])
    print(__doc__.split("Caveats")[0].strip().splitlines()[0])
    print(f"{'L':>6} " + " ".join(f"bs{bs}/ctx{ctx // 1024}k".rjust(13) for bs in BSS for ctx in CTXS[:1] + CTXS[2:]))
    for L in LS:
        print(f"{L:>6} " + " ".join(f"{t2(L) / t1(bs, ctx):13.1f}" for bs in BSS for ctx in CTXS[:1] + CTXS[2:]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
