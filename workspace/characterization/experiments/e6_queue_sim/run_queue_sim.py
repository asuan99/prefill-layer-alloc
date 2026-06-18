"""Queue-simulator driver: sweep arrival rate λ × SM policy, report SLO metrics.

GPU-free. Driven by a measured single-chunk E5 LUT (see latency_model.py). Answers:
does dynamic_protect (decode reserved its E3 floor) beat co_schedule on decode
tail-latency / goodput under sustained load — the regime the microbench can't show?

  python -m experiments.e6_queue_sim.run_queue_sim \
      --lut results_v2/e5_dp/serving_coexec_full_zamba2_2.7b_a100_sxm4_80gb.csv \
      --pf-layer attn --dec-layer ssm --slo-ms 2.0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
for _p in (_CHAR, os.path.abspath(os.path.join(_here, "..", "..", ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.e6_queue_sim.latency_model import LatencyModel
from experiments.e6_queue_sim.workload import gen_requests
from experiments.e6_queue_sim.simulator import simulate


def _pct(xs, q):
    import numpy as np
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def metrics(reqs, makespan, slo_ms):
    ttfts = [r.ttft for r in reqs if r.ttft is not None]
    itls = [v for r in reqs for v in r.itls]
    toks = sum(len(r.itls) for r in reqs)
    return {
        "n_req": len(reqs), "tokens": toks,
        "ttft_p50": round(_pct(ttfts, 50), 3), "ttft_p99": round(_pct(ttfts, 99), 3),
        "itl_p50": round(_pct(itls, 50), 4), "itl_p99": round(_pct(itls, 99), 4),
        "throughput_tok_s": round(toks / makespan * 1000.0, 1) if makespan else 0.0,
        "slo_attain": round(sum(1 for v in itls if v <= slo_ms) / len(itls), 4) if itls else 0.0,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid serving queue simulator (SLO vs policy)")
    p.add_argument("--lut", required=True, help="E5 full CSV (single-chunk LUT preferred)")
    p.add_argument("--policies", nargs="+", default=["co_schedule", "dynamic_protect", "static"])
    p.add_argument("--lambdas", nargs="+", type=float,
                   default=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0], help="arrival rate (req/ms)")
    p.add_argument("--pf-layer", default="ssm", choices=["ssm", "attn"])
    p.add_argument("--dec-layer", default="ssm", choices=["ssm", "attn"])
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--n-requests", type=int, default=500)
    p.add_argument("--prompt-lens", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    p.add_argument("--output-lens", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--max-batch", type=int, default=512)
    p.add_argument("--slo-ms", type=float, default=2.0, help="ITL SLO target (ms)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path(_CHAR) / "results_v2" / "e6")
    return p.parse_args()


def main():
    a = parse_args()
    lm = LatencyModel(a.lut)
    model = os.path.basename(a.lut).replace("serving_coexec_full_", "").replace("serving_coexec_", "").replace(".csv", "")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"=== queue sim: {model}  pf={a.pf_layer}×dec={a.dec_layer} ctx={a.ctx}  SLO(ITL)≤{a.slo_ms}ms ===")
    print(f"  backends in LUT: {lm.backends}")
    hdr = f"{'policy':16} {'λ(req/ms)':>9} {'TTFT_p99':>9} {'ITL_p50':>8} {'ITL_p99':>8} {'tok/s':>8} {'SLO_attain':>10}"
    print(hdr)
    for lam in a.lambdas:
        for pol in a.policies:
            reqs = gen_requests(a.n_requests, lam, a.prompt_lens, a.output_lens, a.chunk, a.seed)
            reqs, st = simulate(reqs, pol, lm, a.pf_layer, a.dec_layer, a.ctx, a.max_batch)
            m = metrics(reqs, st["makespan_ms"], a.slo_ms)
            row = {"model": model, "policy": pol, "lambda_req_ms": lam,
                   "pf_layer": a.pf_layer, "dec_layer": a.dec_layer, "ctx": a.ctx,
                   "slo_ms": a.slo_ms, **m, "notes": ";".join(st["notes"])}
            rows.append(row)
            print(f"{pol:16} {lam:9.3f} {m['ttft_p99']:9.2f} {m['itl_p50']:8.3f} "
                  f"{m['itl_p99']:8.3f} {m['throughput_tok_s']:8.1f} {m['slo_attain']:10.3f}")
        print("  " + "-" * 70)
    out = a.out_dir / f"queue_sim_{model}_{a.pf_layer}x{a.dec_layer}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")
    # crossover hint: where does dynamic_protect SLO-attain exceed co_schedule?
    bylam = {}
    for r in rows:
        bylam.setdefault(r["lambda_req_ms"], {})[r["policy"]] = r["slo_attain"]
    cross = [lam for lam, d in sorted(bylam.items())
             if "dynamic_protect" in d and "co_schedule" in d and d["dynamic_protect"] > d["co_schedule"] + 1e-6]
    print(f"  → dynamic_protect beats co_schedule on SLO_attain at λ={cross}" if cross
          else "  → dynamic_protect never beats co_schedule on SLO_attain in this sweep")


if __name__ == "__main__":
    main()
