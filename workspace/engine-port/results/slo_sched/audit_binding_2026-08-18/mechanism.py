#!/usr/bin/env python3
"""Mechanism probes for reading R-2/R-3.

(1) Is the residual HI TTFT loss the open-loop overload backlog (a capacity
    deficit) rather than a split-positioning constraint?  -> TTFT vs launch index.
(2) Does the loss decomposition (TTFT-only / ITL-only / both) survive moving the
    ITL threshold inside its own pre-registered +-10% band?
(3) Does ttft_pass_pct track achieved throughput across arms (capacity proxy)?
Read-only.
"""
import json, math, statistics, sys
from pathlib import Path

ROOT = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
SLO_DIR = ROOT / "results/slo_sched"
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile

ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
JOBS = [("blk1", "884336"), ("blk2", "884410"), ("blk3", "884411"), ("blk4", "884412")]


def per_round(path):
    """Return per-round lists so launch index is meaningful."""
    rounds = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = [1000.0 * float(v) for v in (rec.get("ttfts") or [])]
            i = rec.get("itls") or []
            i95 = [percentile([1000.0 * float(x) for x in toks], .95) if toks else math.inf
                   for toks in i]
            rounds.append({"ttft": t, "itl95": i95, "duration": float(rec["duration"])})
    return rounds


def main():
    out = {}

    # (1) TTFT vs launch-index decile, pooled over 4 blocks x 3 rounds
    dec = {}
    for arm in ARMS:
        buckets = [[] for _ in range(10)]
        for blk, job in JOBS:
            for r in per_round(SLO_DIR / f"g16_{blk}_{arm}_boot1_{job}_HI.jsonl"):
                n = len(r["ttft"])
                for k, v in enumerate(r["ttft"]):
                    buckets[min(9, 10 * k // n)].append(v)
        dec[arm] = {"median_ttft_ms_by_decile": [statistics.median(b) for b in buckets],
                    "pass_pct_by_decile": [100 * sum(v <= 3000 for v in b) / len(b)
                                           for b in buckets]}
    out["ttft_by_launch_decile_HI"] = dec

    # (2) loss decomposition vs ITL threshold (TTFT SLO fixed at 3000)
    decomp = {}
    for islo in (54.0, 57.0, 58.5, 60.0, 61.5, 63.0, 66.0):
        per_arm = {}
        for arm in ARMS:
            A = B = C = D = 0
            for blk, job in JOBS:
                reqs, _ = load_bench_serving_rounds(
                    SLO_DIR / f"g16_{blk}_{arm}_boot1_{job}_HI.jsonl")
                for r in reqs:
                    tp = r.ttft_ms <= 3000.0
                    ip = (percentile(r.token_itl_ms, .95) <= islo) if r.token_itl_ms else False
                    if tp and ip: A += 1
                    elif tp:      B += 1
                    elif ip:      C += 1
                    else:         D += 1
            n = A + B + C + D
            per_arm[arm] = {"joint_pct": 100 * A / n,
                            "loss_TTFTonly_pp": 100 * C / n,
                            "loss_ITLonly_pp": 100 * B / n,
                            "loss_both_pp": 100 * D / n,
                            "share_of_loss_TTFTonly": C / (n - A) if n > A else float("nan")}
        decomp[f"itl{islo:g}"] = per_arm
    out["loss_decomposition_vs_itl_threshold_HI"] = decomp

    # (3) throughput vs ttft_pass across arms (block-paired)
    thr, tp = {}, {}
    for arm in ARMS:
        th, tt = [], []
        for blk, job in JOBS:
            reqs, dur = load_bench_serving_rounds(
                SLO_DIR / f"g16_{blk}_{arm}_boot1_{job}_HI.jsonl")
            th.append(len(reqs) / dur)
            tt.append(100 * sum(r.ttft_ms <= 3000 for r in reqs) / len(reqs))
        thr[arm], tp[arm] = statistics.fmean(th), statistics.fmean(tt)
    xs = [thr[a] for a in ARMS]; ys = [tp[a] for a in ARMS]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    out["throughput_vs_ttftpass_HI"] = {
        "throughput_req_s": thr, "ttft_pass_pct": tp,
        "pearson_r_all7": num / den,
    }
    xs6 = [thr[a] for a in ARMS[:-1]]; ys6 = [tp[a] for a in ARMS[:-1]]
    mx6, my6 = statistics.fmean(xs6), statistics.fmean(ys6)
    n6 = sum((x - mx6) * (y - my6) for x, y in zip(xs6, ys6))
    d6 = math.sqrt(sum((x - mx6) ** 2 for x in xs6) * sum((y - my6) ** 2 for y in ys6))
    out["throughput_vs_ttftpass_HI"]["pearson_r_excl_d74"] = n6 / d6

    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

# --- appended: ITL vs launch decile (why the two failures are ANTI-correlated) --
def itl_by_decile():
    res = {}
    for arm in ARMS:
        buckets = [[] for _ in range(10)]
        for blk, job in JOBS:
            for r in per_round(SLO_DIR / f"g16_{blk}_{arm}_boot1_{job}_HI.jsonl"):
                n = len(r["itl95"])
                for k, v in enumerate(r["itl95"]):
                    buckets[min(9, 10 * k // n)].append(v)
        res[arm] = {
            "median_itl95_by_decile": [statistics.median(b) for b in buckets],
            "itl_pass_pct_by_decile": [100 * sum(v <= 60 for v in b) / len(b) for b in buckets],
        }
    return res
