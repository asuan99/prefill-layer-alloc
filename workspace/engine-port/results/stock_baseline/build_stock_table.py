#!/usr/bin/env python3
"""Phase A1/A2/A4/A6-history of PLAN_stock_baseline_metrics_2026-11-01: one metrics table for the
existing stock-serving (and same-campaign PD-mux contrast) measurements.

Sources (all local, untracked raw):
  P1-opint (KISTI, cudagraph ON)     p1_opint/p1op_<model>_<plain|agnostic>_rep<k>_<job>.jsonl
  M3 J1 N1 (VESSL, Nano-9B-v2-Base)  multigpu/manual_retrieval/pdmux-multigpu-2106f42-*/.../cells/*
  vLLM 0.22.1 (KISTI, history only)  ../../../vllm_bench/results/real_vllm.csv (aggregates only)

Writes STOCK_METRICS_TABLE.csv (one row per round and one pooled row per cell) and
REPRO_P1OPINT.txt (A2: mean-of-reps vs P1_OPINT_RESULT_2026-08-05.md §3.1). Exit 1 if A2 fails.
"""
import csv
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stock_metrics as sm  # noqa: E402

RES = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(RES, "..", "..", ".."))

COLS = ["source", "substrate", "engine", "arm", "arm_kind", "model", "workload", "rate_or_rung", "round",
        "n_rounds", "n_sent", "n_ok", "duration_s", "req_per_s", "out_tok_per_s", "total_tok_per_s",
        "e2e_source", "e2e_p50_s", "e2e_p95_s", "e2e_p99_s", "ttft_p50_s", "ttft_p95_s", "ttft_p99_s",
        "tpot_p50_s", "tpot_p95_s", "tpot_p99_s", "itl_req_p95_median_s", "slo", "goodput_req_per_s",
        "r_kind", "cite_status"]

# P1-opint §3.1 (TTFT s, TPOT ms; mean over 5 reps). Used only for the A2 reproduction check.
P1_DOC = {
    ("zamba2-27b", "plain"): {2: (1.866, .164, .365, .752, 12.78, 23.78), 3: (2.958, .211, .543, .878, 22.10, 43.17),
                              4: (3.970, .306, 1.366, 2.657, 48.51, 81.18), 6: (4.283, 2.127, 5.580, 6.084, 91.32, 112.26)},
    ("zamba2-27b", "agnostic"): {2: (1.865, .230, .576, .970, 12.53, 19.30), 3: (2.954, .385, 1.121, 1.421, 17.71, 29.26),
                                 4: (3.961, .743, 2.050, 2.842, 30.95, 49.11), 6: (4.515, 2.335, 4.488, 4.944, 48.13, 66.66)},
    ("granite-40-h-micro-base", "plain"): {2: (1.968, .123, .222, .298, 8.44, 13.21), 3: (2.933, .126, .284, .350, 9.90, 17.86),
                                           4: (3.975, .153, .320, .374, 12.54, 20.80), 6: (5.702, .218, .587, .729, 22.94, 40.81)},
    ("granite-40-h-micro-base", "agnostic"): {2: (1.972, .168, .375, .503, 8.73, 11.19), 3: (2.924, .177, .503, .634, 9.73, 12.89),
                                              4: (3.943, .243, .686, 1.047, 11.25, 13.80), 6: (5.557, .691, 1.465, 1.754, 13.27, 14.97)},
}
P1_CLIFF = {("zamba2-27b", "plain", 4), ("zamba2-27b", "plain", 6), ("zamba2-27b", "agnostic", 6)}


def row(base, m, rnd):
    out = dict(base)
    out.update({k: m.get(k) for k in COLS if k in m})
    out["round"] = rnd
    return out


def p1_opint(rows, repro):
    d = os.path.join(RES, "p1_opint")
    ok_all = True
    for model, backend in (("zamba2-27b", "triton"), ("granite-40-h-micro-base", "flashinfer")):
        for arm in ("plain", "agnostic"):
            per_rate = {}
            for rep in range(1, 6):
                paths = sorted(glob.glob(os.path.join(d, f"p1op_{model}_{arm}_rep{rep}_*.jsonl")))
                if not paths:
                    continue
                for ln in open(paths[-1]):
                    if not ln.strip():
                        continue
                    br = json.loads(ln)
                    per_rate.setdefault(int(br["request_rate"]), []).append((rep, sm.from_bench_serving_row(br)))
            for rate, reps in sorted(per_rate.items()):
                cliff = (model, arm, rate) in P1_CLIFF
                base = {"source": "p1_opint", "substrate": "KISTI-A100-PCIe?/Neuron", "engine": f"sglang-0.5.10/{backend}",
                        "arm": arm, "arm_kind": "stock_default" if arm == "plain" else "pdmux_agnostic",
                        "model": model, "workload": "random-ids in2000/out96, 120 req", "rate_or_rung": rate,
                        "slo": "TTFT<=3s & ITL_p95(req)<=60ms", "r_kind": "",
                        "cite_status": ("KISTI; CONSENSUS row1 steady-state only" + ("; CLIFF cell - do not cite" if cliff else "")
                                        + ("; Zamba excluded (decision 46): cite only" if model.startswith("zamba") else "")
                                        + "; fused untuned")}
                per_rep = []
                for rep, (dur, reqs) in reps:
                    m = sm.metrics([(dur, reqs)], 3.0, 0.060)
                    per_rep.append(m)
                    rows.append(row(base, m, f"rep{rep}"))
                pooled = sm.metrics([x for _, x in reps], 3.0, 0.060)
                rows.append(row(base, pooled, "pooled"))
                # A2: mean of per-rep values, compared with the doc table
                mean = lambda k: statistics.mean(m[k] for m in per_rep)
                got = (mean("req_per_s"), mean("ttft_p50_s"), mean("ttft_p95_s"), mean("ttft_p99_s"),
                       mean("tpot_p50_s") * 1e3, mean("tpot_p95_s") * 1e3)
                exp = P1_DOC[(model, arm)][rate]
                # doc rounds to 3 decimals (TTFT s, req/s) and 2 (TPOT ms); req/s in the doc is bench_serving's own
                # request_throughput (completed/duration) which equals ours when all requests complete.
                tol = (0.002, 0.0015, 0.0015, 0.0015, 0.015, 0.015)
                bad = [i for i, (g, e, t) in enumerate(zip(got, exp, tol)) if abs(g - e) > t]
                ok_all &= not bad
                repro.append(f"{model:24s} {arm:9s} r{rate}  got " + " ".join(f"{g:.3f}" for g in got)
                             + "  doc " + " ".join(f"{e:.3f}" for e in exp) + ("  OK" if not bad else f"  MISMATCH idx {bad}"))
    return ok_all


def m3_j1(rows):
    pat = os.path.join(RES, "multigpu", "manual_retrieval", "pdmux-multigpu-2106f42-*", "results", "multigpu",
                       "m3_pdmux-multigpu-2106f42-*", "cells", "m3_N1_*")
    groups = {}
    for cdir in sorted(glob.glob(pat)):
        meta, dur, reqs = sm.from_m3_cell(cdir)
        key = (meta["arm"], meta["shape"], meta["rung"] + ("x" if meta.get("ext") else ""))
        groups.setdefault(key, []).append((meta, dur, reqs))
    for (arm, shape, rung), cells in sorted(groups.items()):
        kind = {"fused": "stock_default", "fused_sym": "stock_sym"}.get(arm, "pdmux_" + arm)
        base = {"source": "m3_j1_N1", "substrate": "VESSL-A100-SXM", "engine": "sglang-0.5.10/flashinfer",
                "arm": arm, "arm_kind": kind, "model": "Nemotron-Nano-9B-v2-Base",
                "workload": {"A": "shape A in256/out512", "B": "shape B in8192/out64", "X": "shape X arXiv 16k"}[shape],
                "rate_or_rung": rung, "slo": "", "r_kind": "",
                "cite_status": "N=1 Job, underpowered, descriptive" + ("; HI TTFT do not cite (C:8792)" if "HI" in rung else "")
                + ("; X: X_CORRECTNESS_FAIL ratios undefined" if shape == "X" else "")}
        for meta, dur, reqs in cells:
            rows.append(row(base, sm.metrics([(dur, reqs)]), f"s{meta['seed']}"))
        rows.append(row(base, sm.metrics([(d, r) for _, d, r in cells]), "pooled"))


def vllm_history(rows):
    p = os.path.join(REPO, "vllm_bench", "results", "real_vllm.csv")
    for r in csv.DictReader(open(p)):
        f = lambda k: float(r[k]) / 1e3 if r[k] not in ("", "—") else None
        rows.append({"source": "vllm_0.22.1_validation", "substrate": "KISTI", "engine": "vllm-0.22.1/FLASH_ATTN",
                     "arm": "vllm_stock", "arm_kind": "stock_default", "model": r["model"],
                     "workload": "random in2048/out128, 120 req", "rate_or_rung": r["request_rate"], "round": "single",
                     "n_rounds": 1, "req_per_s": float(r["req_thru"]), "out_tok_per_s": float(r["out_thru"]),
                     "ttft_p50_s": f("ttft_med"), "ttft_p99_s": f("ttft_p99"), "tpot_p50_s": f("tpot_med"),
                     "tpot_p99_s": f("tpot_p99"), "e2e_source": "none(aggregates only; med not p50-interp)",
                     "cite_status": "HISTORY ONLY: deprecated/ basis, n=1, Zamba excluded, no Nano - not a comparison basis"})


def main():
    rows, repro = [], []
    ok = p1_opint(rows, repro)
    m3_j1(rows)
    vllm_history(rows)
    with open(os.path.join(HERE, "STOCK_METRICS_TABLE.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items() if k in COLS})
    with open(os.path.join(HERE, "REPRO_P1OPINT.txt"), "w") as fh:
        fh.write("A2 P1-opint reproduction: mean over reps of (req/s, TTFT p50/p95/p99 s, TPOT p50/p95 ms)\n")
        fh.write("\n".join(repro) + "\n" + ("ALL OK\n" if ok else "MISMATCH PRESENT\n"))
    print(f"rows={len(rows)} A2={'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
