#!/usr/bin/env python3
"""Descriptive statistics of the 9 recovered boots of E2 VESSL job
pdmux-e2-sticky-vessl-e6f424e-20261004135228 (CPU only, GPU 0).

NOT A LABEL.  e2_label.py is deliberately NOT imported or called: the job stopped
at boot 10/20 (ABORT_P3), and the E2 registration (D23) forbids R1-R5 labels on a
non-completed job and any subset relabelling.  No CI / bootstrap is computed.

Estimators: e2_realized_mix.compute (imported, unmodified) re-run with the
cell's own probe_end, and cross-checked against the in-job cell_*.json.
Goodput predicate: pdmux_eval.analyze.RequestResult.passes (TTFT<=3000 ms AND
per-request token-ITL p95<=50 ms), step function only (E2C-6).
Output: JSON to argv[1].
"""
import csv, datetime as dt, json, os, re, sys
from collections import Counter

ENG = "/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port"
sys.path.insert(0, f"{ENG}/benchmarks")
sys.path.insert(0, f"{ENG}/results/r2_eval/e2_sticky_prereg")
from pdmux_eval.analyze import (percentile, summarize_requests, load_bench_serving_rounds,  # noqa: E402
                                request_tpot_percentiles, controller_summary)
import e2_realized_mix as mix  # noqa: E402

JOB = "pdmux-e2-sticky-vessl-e6f424e-20261004135228"
D = f"{ENG}/results/r2_eval/e2_sticky_vessl/e2v_{JOB}"
META = f"{ENG}/results/e2_sticky_vessl/vessl/{JOB}/meta"
L0 = f"{ENG}/results/r2_eval/lambda0_vessl/lam0v_pdmux-lambda0-vessl-1d47d3e-20261004084945"
L0J = f"{ENG}/results/r2_eval/lambda0_vessl/analysis_2026-10-04/lam0v_descriptive.json"
TTFT_SLO, ITL_SLO = 3000.0, 50.0
SEEDS = ["4386", "4162", "251", "2630"]
BOOTS = [f"a_r4_{a}_{s}" for s in SEEDS for a in ("OFF", "ON")] + ["a_r2_OFF_4386"]
L0_BY_SEED = {"4386": "a_r4", "4162": "a_r4_s2", "251": "a_r4_s3", "2630": "a_r4_s4"}


def req_stats(bench_path):
    reqs, dur = load_bench_serving_rounds(bench_path)
    s = summarize_requests(reqs, 0.0, dur, TTFT_SLO, ITL_SLO)
    rec = json.loads(open(bench_path).readline())
    p95s = [percentile(r.token_itl_ms, 0.95) for r in reqs]
    ttfts = [r.ttft_ms for r in reqs]
    ttft_ok = sum(t <= TTFT_SLO for t in ttfts)
    itl_ok = sum(p <= ITL_SLO for p in p95s)
    good = sum(r.passes(TTFT_SLO, ITL_SLO) for r in reqs)
    # cliff proximity (gate 4/6): how many requests sit within +-5% of each threshold
    near_itl = sum(0.95 * ITL_SLO <= p <= 1.05 * ITL_SLO for p in p95s)
    near_ttft = sum(0.95 * TTFT_SLO <= t <= 1.05 * TTFT_SLO for t in ttfts)
    return {
        "completed": rec.get("completed"), "n_requests_loaded": len(reqs), "duration_s": dur,
        "achieved_req_s": rec.get("completed") / dur, "bench_request_throughput": rec.get("request_throughput"),
        "n_errors": sum(1 for e in rec.get("errors", []) if e),
        "ttft_ms": {k: s[f"ttft_p{k}_ms"] for k in (50, 95, 99)} | {"min": min(ttfts), "max": max(ttfts)},
        "req_itl_p95_ms": {"p50": percentile(p95s, 0.50), "p95": percentile(p95s, 0.95),
                           "min": min(p95s), "max": max(p95s)},
        "pooled_token_itl_ms": {k: s[f"token_itl_p{k}_ms"] for k in (50, 95, 99)},
        "tpot": request_tpot_percentiles(reqs),
        "good": good, "ttft_ok": ttft_ok, "itl_ok": itl_ok,
        "slo_goodput_req_s": s["slo_goodput_req_s"], "good_frac": good / len(reqs),
        "legacy_mean_itl_goodput_req_s": s["legacy_mean_itl_goodput_req_s"],
        "near_itl_slo_pm5pct": near_itl, "near_ttft_slo_pm5pct": near_ttft,
    }


def green(path):
    g = json.load(open(path))
    out = {}
    for grp in g["groups"]:
        sm = lambda x: None if x.get("green_ctx_is_null") else x["green_sm"]["smCount"]
        out[grp["stream_index"]] = {"target": [grp["target_prefill_sm"], grp["target_decode_sm"]],
                                    "green": [sm(grp["prefill"]), sm(grp["decode"])],
                                    "sticky": grp.get("is_sticky_target")}
    return {"groups": out, "sticky_target_index": g.get("sticky_target_index")}


def cpu(path):
    blocks, cur = [], None
    for line in open(path):
        if line.startswith("#"):
            cur = {"hdr": line.strip()}; blocks.append(cur)
        else:
            k, v = line.split(); cur[k] = int(v)
    a, b = blocks
    return {k: b[k] - a[k] for k in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec")} | {
        "bench_s_by_epochs": int(b["hdr"].split("=")[1]) - int(a["hdr"].split("=")[1])}


def tel_scan(path):
    mx, n, cnt_obs = 0, 0, 0
    events = []
    for line in open(path):
        m = re.search(r'"dropped_events":(\d+)', line)
        if m:
            mx = max(mx, int(m.group(1))); n += 1
        if '"runtime_snapshot"' in line and '"phase":"benchmark"' in line:
            r = json.loads(line)
            events.append({k: r.get(k) for k in ("event", "phase", "decode_sms", "timestamp_monotonic_s")})
    cs = controller_summary(events)
    dw = sorted(cs["dwell_s"])
    return {"dropped_max": mx, "n_records_with_dropped_field": n,
            "controller_summary": {"split_transitions": cs["split_transitions"],
                                   "residency_fraction_by_decode_sms": cs.get("residency_fraction", {}),
                                   "residency_s": cs["residency_s"],
                                   "dwell_n": len(dw),
                                   "dwell_s_p50": percentile(dw, 0.5) if dw else None,
                                   "dwell_s_p95": percentile(dw, 0.95) if dw else None,
                                   "dwell_s_max": dw[-1] if dw else None}}


# clk.csv per bench window
BITS = {0x4: "SwPowerCap", 0x8: "HwSlowdown", 0x20: "SwThermal", 0x40: "HwThermal", 0x80: "HwPowerBrake"}
clk = []
with open(f"{META}/clk.csv") as f:
    rd = csv.reader(f); next(rd)
    for r in rd:
        try:
            t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
            clk.append((t, int(r[1].split()[0]), float(r[2].split()[0]), int(r[4].strip(), 16)))
        except Exception:
            pass
timing = {}
for row in csv.DictReader(open(f"{D}/TIMING.tsv"), delimiter="\t"):
    timing[row["boot"]] = row


def clk_window(b):
    t0, t1 = float(timing[b]["bench_start_epoch"]), float(timing[b]["bench_end_epoch"])
    w = [x for x in clk if t0 <= x[0] <= t1]
    if not w:
        return {"n": 0}
    bc = Counter()
    for x in w:
        for bit, name in BITS.items():
            if x[3] & bit:
                bc[name] += 1
    sms = sorted(x[1] for x in w)
    return {"n": len(w), "reason_pct": {k: 100 * v / len(w) for k, v in bc.items()},
            "sm_mhz_p5": percentile(sms, 0.05), "sm_mhz_p50": percentile(sms, 0.5), "sm_mhz_min": sms[0]}


res = {"job": JOB, "boots": {}, "lambda0_vessl_a_r4": {}, "lambda0_vessl_a_r2": {}}
for b in BOOTS:
    cj = json.load(open(f"{D}/cell_{b}.json"))
    re_mix = mix.compute(f"{D}/tel_{b}.jsonl", cj["probe_end"])
    keys = ("e_cnt", "e_time", "e_iter", "e_qcond", "e_pact")
    reproduces = all(abs(re_mix[k]["pct"] - cj[k]["pct"]) < 1e-9 for k in keys) and \
        re_mix["split_transition"] == cj["split_transition"] and re_mix["index_hist_busy"] == cj["index_hist_busy"]
    res["boots"][b] = {
        "mix": {k: cj[k] for k in keys} | {
            "busy_n": cj["busy_n"], "snapshots": cj["snapshots"], "index_hist_busy": cj["index_hist_busy"],
            "split_transition": cj["split_transition"], "idx0_pct": cj["idx0"]["pct"], "idx3_pct": cj["idx3"]["pct"],
            "e_iter_idx0_weight_pct": cj["e_iter"]["idx0_weight_pct"],
            "sm_index_mismatch": cj["sm_index_mismatch"], "negative_gaps": cj.get("negative_gaps"),
            "probe_end_in_range": cj.get("probe_end_in_range"), "dropped_by_probe_filter": cj.get("dropped_by_probe_filter"),
            "bench_all_pass": cj["bench"]["all_pass"], "local_recompute_reproduces_cell_json": reproduces},
        "req": req_stats(f"{D}/bench_{b}.jsonl"),
        "green": green(f"{D}/green_{b}.json"),
        "cpu": cpu(f"{D}/cpu_{b}.txt"),
        "tel": tel_scan(f"{D}/tel_{b}.jsonl"),
        "clk_bench": clk_window(b),
        "timing": timing[b],
    }

l0 = json.load(open(L0J))["vessl"]
for s, c in L0_BY_SEED.items():
    sub = f"{L0}/variance" if c in ("a_r4_s3", "a_r4_s4") else L0
    res["lambda0_vessl_a_r4"][s] = {"cell": c, "req": req_stats(f"{sub}/bench_{c}.jsonl"),
                                    "mix": l0[c]["mix"], "cell_json": l0[c]["cell"]}
res["lambda0_vessl_a_r2"] = {"cell": "a_r2", "req": req_stats(f"{L0}/bench_a_r2.jsonl"),
                             "mix": l0["a_r2"]["mix"], "cell_json": l0["a_r2"]["cell"]}

# paired differences (values only; no CI / bootstrap / label)
d = {s: res["boots"][f"a_r4_ON_{s}"]["req"]["achieved_req_s"] - res["boots"][f"a_r4_OFF_{s}"]["req"]["achieved_req_s"]
     for s in SEEDS}
res["paired_achieved_diff_ON_minus_OFF"] = {"by_seed": d, "mean": sum(d.values()) / 4,
                                            "min": min(d.values()), "max": max(d.values()),
                                            "n_negative": sum(v < 0 for v in d.values())}
json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
print("ok")
