#!/usr/bin/env python3
"""Re-derivation + descriptive statistics for the E2 VESSL re-run Job
pdmux-e2-sticky-vessl-22c8c64-20261006020317 (20/20 boots).  CPU only, GPU 0.

What this script does (and does not):
  * LABELS: re-runs the REGISTERED rule file e2_sticky_prereg/e2_label.py (imported,
    unmodified) on the Job's cell_*.json and checks that the result equals the
    in-Job E2_LABEL.json.  It never computes a label of its own.
  * ESTIMATORS: re-runs e2_realized_mix.compute (imported, unmodified) on every
    telemetry file with the cell's own probe_end and checks the cell_*.json fields.
  * Everything else (request percentiles, canonical-predicate goodput, green
    read-out, cpu.stat, clk, timing, P3 keys, P13) is DESCRIPTIVE, not a label.
  * Cross-check only: pdmux_eval.analyze.paired_bootstrap_ci (seed=1, 10000) on the
    same achieved pairs -- a different bootstrap convention from the registered
    R2 (numpy rng 20260915, B=100000); reported beside it, never instead of it.
Output: JSON to argv[1].
"""
import csv, datetime as dt, json, os, re, sys, tempfile
from collections import Counter

ENG = "/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port"
sys.path.insert(0, f"{ENG}/benchmarks")
sys.path.insert(0, f"{ENG}/results/r2_eval/e2_sticky_prereg")
from pdmux_eval.analyze import (percentile, summarize_requests, load_bench_serving_rounds,  # noqa: E402
                                request_tpot_percentiles, controller_summary, paired_bootstrap_ci)
import e2_realized_mix as mix  # noqa: E402
import e2_label  # noqa: E402

JOB = "pdmux-e2-sticky-vessl-22c8c64-20261006020317"
JOB1 = "pdmux-e2-sticky-vessl-e6f424e-20261004135228"
D = f"{ENG}/results/r2_eval/e2_sticky_vessl/e2v_{JOB}"
META = f"{ENG}/results/e2_sticky_vessl/vessl/{JOB}/meta"
J1 = f"{ENG}/results/r2_eval/e2_sticky_vessl/analysis_2026-10-04/e2v_descriptive.json"
L0 = f"{ENG}/results/r2_eval/lambda0_vessl/lam0v_pdmux-lambda0-vessl-1d47d3e-20261004084945"
L0J = f"{ENG}/results/r2_eval/lambda0_vessl/analysis_2026-10-04/lam0v_descriptive.json"
TTFT_SLO, ITL_SLO = 3000.0, 50.0
SEEDS_A = ["4386", "4162", "251", "2630"]
SEEDS_B = ["4386", "4162"]
ORDER = [f"{c}_{a}_{s}" for c, ss in (("a_r4", SEEDS_A), ("a_r2", SEEDS_A), ("b_r3", SEEDS_B))
         for s in ss for a in ("OFF", "ON")]
L0_CELL = {("a_r4", "4386"): "a_r4", ("a_r4", "4162"): "a_r4_s2", ("a_r4", "251"): "a_r4_s3",
           ("a_r4", "2630"): "a_r4_s4", ("a_r2", "4386"): "a_r2",
           ("b_r3", "4386"): "b_r3", ("b_r3", "4162"): "b_r3_s2"}
L0_VARIANCE = {"a_r4_s3", "a_r4_s4", "b_r3_s3", "b_r3_s4"}
REG_OVERHEAD = {"a": 160.1, "b": 121.0}


def req_stats(bench_path):
    reqs, dur = load_bench_serving_rounds(bench_path)
    s = summarize_requests(reqs, 0.0, dur, TTFT_SLO, ITL_SLO)
    rec = json.loads(open(bench_path).readline())
    p95s = [percentile(r.token_itl_ms, 0.95) for r in reqs]
    ttfts = [r.ttft_ms for r in reqs]
    return {
        "completed": rec.get("completed"), "n_requests_loaded": len(reqs), "duration_s": dur,
        "n_rounds_in_file": sum(1 for _ in open(bench_path)),
        "achieved_req_s": rec.get("completed") / dur, "bench_request_throughput": rec.get("request_throughput"),
        "n_errors": sum(1 for e in rec.get("errors", []) if e),
        "ttft_ms": {k: s[f"ttft_p{k}_ms"] for k in (50, 95, 99)} | {"min": min(ttfts), "max": max(ttfts)},
        "req_itl_p95_ms": {"p50": percentile(p95s, 0.50), "p95": percentile(p95s, 0.95),
                           "p99": percentile(p95s, 0.99), "min": min(p95s), "max": max(p95s)},
        "pooled_token_itl_ms": {k: s[f"token_itl_p{k}_ms"] for k in (50, 95, 99)},
        "tpot": request_tpot_percentiles(reqs),
        "good": sum(r.passes(TTFT_SLO, ITL_SLO) for r in reqs),
        "ttft_ok": sum(t <= TTFT_SLO for t in ttfts), "itl_ok": sum(p <= ITL_SLO for p in p95s),
        "slo_goodput_req_s": s["slo_goodput_req_s"],
        "legacy_mean_itl_goodput_req_s": s["legacy_mean_itl_goodput_req_s"],
        "near_itl_slo_pm5pct": sum(0.95 * ITL_SLO <= p <= 1.05 * ITL_SLO for p in p95s),
        "near_ttft_slo_pm5pct": sum(0.95 * TTFT_SLO <= t <= 1.05 * TTFT_SLO for t in ttfts),
    }


def green(path):
    g = json.load(open(path))
    out = {}
    for grp in g["groups"]:
        sm = lambda x: None if x.get("green_ctx_is_null") else x["green_sm"]["smCount"]
        out[grp["stream_index"]] = {"target": [grp["target_prefill_sm"], grp["target_decode_sm"]],
                                    "green": [sm(grp["prefill"]), sm(grp["decode"])]}
    ok = all(v["green"] == v["target"] or v["green"] == [None, None] for v in out.values())
    return {"groups": out, "sticky_target_index": g.get("sticky_target_index"), "targets_match_or_null": ok}


def cpu(path):
    blocks, cur = [], None
    for line in open(path):
        if line.startswith("#"):
            cur = {"hdr": line.strip()}; blocks.append(cur)
        else:
            k, v = line.split(); cur[k] = int(v)
    a, b = blocks
    return {k: b[k] - a[k] for k in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec")}


def tel_scan(path):
    mx = 0
    events = []
    for line in open(path):
        m = re.search(r'"dropped_events":(\d+)', line)
        if m:
            mx = max(mx, int(m.group(1)))
        if '"runtime_snapshot"' in line and '"phase":"benchmark"' in line:
            r = json.loads(line)
            events.append({k: r.get(k) for k in ("event", "phase", "decode_sms", "timestamp_monotonic_s")})
    cs = controller_summary(events)
    dw = sorted(cs["dwell_s"])
    return {"dropped_max": mx,
            "controller_summary_AUX": {"split_transitions": cs["split_transitions"],
                                       "residency_fraction_by_decode_sms": cs.get("residency_fraction", {}),
                                       "dwell_s_p50": percentile(dw, 0.5) if dw else None,
                                       "dwell_s_p95": percentile(dw, 0.95) if dw else None,
                                       "dwell_s_max": dw[-1] if dw else None}}


# ---- clk ----
BITS = {0x4: "SwPowerCap", 0x8: "HwSlowdown", 0x20: "SwThermal", 0x40: "HwThermal", 0x80: "HwPowerBrake",
        0x2: "AppClk", 0x10: "SyncBoost"}
clk = []
with open(f"{META}/clk.csv") as f:
    rd = csv.reader(f); next(rd)
    for r in rd:
        try:
            t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
            clk.append((t, int(r[1].split()[0]), float(r[2].split()[0]), int(r[4].strip(), 16)))
        except Exception:
            pass
timing = {row["boot"]: row for row in csv.DictReader(open(f"{D}/TIMING.tsv"), delimiter="\t")}
windows = [(float(timing[b]["bench_start_epoch"]), float(timing[b]["bench_end_epoch"])) for b in ORDER]


def in_bench(t):
    return any(a <= t <= b for a, b in windows)


thr = [x for x in clk if (x[3] & ~0x1) != 0]
clk_total = {"samples": len(clk), "throttle_active": len(thr),
             "throttle_in_bench": sum(in_bench(x[0]) for x in thr),
             "throttle_out_bench": sum(not in_bench(x[0]) for x in thr),
             "samples_in_bench": sum(in_bench(x[0]) for x in clk),
             "bits_all": dict(Counter(n for x in thr for bit, n in BITS.items() if x[3] & bit)),
             "bits_out_bench": dict(Counter(n for x in thr if not in_bench(x[0]) for bit, n in BITS.items() if x[3] & bit))}


def clk_window(b):
    t0, t1 = float(timing[b]["bench_start_epoch"]), float(timing[b]["bench_end_epoch"])
    w = [x for x in clk if t0 <= x[0] <= t1]
    bc = Counter(n for x in w for bit, n in BITS.items() if x[3] & bit)
    sms = sorted(x[1] for x in w)
    return {"n": len(w), "reason_pct": {k: 100 * v / len(w) for k, v in bc.items()},
            "sm_mhz_p50": percentile(sms, 0.5), "sm_mhz_min": sms[0]}


# ---- settle / port / banners ----
settle = {}
for line in open(f"{D}/SETTLE.txt"):
    p = line.split()
    settle[p[0]] = dict(kv.split("=") for kv in p[1:])
ports = {}
for line in open(f"{D}/PORT_RANGE.txt"):
    p = line.split()
    if len(p) == 3 and p[1].startswith("port="):
        ports[p[0]] = {"port": int(p[1][5:]), "tries": int(p[2][6:])}
banners = {}
for line in open(f"{D}/BANNERS.txt"):
    p = line.split()
    banners[p[0]] = {"mamba": int(p[1].split("=")[1]), "tokens": int(p[2].split("=")[1])}
keys_seq = [tuple(map(int, l.split())) for l in open(f"{D}/P3_KEYS.txt") if l.strip()]

res = {"job": JOB, "order": ORDER, "boots": {}}
prev_end = None
for i, b in enumerate(ORDER):
    cell = b.split("_")[0] + "_" + b.split("_")[1]
    cj = json.load(open(f"{D}/cell_{b}.json"))
    re_mix = mix.compute(f"{D}/tel_{b}.jsonl", cj["probe_end"])
    ks = ("e_cnt", "e_time", "e_iter", "e_qcond", "e_pact")
    reproduces = all(abs(re_mix[k]["pct"] - cj[k]["pct"]) < 1e-9 for k in ks) and \
        re_mix["split_transition"] == cj["split_transition"] and re_mix["index_hist_busy"] == cj["index_hist_busy"]
    t = {k: int(v) for k, v in timing[b].items() if k.endswith("epoch") or k == "bench_rc"}
    bench_s = t["bench_end_epoch"] - t["bench_start_epoch"]
    wall = t["end_epoch"] - t["admit_epoch"]
    res["boots"][b] = {
        "seq": i + 1,
        "mix": {k: cj[k] for k in ks} | {
            "busy_n": cj["busy_n"], "index_hist_busy": cj["index_hist_busy"], "idx0": cj["idx0"], "idx3": cj["idx3"],
            "split_transition": cj["split_transition"], "sm_index_mismatch": cj["sm_index_mismatch"],
            "negative_gaps": cj.get("negative_gaps"), "probe_end": cj["probe_end"],
            "dropped_by_probe_filter": cj.get("dropped_by_probe_filter"),
            "bench_all_pass": cj["bench"]["all_pass"], "bench_achieved_cell_json": cj["bench"]["achieved_req_s"],
            "local_recompute_reproduces_cell_json": reproduces},
        "req": req_stats(f"{D}/bench_{b}.jsonl"),
        "green": green(f"{D}/green_{b}.json"),
        "cpu_bench_delta": cpu(f"{D}/cpu_{b}.txt"),
        "tel": tel_scan(f"{D}/tel_{b}.jsonl"),
        "clk_bench": clk_window(b),
        "timing": t | {"bench_s": bench_s, "boot_wall_s": wall, "overhead_s": wall - bench_s,
                       "gap_from_prev_end_s": (t["admit_epoch"] - prev_end) if prev_end else None,
                       "admit_to_health_s": t["health_epoch"] - t["admit_epoch"],
                       "health_to_bench_s": t["bench_start_epoch"] - t["health_epoch"],
                       "bench_end_to_end_s": t["end_epoch"] - t["bench_end_epoch"]},
        "settle": settle.get(b), "port": ports.get(b), "banner": banners[b], "p3_key_seq": list(keys_seq[i]),
    }
    prev_end = t["end_epoch"]

# ---- label reproduction (registered code, unmodified) ----
cells = e2_label.load_cells(D)
# registered --expect order (e2_vessl_plan_r2.py --expect); a sorted order reproduces every rule value
# but not the byte-identical JSON (list order), so the registered order is used.
expect = [(c, s) for c, ss in (("a_r4", SEEDS_A), ("a_r2", SEEDS_A), ("b_r3", SEEDS_B)) for s in ss]
in_job = json.load(open(f"{D}/E2_LABEL.json"))
with tempfile.TemporaryDirectory() as td:
    out = os.path.join(td, "L.json")
    exp = ",".join(f"{c}/{s}" for c, s in expect)
    old = sys.argv
    sys.argv = ["e2_label.py", "--cells-dir", D, "--expect", exp, "--out", out]
    import contextlib, io
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            e2_label.main()
    except SystemExit:
        pass
    sys.argv = old
    relabel = json.load(open(out))
res["label_reproduction"] = {"identical_to_in_job_E2_LABEL": relabel == in_job,
                             "rules_identical": relabel.get("rules") == in_job.get("rules")}
res["E2_LABEL_in_job"] = in_job

# ---- R2 cross-check with canonical analyze.paired_bootstrap_ci (different convention) ----
res["paired"] = {}
for c in ("a_r4", "a_r2"):
    off = {s: res["boots"][f"{c}_OFF_{s}"]["req"]["achieved_req_s"] for s in SEEDS_A}
    on = {s: res["boots"][f"{c}_ON_{s}"]["req"]["achieved_req_s"] for s in SEEDS_A}
    d = {s: on[s] - off[s] for s in SEEDS_A}
    res["paired"][c] = {"off": off, "on": on, "d": d, "d_over_off_pct": {s: 100 * d[s] / off[s] for s in SEEDS_A},
                        "analyze_paired_bootstrap_ci_seed1_10000": paired_bootstrap_ci(off, on)}
offb = {s: res["boots"][f"b_r3_OFF_{s}"]["req"]["achieved_req_s"] for s in SEEDS_B}
onb = {s: res["boots"][f"b_r3_ON_{s}"]["req"]["achieved_req_s"] for s in SEEDS_B}
res["paired"]["b_r3"] = {"off": offb, "on": onb, "d": {s: onb[s] - offb[s] for s in SEEDS_B}}

# ---- P13 ----
g0, g1 = json.load(open(f"{D}/greedy_OFF.json")), json.load(open(f"{D}/greedy_ON.json"))
res["p13"] = {"n_off": len(g0["texts"]), "n_on": len(g1["texts"]), "texts_equal": g0["texts"] == g1["texts"],
              "per_prompt_equal": [a == b for a, b in zip(g0["texts"], g1["texts"])],
              "empty_any": any(not x for x in g0["texts"] + g1["texts"]),
              "bytes_off": len(json.dumps(g0["texts"]).encode()), "bytes_on": len(json.dumps(g1["texts"]).encode()),
              "in_job_verdict": open(f"{D}/P13_VERDICT.txt").read().strip()}

# ---- job-level meta ----
cg = open(f"{META}/cgroup_cpu.txt").read()
def _blk(txt, tag):
    seg = txt.split(tag)[1].split("#")[0]
    return {l.split()[0]: int(l.split()[1]) for l in seg.strip().splitlines() if len(l.split()) == 2}
a_, b_ = _blk(cg, "# at start"), _blk(cg, "# at end")
res["meta"] = {"timing": open(f"{META}/timing.txt").read().split(), "clk": clk_total,
               "clk_summary": json.load(open(f"{META}/clk_summary.json")),
               "substrate": json.load(open(f"{META}/substrate.json")),
               "cgroup_job_delta": {k: b_[k] - a_[k] for k in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec")},
               "cost": open(f"{META}/../cost.txt").read()}
st = dt.datetime.fromisoformat(res["meta"]["timing"][0].split("=")[1].replace("Z", "+00:00")).timestamp()
en = dt.datetime.fromisoformat(res["meta"]["timing"][1].split("=")[1].replace("Z", "+00:00")).timestamp()
res["meta"]["container_s"] = en - st
res["meta"]["container_h"] = (en - st) / 3600

# ---- first (aborted) Job a_r4, E2VC-13 ----
j1 = json.load(open(J1))
res["job1_a_r4"] = {s: {"off": j1["boots"][f"a_r4_OFF_{s}"]["req"]["achieved_req_s"],
                        "on": j1["boots"][f"a_r4_ON_{s}"]["req"]["achieved_req_s"]} for s in SEEDS_A}
for s, v in res["job1_a_r4"].items():
    v["d"] = v["on"] - v["off"]; v["d_over_off_pct"] = 100 * v["d"] / v["off"]

# ---- lambda0 VESSL same-cell OFF (different Job / GPU) ----
l0 = json.load(open(L0J))["vessl"]
res["lambda0_vessl"] = {}
for (c, s), lc in L0_CELL.items():
    sub = f"{L0}/variance" if lc in L0_VARIANCE else L0
    res["lambda0_vessl"][f"{c}/{s}"] = {"l0_cell": lc, "req": req_stats(f"{sub}/bench_{lc}.jsonl"),
                                        "mix": l0[lc]["mix"]}
json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
print("ok", res["label_reproduction"], res["p13"]["texts_equal"])
