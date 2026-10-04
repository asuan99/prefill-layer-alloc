#!/usr/bin/env python3
"""Descriptive analysis of lam0v job (CPU only). Output: JSON to stdout path arg."""
import csv, json, os, re, statistics as st, subprocess, sys, datetime as dt
from collections import Counter, defaultdict

ENG = "/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port"
JOB = "pdmux-lambda0-vessl-1d47d3e-20261004084945"
V = f"{ENG}/results/r2_eval/lambda0_vessl/lam0v_{JOB}"
K = f"{ENG}/results/r2_eval/lambda0_prereg/lam0_908623"
META = f"{ENG}/results/lambda0_vessl/vessl/{JOB}/meta"
MIX = f"{ENG}/results/r2_eval/e2_sticky_prereg/e2_realized_mix.py"
OUT = sys.argv[1]

REV5 = ["a_r0", "a_r1", "a_r2", "a_r3", "a_r4", "a_r4_s2", "b_r0", "b_r1", "b_r2", "b_r3", "b_r3_s2"]
VAR = ["a_r4_s3", "b_r3_s3", "a_r4_s4", "b_r3_s4"]
DEADLINE = 1791119405

def d(cell):
    return f"{V}/variance" if cell in VAR else V

def utc(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()

# cell start from target.log CELL header remaining_s
starts = {}
for line in open(f"{ENG}/results/lambda0_vessl/vessl/{JOB}/logs/target.log"):
    m = re.match(r"#+ CELL (\S+) .* remaining_s=(\d+)", line)
    if m:
        starts[m.group(1)] = DEADLINE - int(m.group(2))

SRV_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")
def srv_stats(path):
    run, q, last = 0, 0, None
    for line in open(path, errors="replace"):
        if "Decode batch" in line or "Prefill batch" in line:
            m = re.search(r"#running-req: (\d+)", line); run = max(run, int(m.group(1))) if m else run
            m = re.search(r"#queue-req: (\d+)", line); q = max(q, int(m.group(1))) if m else q
            m = SRV_TS.match(line)
            if m:
                last = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
    return run, q, last

def banner(path):
    t = open(path, errors="replace").read()
    mt = re.findall(r"max_total_num_tokens=(\d+)", t)
    mm = re.findall(r"max_mamba_cache_size: (\d+)", t)
    mr = re.findall(r"max_running_requests=(\d+)", t)
    cg = re.findall(r"disable_cuda_graph=(\w+)", t)
    return {"max_total_num_tokens": sorted(set(mt)), "max_mamba_cache_size": sorted(set(mm)),
            "max_running_requests": sorted(set(mr)), "disable_cuda_graph": sorted(set(cg))}

def tel_dropped(path):
    mx, last, n = 0, None, 0
    for line in open(path):
        m = re.search(r'"dropped_events":(\d+)', line)
        if m:
            v = int(m.group(1)); mx = max(mx, v); last = v; n += 1
    return {"n_records_with_field": n, "max": mx, "final": last}

def green(path):
    g = json.load(open(path))
    out = {}
    for grp in g["groups"]:
        i = grp["stream_index"]
        def sm(x):
            return None if x.get("green_ctx_is_null") else x["green_sm"]["smCount"]
        out[i] = {"target": (grp["target_prefill_sm"], grp["target_decode_sm"]),
                  "green": (sm(grp["prefill"]), sm(grp["decode"]))}
    ok = all((v["green"] == v["target"]) or (v["green"][0] is None and v["green"][1] is None and 0 in v["target"])
             for v in out.values())
    return {"groups": out, "targets_match_or_null_for_unsplit": ok, "sticky_target_index": g.get("sticky_target_index")}

def mix_run(tel):
    r = subprocess.run([sys.executable, MIX, tel], capture_output=True, text=True)
    txt = r.stdout
    j = json.loads(txt[txt.index("{"):])
    return j

res = {"vessl": {}, "kisti": {}}
for c in REV5 + VAR:
    cj = json.load(open(f"{d(c)}/cell_{c}.json"))
    mj = json.load(open(f"{d(c)}/mix_{c}.json"))
    run, q, last = srv_stats(f"{d(c)}/srv_{c}.log")
    res["vessl"][c] = {
        "cell": {k: cj.get(k) for k in ["achieved_rate", "achieved_over_realized", "offered_rate", "num_prompts", "duration_s",
                                        "span_s", "drain_s", "drain_pred_s", "drain_model_ok", "ttft_p50_s", "ttft_p95_s",
                                        "ttft_p99_s", "itl_p50_s", "itl_p95_s", "n_errors", "client_seed", "ebar_guard_ok"]},
        "mix": {k: mj[k]["pct"] for k in ["e_cnt", "e_time", "e_iter", "e_qcond", "e_pact"]} | {
            "busy_n": mj["busy_n"], "idx0_pct": mj["idx0"]["pct"], "idx3_pct": mj["idx3"]["pct"],
            "index_hist_busy": mj.get("index_hist_busy"), "split_transition": mj["split_transition"],
            "sm_index_mismatch": mj.get("sm_index_mismatch")},
        "dropped": tel_dropped(f"{d(c)}/tel_{c}.jsonl"),
        "banner": banner(f"{d(c)}/banner_{c}.txt"),
        "green": green(f"{d(c)}/green_{c}.json"),
        "srv_max_running": run, "srv_max_queue": q,
        "cell_start": starts[c], "cell_end": utc(open(f"{d(c)}/cell_{c}.complete").read().strip()),
        "bench_end": last, "bench_start": last - cj["duration_s"] if last else None,
    }

for c in REV5:
    cj = json.load(open(f"{K}/cell_{c}.json"))
    run, q, _ = srv_stats(f"{K}/srv_{c}.log")
    mj = mix_run(f"{K}/tel_{c}.jsonl")
    res["kisti"][c] = {
        "cell": {k: cj.get(k) for k in ["achieved_rate", "achieved_over_realized", "duration_s", "drain_s", "drain_pred_s",
                                        "ttft_p50_s", "ttft_p95_s", "ttft_p99_s", "itl_p50_s", "itl_p95_s"]},
        "mix": {k: mj[k]["pct"] for k in ["e_cnt", "e_time", "e_iter", "e_qcond", "e_pact"]} | {"busy_n": mj["busy_n"]},
        "dropped": tel_dropped(f"{K}/tel_{c}.jsonl"),
        "banner_srv": banner(f"{K}/srv_{c}.log"),
        "srv_max_running": run, "srv_max_queue": q,
    }

# ---------------- clk.csv
BITS = {0x1: "GpuIdle", 0x2: "AppClocksSetting", 0x4: "SwPowerCap", 0x8: "HwSlowdown", 0x10: "SyncBoost",
        0x20: "SwThermalSlowdown", 0x40: "HwThermalSlowdown", 0x80: "HwPowerBrakeSlowdown", 0x100: "DisplayClockSetting"}
rows = []
with open(f"{META}/clk.csv") as f:
    rd = csv.reader(f); next(rd)
    for r in rd:
        try:
            t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
            sm = int(r[1].strip().split()[0]); pw = float(r[2].strip().split()[0]); tmp = int(r[3].strip())
            v = int(r[4].strip(), 16)
        except Exception:
            continue
        rows.append((t, sm, pw, tmp, v))

def summarize(rs):
    n = len(rs)
    if not n:
        return {"n": 0}
    bc = Counter(); combo = Counter(); nonidle = 0
    for _, _, _, _, v in rs:
        combo[hex(v)] += 1
        if v & ~0x1:
            nonidle += 1
        for b, name in BITS.items():
            if v & b:
                bc[name] += 1
        other = v & ~sum(BITS)
        if other:
            bc[f"other_{hex(other)}"] += 1
    sms = sorted(r[1] for r in rs); pws = sorted(r[2] for r in rs); tm = [r[3] for r in rs]
    pct = lambda a, p: a[min(len(a) - 1, int(round(p / 100 * (len(a) - 1))))]
    ni = [r for r in rs if r[4] & ~0x1]
    return {"n": n, "nonidle_reason_samples": nonidle, "nonidle_pct": 100 * nonidle / n,
            "bit_counts": dict(bc), "bit_pct": {k: 100 * v / n for k, v in bc.items()},
            "combos": dict(combo.most_common(12)),
            "sm_mhz": {"min": sms[0], "p1": pct(sms, 1), "p5": pct(sms, 5), "p50": pct(sms, 50), "p95": pct(sms, 95), "max": sms[-1],
                       "mean": st.mean(sms), "frac_lt_1410": sum(1 for s in sms if s < 1410) / n,
                       "frac_lt_1350": sum(1 for s in sms if s < 1350) / n},
            "sm_mhz_when_nonidle_reason": ({"p5": pct(sorted(r[1] for r in ni), 5), "p50": pct(sorted(r[1] for r in ni), 50),
                                             "min": min(r[1] for r in ni)} if ni else None),
            "power_w": {"p50": pct(pws, 50), "p95": pct(pws, 95), "max": pws[-1]},
            "temp_c": {"max": max(tm), "mean": st.mean(tm)}}

clk = {"overall": summarize(rows), "span": (rows[0][0], rows[-1][0])}
# hist by 5-min bins
bins = defaultdict(list)
for r in rows:
    bins[int((r[0] - rows[0][0]) // 300)].append(r)
clk["by_5min"] = {}
for b in sorted(bins):
    rs = bins[b]; s = summarize(rs)
    clk["by_5min"][dt.datetime.fromtimestamp(rows[0][0] + 300 * b, dt.timezone.utc).strftime("%H:%M")] = {
        "n": s["n"], "nonidle_pct": s["nonidle_pct"], "bit_pct": {k: round(v, 2) for k, v in s["bit_pct"].items() if k != "GpuIdle"},
        "sm_p50": s["sm_mhz"]["p50"], "sm_min": s["sm_mhz"]["min"], "pw_p95": s["power_w"]["p95"], "temp_max": s["temp_c"]["max"]}
# per-cell windows
clk["cells"] = {}
for c, r in res["vessl"].items():
    cw = [x for x in rows if r["cell_start"] <= x[0] <= r["cell_end"]]
    bw = [x for x in rows if r["bench_start"] and r["bench_start"] <= x[0] <= r["bench_end"]]
    clk["cells"][c] = {"cell_window": summarize(cw), "bench_window": summarize(bw)}
# outside any cell window (preflight etc.)
inwin = lambda t: any(r["cell_start"] <= t <= r["cell_end"] for r in res["vessl"].values())
clk["outside_cells"] = summarize([x for x in rows if not inwin(x[0])])
# contiguous episodes of non-idle reasons
eps = []; cur = None
for x in rows:
    if x[4] & ~0x1:
        if cur and x[0] - cur[1] <= 0.35:
            cur[1] = x[0]; cur[2] += 1; cur[3] |= x[4]
        else:
            if cur: eps.append(cur)
            cur = [x[0], x[0], 1, x[4]]
if cur: eps.append(cur)
durs = sorted(e[1] - e[0] for e in eps)
clk["episodes"] = {"n": len(eps), "dur_s_p50": durs[len(durs) // 2] if durs else None, "dur_s_max": durs[-1] if durs else None,
                   "top10": [(dt.datetime.fromtimestamp(e[0], dt.timezone.utc).strftime("%H:%M:%S"), round(e[1] - e[0], 1), e[2], hex(e[3]))
                             for e in sorted(eps, key=lambda e: -(e[1] - e[0]))[:10]]}
res["clk"] = clk
json.dump(res, open(OUT, "w"), indent=1, default=str)
print("ok", len(rows))
