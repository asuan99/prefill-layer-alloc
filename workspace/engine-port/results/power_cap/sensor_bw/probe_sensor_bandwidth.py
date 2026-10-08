#!/usr/bin/env python3
"""GPU power-sensor bandwidth probe (VESSL A100, ~2 min).  Read-only NVML queries + a square-wave load.

Question: below 100 ms, do NVML power readings / the energy counter resolve a known on/off load?
Load: 50 % duty square wave with periods P in {10, 20, 50, 100, 200, 400} ms, 6 s each, 2 s idle between.
Ground truth: the load process logs every on/off edge (wall clock, after cuda synchronize).
Sensors (separate process, ~1 ms polling, wall clock): nvmlDeviceGetPowerUsage (mW), field values
POWER_INSTANT / POWER_AVERAGE (mW, if supported), nvmlDeviceGetTotalEnergyConsumption (mJ).
Per (source, P): update interval (median gap between value changes), and modulation depth =
(mean in ON half - mean in OFF half) / reference swing (continuous REF_S load vs the following idle,
middle 50 % of each).  Depth ~1 = the source resolves that period; ~0.64 (=2/pi... for a 1st-order
filter at its corner) or lower = smoothed; ~0 = invisible.
For the counter: per-half-cycle energy difference vs the same reference.
Raw samples -> <out>/samples.csv, edges -> <out>/edges.csv, summary -> <out>/summary.json + stdout.
"""
import json, multiprocessing as mp, os, statistics as st, sys, time

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sensor_bw"
PERIODS_MS = [400, 200, 100, 50, 20, 10]
BLOCK_S, GAP_S = float(os.environ.get("SBW_BLOCK_S", 6.0)), float(os.environ.get("SBW_GAP_S", 2.0))
REF_S = float(os.environ.get("SBW_REF_S", 4.0))
FI_AVG, FI_INST = 185, 186


def poller(stop, out):
    import pynvml as n
    n.nvmlInit(); h = n.nvmlDeviceGetHandleByIndex(0)
    have_fi = True
    try:
        n.nvmlDeviceGetFieldValues(h, [FI_INST, FI_AVG])
    except Exception as e:  # noqa: BLE001
        have_fi = False; print("FIELD_VALUES_UNAVAILABLE", repr(e), flush=True)
    with open(out, "w") as f:
        f.write("t,usage_mW,inst_mW,avg_mW,energy_mJ\n")
        while not stop.is_set():
            t = time.time()
            u = n.nvmlDeviceGetPowerUsage(h)
            ins = avg = ""
            if have_fi:
                try:
                    fv = n.nvmlDeviceGetFieldValues(h, [FI_INST, FI_AVG])
                    ins = fv[0].value.uiVal if fv[0].nvmlReturn == 0 else ""
                    avg = fv[1].value.uiVal if fv[1].nvmlReturn == 0 else ""
                except Exception:  # noqa: BLE001
                    pass
            e = n.nvmlDeviceGetTotalEnergyConsumption(h)
            f.write(f"{t:.6f},{u},{ins},{avg},{e}\n")
            time.sleep(0.0005)


def load(edges_path):
    import torch
    n_ = 8192
    while True:   # shrink the matmul until one call takes <= 1 ms, so a 5 ms ON half holds several calls
        a = torch.randn(n_, n_, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
        for _ in range(5):
            a @ b
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(10):
            a @ b
        torch.cuda.synchronize(); mm = (time.time() - t) / 10
        if mm <= 0.001 or n_ <= 1024:
            break
        n_ //= 2
    q = max(1, int(0.001 / mm))             # calls queued per sync: ~1 ms of saturated work per chunk
    print("MATMUL_N", n_, "QUEUE", q, flush=True)
    def chunk():
        for _ in range(q):
            a @ b
        torch.cuda.synchronize()
    mm = mm * q                               # one chunk
    edges = []
    time.sleep(GAP_S)
    # reference: continuous load REF_S, then idle GAP_S (full-swing amplitude for every source)
    t_on = time.time(); edges.append((0, "on", t_on))
    while time.time() - t_on < REF_S:
        chunk()
    edges.append((0, "off", time.time()))
    time.sleep(GAP_S)
    edges.append((0, "end", time.time()))
    for P in PERIODS_MS:
        half = P / 2000.0
        t_end = time.time() + BLOCK_S
        while time.time() < t_end:
            t_on = time.time(); edges.append((P, "on", t_on))
            while True:                              # at least one chunk per ON half
                chunk()
                if time.time() - t_on >= half - mm / 2:
                    break
            t_off = time.time(); edges.append((P, "off", t_off))
            while time.time() - t_off < half:
                time.sleep(0.0002)
        time.sleep(GAP_S)
    with open(edges_path, "w") as f:
        f.write("P_ms,edge,t\n")
        f.writelines(f"{p},{k},{t:.6f}\n" for p, k, t in edges)
    print("MATMUL_MS", round(1000 * mm, 2), flush=True)


def analyze():
    import csv
    S = list(csv.DictReader(open(f"{OUT}/samples.csv")))
    E = list(csv.DictReader(open(f"{OUT}/edges.csv")))
    t = [float(r["t"]) for r in S]
    src = {k: [float(r[k]) if r[k] != "" else None for r in S] for k in ("usage_mW", "inst_mW", "avg_mW", "energy_mJ")}
    gaps = [b - a for a, b in zip(t, t[1:])]
    res = {"n_samples": len(S), "poll_dt_ms_p50": 1000 * st.median(gaps), "poll_dt_ms_p99": 1000 * sorted(gaps)[int(.99 * len(gaps))],
           "sources": {}}
    for k, v in src.items():
        if all(x is None for x in v):
            res["sources"][k] = "unavailable"; continue
        ch = [t[i] for i in range(1, len(v)) if v[i] is not None and v[i - 1] is not None and v[i] != v[i - 1]]
        cg = [b - a for a, b in zip(ch, ch[1:])]
        res["sources"][k] = {"update_ms_p10": 1000 * sorted(cg)[len(cg) // 10] if cg else None,
                             "update_ms_p50": 1000 * st.median(cg) if cg else None, "n_changes": len(ch)}
    # half-cycle windows per period
    import bisect
    def halves(P):
        ed = [(k, float(x)) for p, k, x in ((r["P_ms"], r["edge"], r["t"]) for r in E) if int(p) == P]
        out = []
        for (k, a), (_, b) in zip(ed, ed[1:]):
            out.append((k, a, b))
        n = len(out); return out[n // 4: 3 * n // 4]          # middle 50 % of the block
    def mean_in(vals, a, b):
        i, j = bisect.bisect_left(t, a), bisect.bisect_right(t, b)
        w = [vals[x] for x in range(i, j) if vals[x] is not None]
        return st.fmean(w) if w else None
    def ctr_at(x):
        v = src["energy_mJ"]; i = bisect.bisect_left(t, x)
        i = min(max(i, 1), len(t) - 1)
        return v[i - 1] + (v[i] - v[i - 1]) * (x - t[i - 1]) / (t[i] - t[i - 1]) if t[i] > t[i - 1] else v[i]
    swing = {}
    res["periods"] = {}
    for P in PERIODS_MS:
        H = halves(P); row = {"n_halves": len(H)}
        for k in ("usage_mW", "inst_mW", "avg_mW"):
            if res["sources"][k] == "unavailable":
                continue
            on = [m for kk, a, b in H if kk == "on" for m in [mean_in(src[k], a, b)] if m is not None]
            off = [m for kk, a, b in H if kk == "off" for m in [mean_in(src[k], a, b)] if m is not None]
            d = (st.fmean(on) - st.fmean(off)) / 1000 if on and off else None
            row[f"{k}_on_minus_off_W"] = d
        pon = [(ctr_at(b) - ctr_at(a)) / (b - a) for kk, a, b in H if kk == "on" and b > a]
        poff = [(ctr_at(b) - ctr_at(a)) / (b - a) for kk, a, b in H if kk == "off" and b > a]
        row["on_half_ms_p50"] = 1000 * st.median([b - a for kk, a, b in H if kk == "on"]) if H else None
        row["counter_on_minus_off_W"] = (st.fmean(pon) - st.fmean(poff)) / 1000 if pon and poff else None
        res["periods"][str(P)] = row
    ed0 = {k: float(x) for p_, k, x in ((r["P_ms"], r["edge"], r["t"]) for r in E) if int(p_) == 0}
    on_a, on_b = ed0["on"], ed0["off"]; off_a, off_b = ed0["off"], ed0["end"]
    q_ = lambda a, b: (a + 0.25 * (b - a), a + 0.75 * (b - a))
    ref = {}
    for k in ("usage_mW", "inst_mW", "avg_mW"):
        if res["sources"][k] != "unavailable":
            ref[f"{k}_on_minus_off_W"] = (mean_in(src[k], *q_(on_a, on_b)) - mean_in(src[k], *q_(off_a, off_b))) / 1000
    (a1, b1), (a2, b2) = q_(on_a, on_b), q_(off_a, off_b)
    ref["counter_on_minus_off_W"] = ((ctr_at(b1) - ctr_at(a1)) / (b1 - a1) - (ctr_at(b2) - ctr_at(a2)) / (b2 - a2)) / 1000
    res["reference_continuous_load"] = ref
    for P, row in res["periods"].items():
        for k in list(row):
            if k.endswith("_on_minus_off_W") and row[k] is not None and ref.get(k):
                row[k.replace("_on_minus_off_W", "_depth")] = row[k] / ref[k]
    json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    stop = mp.Event()
    p = mp.Process(target=poller, args=(stop, f"{OUT}/samples.csv")); p.start()
    time.sleep(1.0)
    load(f"{OUT}/edges.csv")
    stop.set(); p.join()
    analyze()
    print("PROBE_DONE", flush=True)
