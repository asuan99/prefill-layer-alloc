#!/usr/bin/env python3
"""Partition-level power observation on one GPU (VESSL A100, ~2.5 min).  Read-only power queries.

Question: the board power meter is one number for the whole GPU.  Can the contribution of each
green-context partition (prefill-side SMs vs decode-side SMs) still be observed?  Method = controlled
exclusion: run each partition's work ALONE and TOGETHER, so the board increment over idle can be
attributed by design, not by a per-partition meter (which does not exist).

Workloads (synthetic, not the serving engine):
  PREFILL-like = compute-bound bf16 GEMM (M=N=K=4096)   on stream A (prefill_sm)
  DECODE-like  = memory-bound bf16 GEMV-batch (16384x16384 weight x 16 tokens) on stream B (decode_sm)
Splits (prefill_sm, decode_sm): (92,16), (64,44), (16,92) via sgl_kernel create_greenctx_stream_by_value,
realized smCount read back from the driver (cuStreamGetGreenCtx + cuGreenCtxGetDevResource).
Plus whole-GPU references (default stream, no green context): PREFILL-only, DECODE-only.
Per split, steady phases of PHASE_S each: idle, P-only, D-only, P+D; then a TOGGLE block: decode runs
continuously while prefill toggles 200 ms ON / 200 ms OFF (bursty prefill over decode).
Sensors (separate processes): NVML buffered power samples (~20 ms), energy counter (100 ms).
Throughput per phase per stream (work units/s) is recorded so energy per unit and interference
(decode throughput with vs without prefill) are visible.

Outputs in <out>/: buffered.csv, counter.csv, phases.csv (label,start,end), work.json, green.json,
summary.json (+ stdout).  --no-green: two plain streams (local code-path check only; NOT a measurement).
"""
import ctypes, json, multiprocessing as mp, os, statistics as st, sys, threading, time

OUT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/tmp/partition_power"
NO_GREEN = "--no-green" in sys.argv
PHASE_S = float(os.environ.get("PP_PHASE_S", 4.0))
GAP_S = float(os.environ.get("PP_GAP_S", 1.5))
SPLITS = [(92, 16), (64, 44), (16, 92)]
TOGGLE_HALF_S = 0.2


# ------------------------------------------------------------------ sensors
def drainer(stop, out):
    import pynvml as n
    n.nvmlInit(); h = n.nvmlDeviceGetHandleByIndex(0); last = 0
    with open(out, "w") as f:
        f.write("t_us,power_mW\n")
        while not stop.is_set():
            try:
                _, smp = n.nvmlDeviceGetSamples(h, n.NVML_TOTAL_POWER_SAMPLES, last)
                for x in smp:
                    if x.timeStamp > last:
                        f.write(f"{x.timeStamp},{x.sampleValue.uiVal}\n"); last = x.timeStamp
            except Exception as e:  # noqa: BLE001
                f.write(f"# error {e!r}\n")
            f.flush(); time.sleep(0.2)


def counter(stop, out):
    import pynvml as n
    n.nvmlInit(); h = n.nvmlDeviceGetHandleByIndex(0)
    with open(out, "w") as f:
        f.write("t,energy_mJ\n")
        while not stop.is_set():
            f.write(f"{time.time():.4f},{n.nvmlDeviceGetTotalEnergyConsumption(h)}\n"); time.sleep(0.02)


# ------------------------------------------------------------------ green context read-back
class CUdevResource(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("_pad", ctypes.c_ubyte * 92), ("_u", ctypes.c_ubyte * 48)]


def green_sm(stream_ptr):
    try:
        lib = ctypes.CDLL("libcuda.so.1"); g = ctypes.c_void_p()
        rc = lib.cuStreamGetGreenCtx(ctypes.c_void_p(stream_ptr), ctypes.byref(g))
        if rc != 0 or not g.value:
            return {"rc": rc, "green_ctx_is_null": True}
        res = CUdevResource()
        rc2 = lib.cuGreenCtxGetDevResource(g, ctypes.byref(res), ctypes.c_int(1))
        return {"rc": rc2, "smCount": int((ctypes.c_uint * 3).from_buffer_copy(bytes(res._u)[:12])[0]) if rc2 == 0 else None}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


# ------------------------------------------------------------------ workloads
def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda:0")
    A = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16); B_ = torch.randn_like(A)
    W = torch.randn(16384, 16384, device=dev, dtype=torch.bfloat16); x = torch.randn(16384, 16, device=dev, dtype=torch.bfloat16)
    def p_unit(): torch.matmul(A, B_)
    def d_unit(): torch.matmul(W, x)
    for _ in range(3):
        p_unit(); d_unit()
    torch.cuda.synchronize()

    phases, work, green = [], {}, {}

    def mark(label, a, b): phases.append((label, a, b))

    def run(label, sp, sd, do_p, do_d, dur, toggle=False):
        """Each active stream gets its own Python thread issuing chunks of units and syncing its stream."""
        cnt = {"p": 0, "d": 0}; stop = threading.Event(); p_on = threading.Event(); p_on.set()
        edges = []
        def loop(kind, stream, unit, k):
            with torch.cuda.stream(stream):
                while not stop.is_set():
                    if kind == "p" and toggle and not p_on.is_set():
                        time.sleep(0.0005); continue
                    for _ in range(k):
                        unit()
                    stream.synchronize(); cnt[kind] += k
        ths = []
        if do_p: ths.append(threading.Thread(target=loop, args=("p", sp, p_unit, 2)))
        if do_d: ths.append(threading.Thread(target=loop, args=("d", sd, d_unit, 4)))
        t0 = time.time()
        for t in ths: t.start()
        if toggle:
            on = True
            while time.time() - t0 < dur:
                edges.append(("p_on" if on else "p_off", time.time()))
                (p_on.set() if on else p_on.clear())
                time.sleep(TOGGLE_HALF_S); on = not on
        else:
            time.sleep(dur)
        stop.set()
        for t in ths: t.join()
        t1 = time.time(); mark(label, t0, t1)
        work[label] = {"dur_s": t1 - t0, "p_units": cnt["p"], "d_units": cnt["d"],
                       "p_units_per_s": cnt["p"] / (t1 - t0), "d_units_per_s": cnt["d"] / (t1 - t0)}
        if toggle:
            work[label]["toggle_edges"] = edges

    def idle(label, dur):
        t0 = time.time(); time.sleep(dur); mark(label, t0, time.time())

    default = torch.cuda.current_stream()
    idle("idle_start", PHASE_S)
    run("whole_P", default, default, True, False, PHASE_S); idle("gap", GAP_S)
    run("whole_D", default, default, False, True, PHASE_S); idle("gap", GAP_S)
    for (ps, ds) in SPLITS:
        tag = f"P{ps}D{ds}"
        if NO_GREEN:
            sp, sd = torch.cuda.Stream(), torch.cuda.Stream()
            green[tag] = {"mode": "no-green (plain streams)"}
        else:
            from sgl_kernel.spatial import create_greenctx_stream_by_value
            sp, sd = create_greenctx_stream_by_value(ps, ds, 0)
            green[tag] = {"target": [ps, ds], "prefill": green_sm(int(sp.cuda_stream)), "decode": green_sm(int(sd.cuda_stream))}
        idle(f"{tag}_idle", PHASE_S)
        run(f"{tag}_P", sp, sd, True, False, PHASE_S); idle("gap", GAP_S)
        run(f"{tag}_D", sp, sd, False, True, PHASE_S); idle("gap", GAP_S)
        run(f"{tag}_PD", sp, sd, True, True, PHASE_S); idle("gap", GAP_S)
        run(f"{tag}_toggle", sp, sd, True, True, PHASE_S, toggle=True); idle("gap", GAP_S)
    idle("idle_end", PHASE_S)
    with open(f"{OUT}/phases.csv", "w") as f:
        f.write("label,start,end\n"); f.writelines(f"{l},{a:.6f},{b:.6f}\n" for l, a, b in phases)
    json.dump(work, open(f"{OUT}/work.json", "w"), indent=1)
    json.dump(green, open(f"{OUT}/green.json", "w"), indent=1)
    print("GREEN", json.dumps(green), flush=True)


def analyze():
    import bisect
    P = [ln.strip().split(",") for ln in open(f"{OUT}/phases.csv")][1:]
    B = [ln.strip().split(",") for ln in open(f"{OUT}/buffered.csv") if ln[0].isdigit()]
    bt = [int(r[0]) / 1e6 for r in B]; bp = [int(r[1]) / 1000 for r in B]
    C = [ln.strip().split(",") for ln in open(f"{OUT}/counter.csv")][1:]
    ct = [float(r[0]) for r in C]; ce = [int(r[1]) for r in C]
    work = json.load(open(f"{OUT}/work.json"))
    def bmean(a, b):
        i, j = bisect.bisect_left(bt, a), bisect.bisect_right(bt, b)
        return st.fmean(bp[i:j]) if j > i else None
    def cmean(a, b):
        def at(x):
            i = bisect.bisect_left(ct, x); i = min(max(i, 1), len(ct) - 1)
            return ce[i - 1] + (ce[i] - ce[i - 1]) * (x - ct[i - 1]) / (ct[i] - ct[i - 1])
        return (at(b) - at(a)) / 1000 / (b - a)
    res = {"phases": {}, "splits": {}}
    for label, a, b in P:
        if label == "gap":
            continue
        a, b = float(a), float(b); a2, b2 = a + 0.5, b - 0.2          # skip ramp-in / ramp-out
        row = {"buf_W": bmean(a2, b2), "ctr_W": cmean(a2, b2)}
        if label in work:
            row |= {k: work[label][k] for k in ("p_units_per_s", "d_units_per_s")}
        res["phases"][label] = row
    idle = st.fmean([res["phases"][k]["ctr_W"] for k in res["phases"] if k.endswith("idle") or k.startswith("idle")])
    res["idle_W"] = idle
    for ps, ds in SPLITS:
        tag = f"P{ps}D{ds}"; ph = res["phases"]
        dP, dD, dPD = (ph[f"{tag}_{k}"]["ctr_W"] - idle for k in ("P", "D", "PD"))
        rP, rD = ph[f"{tag}_P"]["p_units_per_s"], ph[f"{tag}_D"]["d_units_per_s"]
        rPD_p, rPD_d = ph[f"{tag}_PD"]["p_units_per_s"], ph[f"{tag}_PD"]["d_units_per_s"]
        res["splits"][tag] = {
            "dW_P_only": dP, "dW_D_only": dD, "dW_PD": dPD,
            "additivity_residual_W": dPD - (dP + dD), "additivity_ratio": dPD / (dP + dD) if dP + dD else None,
            "P_rate_alone": rP, "P_rate_with_D": rPD_p, "P_slowdown": rPD_p / rP if rP else None,
            "D_rate_alone": rD, "D_rate_with_P": rPD_d, "D_slowdown": rPD_d / rD if rD else None,
            "J_per_P_unit_alone": (dP + idle) / rP if rP else None, "J_per_D_unit_alone": (dD + idle) / rD if rD else None,
        }
        # toggle: buffered power during prefill-ON vs prefill-OFF halves (middle 50 %, shifted by sensor lag 20 ms)
        ed = work[f"{tag}_toggle"]["toggle_edges"]
        halves = [(k, a + 0.02, b + 0.02) for (k, a), (_, b) in zip(ed, ed[1:])]
        halves = halves[len(halves) // 4: 3 * len(halves) // 4]
        on = [m for k, a, b in halves if k == "p_on" for m in [bmean(a + 0.25 * (b - a), b - 0.25 * (b - a))] if m]
        off = [m for k, a, b in halves if k == "p_off" for m in [bmean(a + 0.25 * (b - a), b - 0.25 * (b - a))] if m]
        res["splits"][tag]["toggle_prefill_on_W"] = st.fmean(on) if on else None
        res["splits"][tag]["toggle_prefill_off_W"] = st.fmean(off) if off else None
        res["splits"][tag]["toggle_step_W"] = (st.fmean(on) - st.fmean(off)) if on and off else None
    json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    stop = mp.Event()
    pr = [mp.Process(target=drainer, args=(stop, f"{OUT}/buffered.csv")), mp.Process(target=counter, args=(stop, f"{OUT}/counter.csv"))]
    for p in pr: p.start()
    time.sleep(1.0)
    try:
        main()
    finally:
        time.sleep(0.5); stop.set()
        for p in pr: p.join()
    analyze()
    print("PROBE_DONE", flush=True)
