#!/usr/bin/env bash
# VESSL energy-counter probe (~1 min).  Read-only GPU queries + a 20 s matmul load.
# Checks: (1) NVML nvmlDeviceGetTotalEnergyConsumption readable in the container,
# (2) counter delta vs integral of power.draw (1 s avg on A100) and power.draw.instant at 100 ms,
# (3) counter resolution / update cadence, (4) whether host RAPL is readable (informational only).
set -u
echo "# $(date -u +%FT%TZ) uid=$(id -u)"
nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader
nvidia-smi --query-gpu=timestamp,power.draw,power.draw.instant,clocks.sm,clocks_event_reasons.active \
  --format=csv,noheader -lms 100 > /tmp/pw.csv 2>/tmp/pw.err &
SMI=$!
python3 - <<'PY'
import time, json, threading
import pynvml as n
n.nvmlInit(); h = n.nvmlDeviceGetHandleByIndex(0)
try:
    e0 = n.nvmlDeviceGetTotalEnergyConsumption(h); print("ENERGY_READ_OK", e0)
except n.NVMLError as e:
    print("ENERGY_READ_FAIL", e); raise SystemExit(0)
# cadence: how often does the counter change at idle (poll 5 ms for 2 s)
vals, t0 = [], time.time()
while time.time() - t0 < 2.0:
    vals.append((time.time(), n.nvmlDeviceGetTotalEnergyConsumption(h))); time.sleep(0.005)
ch = [vals[i][0] for i in range(1, len(vals)) if vals[i][1] != vals[i-1][1]]
gaps = [b - a for a, b in zip(ch, ch[1:])]
print("IDLE_COUNTER_CHANGES", len(ch), "median_gap_ms", round(1000*sorted(gaps)[len(gaps)//2], 1) if gaps else None)
# phases: idle 5 s, load 20 s, idle 5 s; record counter at each boundary with wall time
import torch
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
torch.cuda.synchronize()
marks = []
def mark(tag):
    torch.cuda.synchronize(); marks.append((tag, time.time(), n.nvmlDeviceGetTotalEnergyConsumption(h)))
mark("idle0_start"); time.sleep(5); mark("idle0_end")
mark("load_start"); t = time.time()
while time.time() - t < 20:
    for _ in range(20): c = a @ b
    torch.cuda.synchronize()
mark("load_end")
mark("idle1_start"); time.sleep(5); mark("idle1_end")
json.dump(marks, open("/tmp/marks.json", "w"))
for i in range(0, len(marks), 2):
    (ta, ws, es), (tb, we, ee) = marks[i], marks[i+1]
    print("PHASE", ta.replace("_start", ""), "dur_s", round(we - ws, 3), "dE_J", round((ee - es) / 1000, 1), "mean_W", round((ee - es) / 1000 / (we - ws), 1))
PY
sleep 1; kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
python3 - <<'PY'
import csv, json, datetime as dt
rows = []
for r in csv.reader(open("/tmp/pw.csv")):
    try:
        t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
        rows.append((t, float(r[1].split()[0]), float(r[2].split()[0])))
    except Exception:
        pass
print("SMI_ROWS", len(rows), "err:", open("/tmp/pw.err").read().strip()[:200])
try:
    marks = json.load(open("/tmp/marks.json"))
except Exception:
    raise SystemExit(0)
def integ(a, b, col):
    w = [x for x in rows if a <= x[0] <= b]
    if len(w) < 2: return None
    return sum(w[i][col] * (w[i+1][0] - w[i][0]) for i in range(len(w) - 1))
for i in range(0, len(marks), 2):
    (ta, ws, es), (tb, we, ee) = marks[i], marks[i+1]
    ec = (ee - es) / 1000
    pa, pi = integ(ws, we, 1), integ(ws, we, 2)
    f = lambda v: None if v is None else round(v, 1)
    print("COMPARE", ta.replace("_start", ""), "counter_J", round(ec, 1), "int_power.draw_J", f(pa), "int_instant_J", f(pi),
          "ratio_draw", f(pa / ec * 100) if pa else None, "ratio_instant", f(pi / ec * 100) if pi else None)
PY
echo "## RAPL (informational: shared host, not attributable)"
ls /sys/class/powercap 2>&1 | head -3; cat /sys/class/powercap/intel-rapl:0/energy_uj 2>&1 | head -1
echo PROBE_DONE
