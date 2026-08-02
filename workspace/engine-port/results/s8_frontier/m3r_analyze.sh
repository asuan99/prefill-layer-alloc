#!/bin/bash
# M3R realizability summary (DESIGN.md sec 4.3.8(f)).  DIAGNOSTIC ONLY.
# Usage: ./m3r_analyze.sh <job_id>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JOB="${1:?usage: m3r_analyze.sh <job_id>}"
python3 - "$HERE" "$JOB" <<'PY'
import re, sys, collections
here, job = sys.argv[1], sys.argv[2]
rows = {}
hdr = re.compile(r"M3R_PIN arm=(\w+) cell=(d\d+) rate=(\d+):")
gate = re.compile(r"pin_frac=([\d.]+), lower95=([\d.]+), n_episodes=(\d+)")
hist = re.compile(r"E1_REALIZED_hist.*total=([\d.]+)s\): (.*)")
conc = re.compile(r"prefill_active_time_frac=([\d.]+) decode_active_time_frac=([\d.]+)")
import glob
for fn in sorted(glob.glob(f"{here}/m3r_*_{job}_result.txt")):
    cur = None
    for ln in open(fn):
        m = hdr.search(ln)
        if m:
            cur = (m[1], m[2], int(m[3])); rows.setdefault(cur, {}); continue
        if not cur: continue
        m = gate.search(ln)
        if m: rows[cur].update(pf=float(m[1]), lo=float(m[2]), ne=int(m[3]))
        m = hist.search(ln)
        if m:
            rows[cur]["t_prefill"] = float(m[1])
            unsplit = re.search(r"P108:([\d.]+)s\((\d+)%\)", m[2])
            rows[cur]["unsplit_pct"] = int(unsplit[2]) if unsplit else 0
        m = conc.search(ln)
        if m: rows[cur].update(pa=float(m[1]), da=float(m[2]))

print("=== M3R PARTITION REALIZABILITY (diagnostic; no SLO, no goodput) ===")
print("question: of the time prefill actually runs, how much runs at the TARGET partition?\n")
print(f"{'arm':>4} {'cell':>5} {'rate':>4} {'pin_frac':>8} {'lower95':>7} {'n_ep':>5} "
      f"{'unsplit%':>8} {'prefill_active_s':>16} {'pa_frac':>7}")
for k in sorted(rows, key=lambda k: (k[0], int(k[1][1:]), k[2])):
    r = rows[k]
    if "pf" not in r:
        print(f"{k[0]:>4} {k[1]:>5} {k[2]:>4}   (no pin output -- boot or probe failed)")
        continue
    flag = "  <-- target NOT held" if r["lo"] < 0.80 else ""
    print(f"{k[0]:>4} {k[1]:>5} {k[2]:>4} {r['pf']:8.3f} {r['lo']:7.3f} {r['ne']:5d} "
          f"{r.get('unsplit_pct',0):7d}% {r.get('t_prefill',float('nan')):16.2f} "
          f"{r.get('pa',float('nan')):7.3f}{flag}")

print("\n--- directional prediction under the CONSENSUS 1-22 auto-revert mechanism ---")
print("if unsplit time comes from decode being momentarily EMPTY, then raising the")
print("rate should RAISE pin_frac. Per (arm, cell), pin_frac vs rate:")
by = collections.defaultdict(list)
for k, r in rows.items():
    if "pf" in r: by[(k[0], k[1])].append((k[2], r["pf"]))
for k in sorted(by, key=lambda k: (k[0], int(k[1][1:]))):
    v = sorted(by[k])
    trend = "rises" if v[-1][1] > v[0][1] + 0.02 else ("falls" if v[-1][1] < v[0][1] - 0.02 else "flat")
    print(f"  {k[0]:>4} {k[1]:>5}: " + "  ".join(f"r{r}:{p:.2f}" for r, p in v) + f"   => {trend}")
print("\nA cell that never reaches lower95 >= 0.80 at any rate is NOT an operating")
print("point on this substrate at this load -- its label is not its allocation.")
PY
