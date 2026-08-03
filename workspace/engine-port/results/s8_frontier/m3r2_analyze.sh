#!/bin/bash
# M3R rev2 summary (DESIGN.md sec 4.3.8(g)). DIAGNOSTIC ONLY -- no ITL, no SLO.
# Usage: ./m3r2_analyze.sh <job_id>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JOB="${1:?usage: m3r2_analyze.sh <job_id>}"
python3 - "$HERE" "$JOB" <<'PY'
import re, sys, glob, statistics as st, collections
here, job = sys.argv[1], sys.argv[2]

hdr = re.compile(r"M3R2_PIN arm=(\w+) cell=(d\d+) rate=(\d+) boot=(\d+) seed=(\d+):")
gate = re.compile(r"(PASS|FAIL).*?pin_frac=([\d.]+), lower95=([\d.]+), "
                  r"n_episodes=(\d+), n_pa_snapshots=(\d+)")
rows, cur = {}, None
unmeasurable = set()
for fn in sorted(glob.glob(f"{here}/m3r2_*_{job}_result.txt")):
    for ln in open(fn):
        m = hdr.search(ln)
        if m:
            cur = (m[1], m[2], int(m[3]), int(m[4]), int(m[5])); continue
        m = re.search(r"M3R2_UNMEASURABLE arm=(\w+) cell=(d\d+) rate=(\d+) boot=(\d+) seed=(\d+)", ln)
        if m:
            unmeasurable.add((m[1], m[2], int(m[3]), int(m[4]), int(m[5]))); continue
        m = gate.search(ln)
        if m and cur:
            rows[cur] = dict(ok=m[1] == "PASS", pf=float(m[2]), lo=float(m[3]),
                             ne=int(m[4]), npa=int(m[5]))

if not rows:
    print(f"no M3R2_PIN rows for job={job}"); sys.exit(1)

print("=== M3R rev2: partition realizability with trace-force ON ===")
print("DIAGNOSTIC ONLY. This job reports no ITL and no goodput.\n")

npas = [r["npa"] for r in rows.values()]
print(f"--- instrument check (the binding constraint in rev1) ---")
print(f"  prefill-active snapshots per probe: min {min(npas)}  median "
      f"{st.median(npas):.0f}  max {max(npas)}")
print(f"  rev1 (trace-force OFF) had 2-28. UNMEASURABLE probes: {len(unmeasurable)}/{len(rows)}")
if len(unmeasurable) == len(rows):
    print("  ⇒ trace-force did NOT fix the sampling. Realizability remains unmeasured;")
    print("    do not read the table below as pass/fail.")

print(f"\n--- per (arm, cell, rate): pin_frac by boot x seed ---")
by = collections.defaultdict(dict)
for (arm, cell, rate, b, s), r in rows.items():
    by[(arm, cell, rate)][(b, s)] = r
for k in sorted(by, key=lambda k: (k[0], int(k[1][1:]), k[2])):
    cells = by[k]
    boots = sorted({b for b, _ in cells}); seeds = sorted({s for _, s in cells})
    print(f"\n  {k[0]} {k[1]} rate={k[2]}")
    print("      " + "".join(f"  seed{s:<6}" for s in seeds))
    for b in boots:
        line = f"    b{b}"
        for s in seeds:
            r = cells.get((b, s))
            line += f"  {r['pf']:.3f}{'*' if r and not r['ok'] else ' '}    " if r else "     -      "
        print(line)
    # seed-indexed vs boot-indexed: which factor explains more spread?
    vals = {(b, s): cells[(b, s)]["pf"] for (b, s) in cells}
    if len(boots) > 1 and len(seeds) > 1:
        seed_means = [st.fmean([vals[(b, s)] for b in boots if (b, s) in vals]) for s in seeds]
        boot_means = [st.fmean([vals[(b, s)] for s in seeds if (b, s) in vals]) for b in boots]
        sv = st.pvariance(seed_means); bv = st.pvariance(boot_means)
        which = ("SEED (workload)" if sv > 2 * bv else
                 "BOOT (server state)" if bv > 2 * sv else "neither dominates")
        print(f"    spread of seed-means {sv:.5f} vs boot-means {bv:.5f}  =>  {which}")
    # ★2026-08-03: UNMEASURABLE probes must NOT be counted as PASS. The first
    # version of this line did exactly that -- all 20 of them -- reprinting the
    # "not measured looks like passed" defect that e1_pin_check.py and
    # m3_analyze.py were both just fixed for. A probe below the snapshot floor
    # is reported in its own bucket and excluded from the PASS denominator.
    unm_here = sum(1 for (b, s) in cells
                   if (k[0], k[1], k[2], b, s) in unmeasurable)
    judged = {(b, s): r for (b, s), r in cells.items()
              if (k[0], k[1], k[2], b, s) not in unmeasurable}
    nfail = sum(1 for r in judged.values() if not r["ok"])
    print(f"    {len(judged)-nfail}/{len(judged)} PASS among JUDGED probes"
          + (f";  {unm_here}/{len(cells)} UNMEASURABLE (excluded)" if unm_here else "")
          + ("   <-- target NOT held in some probes" if nfail else ""))

print("\n--- what this can and cannot say (pre-registered) ---")
print("  CAN: whether the variation is seed-indexed or boot-indexed (M3 confounded them),")
print("       and whether M3's specific failures reproduce under a working instrument.")
print("  CANNOT: an unconditional failure rate over workloads -- the seeds were chosen")
print("       to span M3's observed pass/fail blocks, so they are not a random sample.")
PY
