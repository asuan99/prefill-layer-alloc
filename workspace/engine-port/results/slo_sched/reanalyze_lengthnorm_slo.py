#!/usr/bin/env python3
"""Re-score the EXISTING varying-trace goodput under a LENGTH-NORMALIZED TTFT SLO.

No re-run: every sgptv{Lo,Hi}_*.jsonl already stores per-request input_lens/ttfts/itls,
so we just replace the goodput indicator TTFT<=3s  with  TTFT<=SLO(L)=a+b*L and recompute.

Question (user, 2026-07-18): the fixed 3s TTFT SLO penalizes long requests -- exactly the
ones where Diff A opens. Does the HE0 ranking (static d44 > dynamic bind) SURVIVE a
length-aware SLO, or is it an artifact of the fixed threshold?

Method:
  1. reproduce fixed-3s goodput  -> sanity vs CONSENSUS (d44 3.220, bind+GATE 3.132, ...)
  2. estimate the PHYSICAL prefill cost line a0+b0*L from the LO (rate3, low-queue) phase
  3. re-score under SLO(L)=k*(a0+b0*L) for several slack multipliers k
  4. report per-policy mean+/-std and the RANKING at each SLO
  5. show fixed-SLO pass-rate by input-length bucket (does the fixed SLO discriminate?)
"""
import json, glob, os, re, statistics as st

TPOT_SLO = 0.06  # unchanged: TPOT is per-token, not length-dependent

def server_log_for(mode, rep, job):
    pat = f"sgptvsrv_{mode}_rep{rep}_L3H12_{job}.log"
    return pat if os.path.exists(pat) else None

def gate_flag(mode, rep, job):
    """bind runs: gate ON iff the server logged SLO-FEAS refusals (campaign only tags rep #,
    not gate state, so we must read the server log)."""
    if mode != "bind":
        return None
    lg = server_log_for(mode, rep, job)
    if not lg:
        return "?"
    n = 0
    try:
        with open(lg) as f:
            for ln in f:
                if "SLO-FEAS refused" in ln:
                    n += 1
    except Exception:
        return "?"
    return "GATE" if n > 0 else "nogate"

def rounds(path):
    """yield (input_lens, ttfts, mean_itls, duration) per round-object in a jsonl file."""
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        o = json.loads(ln)
        IL = o.get("input_lens") or []
        TT = o.get("ttfts") or []
        IT = o.get("itls") or []
        d = o.get("duration") or 0.0
        mitl = [ (sum(x)/len(x) if x else 9.0) for x in IT ]
        yield IL, TT, mitl, d

def goodput(files, slo_fn):
    """combined goodput over LO+HI files, denominator = SUM of round durations (harness bugfix)."""
    good = 0; dur = 0.0; n = 0
    for f in files:
        for IL, TT, MI, d in rounds(f):
            dur += d
            for i in range(len(TT)):
                L = IL[i] if i < len(IL) else 0
                if TT[i] <= slo_fn(L) and (MI[i] if i < len(MI) else 9.0) <= TPOT_SLO:
                    good += 1
                n += 1
    return (good/dur if dur else 0.0), good, n, dur

def discover():
    """group jsonl by policy label. returns {label: [ (rep, [Lo,Hi files]) ]}."""
    groups = {}
    for lo in sorted(glob.glob("sgptvLo_*.jsonl")):
        m = re.match(r"sgptvLo_([a-z0-9]+)_rep(\d+)_L3H12_(\d+)\.jsonl", lo)
        if not m:
            continue
        mode, rep, job = m.group(1), m.group(2), m.group(3)
        hi = lo.replace("sgptvLo_", "sgptvHi_")
        if not os.path.exists(hi):
            continue
        if mode == "bind":
            g = gate_flag(mode, rep, job)
            label = f"bind+{g}"
        else:
            label = mode
        groups.setdefault(label, []).append((f"{rep}/{job}", [lo, hi]))
    return groups

def phys_line():
    """fit ttft ~ a0 + b0*input_len on the LO (rate3) phase of the STATIC d44 runs (least
    queueing distortion -> closest to pure prefill cost). Returns (a0 seconds, b0 s/tok)."""
    xs = []; ys = []
    for lo in glob.glob("sgptvLo_d44_rep*_L3H12_*.jsonl"):
        for IL, TT, MI, d in rounds(lo):
            for i in range(len(TT)):
                if i < len(IL):
                    xs.append(IL[i]); ys.append(TT[i])
    n = len(xs)
    mx = sum(xs)/n; my = sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs); sxy = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    b0 = sxy/sxx; a0 = my - b0*mx
    return a0, b0, n

def bucket_passrate(files, slo_fn, edges):
    """fixed-vs-normalized pass-rate by input-length bucket, to expose discrimination."""
    buckets = {i: [0,0] for i in range(len(edges)+1)}
    for f in files:
        for IL, TT, MI, d in rounds(f):
            for i in range(len(TT)):
                L = IL[i] if i < len(IL) else 0
                bi = sum(1 for e in edges if L >= e)
                ok = TT[i] <= slo_fn(L) and (MI[i] if i < len(MI) else 9.0) <= TPOT_SLO
                buckets[bi][0] += int(ok); buckets[bi][1] += 1
    return buckets

# ---------------------------------------------------------------------------
groups = discover()
a0, b0, nfit = phys_line()
print(f"# PHYSICAL prefill line (fit on d44 LO phase, n={nfit}): TTFT ~ {a0*1000:.0f}ms + {b0*1000:.3f}ms/tok*L")
print(f"#   => a 217-tok req budgets {(a0+b0*217)*1000:.0f}ms ; a 2776-tok(p99) req {(a0+b0*2776)*1000:.0f}ms\n")

# SLO scenarios: fixed 3s (reproduce), and length-normalized k*(a0+b0*L) for k in {2,3,4}
scenarios = [("fixed-3s", lambda L: 3.0)]
for k in (2.0, 3.0, 4.0):
    scenarios.append((f"norm-k{k:.0f}", (lambda k: (lambda L: k*(a0+b0*L)))(k)))
# also an absolute floor variant so tiny reqs aren't starved of budget
scenarios.append(("norm-k3+0.5floor", lambda L: max(0.5, 3.0*(a0+b0*L))))

order = ["d44","d34","d24","d16","slo","bind+GATE","bind+nogate"]
def sortkey(lbl):
    return order.index(lbl) if lbl in order else 99

print("## Per-policy COMBINED goodput under each SLO (mean +/- std over reps)\n")
hdr = "policy".ljust(16) + "n   " + "".join(s[0].ljust(20) for s in scenarios)
print(hdr); print("-"*len(hdr))
ranking = {name: [] for name,_ in scenarios}
for lbl in sorted(groups, key=sortkey):
    reps = groups[lbl]
    row = lbl.ljust(16) + f"{len(reps):<4}"
    for sname, sfn in scenarios:
        vals = [goodput(files, sfn)[0] for _, files in reps]
        m = st.mean(vals); s = st.pstdev(vals) if len(vals)>1 else 0.0
        row += f"{m:.3f}+/-{s:.3f}".ljust(20)
        ranking[sname].append((m, lbl))
    print(row)

print("\n## RANKING by SLO (best -> worst); watch whether d44 stays #1 and bind stays below static\n")
for sname,_ in scenarios:
    rk = sorted(ranking[sname], reverse=True)
    print(f"  {sname:16} " + "  >  ".join(f"{l}({m:.3f})" for m,l in rk))

print("\n## FIXED-3s pass-rate by input-length bucket (d44 reps) -- does the fixed SLO discriminate?\n")
edges = [500, 1000, 2000]
d44files = [f for _,fs in groups.get("d44",[]) for f in fs]
labels = ["<500","500-1k","1k-2k",">=2k"]
for sname, sfn in [("fixed-3s", lambda L:3.0), ("norm-k3", (lambda L: 3.0*(a0+b0*L)))]:
    b = bucket_passrate(d44files, sfn, edges)
    print(f"  {sname}: " + " | ".join(f"{labels[i]} {b[i][0]}/{b[i][1]}={100*b[i][0]/b[i][1] if b[i][1] else 0:.0f}%" for i in range(4)))
