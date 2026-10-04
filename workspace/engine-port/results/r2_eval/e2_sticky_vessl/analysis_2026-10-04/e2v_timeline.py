#!/usr/bin/env python3
"""Post-hoc overlay of the E2 VESSL run against the VESSL dashboard.

Inputs (fetched job dir):  results/.../e2v_<job>/TIMING.tsv, BUDGET.txt, BANNERS.txt,
                           meta/clk.csv, meta/timing.txt
Outputs: per-boot table in KST (admit/health/bench_start/bench_end/end), gap to next boot,
         phase durations, and mechanical checks:
  C1  exactly 20 boots in registered order
  C2  teardown gap (end_i -> health_{i+1}) within [15 s, 400 s]
  C3  no bench longer than T_cap (401/429/870 s)
  C4  clk.csv power.draw high (>= 60% of max) fraction inside bench windows vs outside
Descriptive only -- this is NOT a decision rule of the E2 registration.
"""
import csv, sys, pathlib, datetime as dt
KST = dt.timezone(dt.timedelta(hours=9))
TCAP = {"a_r4": 401, "a_r2": 429, "b_r3": 870}

def kst(e):
    return dt.datetime.fromtimestamp(int(e), KST).strftime("%H:%M:%S") if e not in ("NA", "") else "NA"

def main(jobdir):
    jobdir = pathlib.Path(jobdir)
    timing = next(jobdir.rglob("TIMING.tsv"))
    rows = list(csv.DictReader(open(timing), delimiter="\t"))
    print(f"boots={len(rows)}  (C1 {'OK' if len(rows)==20 else 'FAIL'})")
    hdr = f"{'boot':22s} {'admit':8s} {'health':8s} {'b_start':8s} {'b_end':8s} {'end':8s} {'load':>5s} {'pre':>5s} {'bench':>6s} {'post':>5s} {'gap':>5s} rc kill"
    print(hdr)
    bad_gap = bad_cap = 0
    for i, r in enumerate(rows):
        def f(k):
            v = r[k]; return int(v) if v not in ("NA", "") else None
        a, h, bs, be, e = (f(k) for k in ("admit_epoch","health_epoch","bench_start_epoch","bench_end_epoch","end_epoch"))
        load = (h - a) if h and a else None
        pre = (bs - h) if bs and h else None
        bench = (be - bs) if be and bs else None
        post = (e - be) if e and be else None
        nxt = rows[i+1] if i+1 < len(rows) else None
        gap = None
        if nxt and e and nxt["health_epoch"] not in ("NA",""):
            gap = int(nxt["health_epoch"]) - e
            if not (15 <= gap <= 400): bad_gap += 1
        cell = r["boot"].split("/")[0] if "/" in r["boot"] else r["boot"][:4]
        cap = next((v for k, v in TCAP.items() if r["boot"].startswith(k)), None)
        if bench and cap and bench > cap: bad_cap += 1
        fmt = lambda x: f"{x:5d}" if x is not None else "   NA"
        print(f"{r['boot']:22s} {kst(r['admit_epoch']):8s} {kst(r['health_epoch']):8s} {kst(r['bench_start_epoch']):8s} {kst(r['bench_end_epoch']):8s} {kst(r['end_epoch']):8s} {fmt(load)} {fmt(pre)} {fmt(bench):>6s} {fmt(post)} {fmt(gap)} {r['bench_rc']:>2s} {r['kill_kind']}")
    print(f"C2 teardown gaps out of [15,400]s: {bad_gap}  C3 bench > T_cap: {bad_cap}")
    clk = jobdir / "meta" / "clk.csv"
    if clk.exists():
        import statistics
        recs = []
        for line in open(clk):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3 or not parts[0][:2].isdigit(): continue
            try:
                ts = dt.datetime.strptime(parts[0][:23], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc)  # nvidia-smi prints container-local time = UTC on VESSL
                pw = float(parts[2].split()[0])
            except Exception: continue
            recs.append((ts.timestamp(), pw))
        if recs:
            pmax = max(p for _, p in recs); thr = 0.6 * pmax
            inb = outb = inb_hi = outb_hi = 0
            wins = [(int(r["bench_start_epoch"]), int(r["bench_end_epoch"])) for r in rows if r["bench_start_epoch"] not in ("NA","") and r["bench_end_epoch"] not in ("NA","")]
            for t, p in recs:
                inside = any(s <= t <= e for s, e in wins)
                if inside: inb += 1; inb_hi += p >= thr
                else: outb += 1; outb_hi += p >= thr
            print(f"C4 clk.csv power>=60%max: inside bench {inb_hi/max(inb,1):.2%} ({inb} samples) | outside {outb_hi/max(outb,1):.2%} ({outb} samples)  [clk timestamps parsed as UTC]")
    t = jobdir / "meta" / "timing.txt"
    if t.exists(): print(open(t).read().strip())

if __name__ == "__main__":
    main(sys.argv[1])
