#!/usr/bin/env python3
"""Block-level r = stat(d16)/stat(d54) for the C2 D=54 anchor campaign.

Reuses results/s8_frontier/c2_anchor.py's collect()/labeling/statistic
UNCHANGED (same SPLIT_HI=0.90 time-weighted realized-partition label, same
per-token-ITL unit, same p95/p50/mean estimators, same in-window vs all-busy
gate distinction). The only thing this file adds is the REPLICATION UNIT:

  c2_anchor.py's own "job" column is, by construction, one server boot per
  cell (s8_sweep.sbatch reboots between cells). Because r crosses cells,
  the true replicate for r is a BLOCK = one boot of d16 AND one boot of d54
  from the same campaign run. s8_sweep_d54.sbatch tags each block's files
  with job field "<slurm_id>_blk<N>", so treating that whole string as
  c2_anchor's "job" already IS the block grouping -- no new label invented.

  Do NOT use c2_anchor.py's own per-rep "r rep mean+-sd" column for this
  campaign: reps 1..4 share one boot (pseudo-replication), which is exactly
  the flaw this campaign was commissioned to avoid. This script pools all
  reps within a block into one point estimate per block, then takes
  mean/sd/CI ACROSS BLOCKS (n_indep = number of blocks with valid data in
  both cells), matching the M3/872077 blocking lesson in MEMORY.md.

Run:
  python3 d54_block_ratio.py <slurm_job_id> [--arm T8] [--blocks 4]
"""
import argparse
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "s8_frontier"))
import c2_anchor as ca  # noqa: E402

TCRIT = ca.TCRIT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slurm_job", help="SLURM job id used in the filenames")
    ap.add_argument("--arm", default=None, help="restrict to one arm label (default: all found)")
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--ctx", default="1024")
    args = ap.parse_args()

    data, resid, meta = ca.collect(job=None, ctx=args.ctx)
    block_jobs = [f"{args.slurm_job}_blk{b}" for b in range(1, args.blocks + 1)]

    pooled = {}
    for (arm, cell, jb, rep), rows in data.items():
        if jb not in block_jobs:
            continue
        pooled.setdefault((arm, cell, jb), []).extend(rows)

    arms = sorted({a for (a, c, j) in pooled})
    if args.arm:
        arms = [a for a in arms if a == args.arm]

    is_split = lambda r: r[1] >= ca.SPLIT_HI       # noqa: E731
    is_un = lambda r: r[2] >= ca.SPLIT_HI          # noqa: E731

    print("=" * 78)
    print(f"D54 BLOCK ANCHOR  job={args.slurm_job}  blocks={args.blocks}  "
          f"arms={arms}  (reusing c2_anchor.py definitions unmodified)")
    print("=" * 78)

    print("\n[1] REALIZED-PARTITION GATE per block (in-window vs all-busy, same as c2_anchor [1])")
    print(f"{'block':>10} {'arm':>4} {'cell':>5} | {'inwin@D':>8} {'inwin@108':>10} | "
          f"{'allbusy@D':>10} {'allbusy@108':>12} {'mean_bs':>8} | gate(in-win>=0.90)")
    gate_ok = {}
    for (arm, cell, jb), rows in sorted(pooled.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
        v = resid.get((arm, cell, jb), {})
        tot = sum(r[4] for r in rows)
        wD = sum(r[1] * r[4] for r in rows) / tot if tot else float("nan")
        w108 = sum(r[2] * r[4] for r in rows) / tot if tot else float("nan")
        ok = wD >= 0.90
        gate_ok[(arm, cell, jb)] = ok
        print(f"{jb:>10} {arm:>4} {cell:>5} | {wD:8.3f} {w108:10.3f} | "
              f"{v.get('frac_target', float('nan')):10.3f} {v.get('frac_108', float('nan')):12.3f} "
              f"{v.get('mean_bs_busy', float('nan')):8.2f} | {'PASS' if ok else 'FAIL'}")

    def block_stat(arm, cell, jb, sel, stat):
        rows = [r for r in pooled.get((arm, cell, jb), []) if sel(r)]
        if len(rows) < 100:
            return None, 0
        return ca.stats_of([r[0] for r in rows])[stat], len(rows)

    for arm in arms:
        print(f"\n[2] BLOCK-LEVEL r = SPLIT(d16)/SPLIT(d54), arm={arm}  "
              f"(n_indep = #blocks with PASS gate on both cells)")
        print(f"{'stat':>5} | {'per-block r':<60} | {'mean+-sd':>16} | {'t95 CI':>18} | n_indep")
        for stat in ("p95", "p50", "mean"):
            rs = []
            detail = []
            for b in range(1, args.blocks + 1):
                jb = f"{args.slurm_job}_blk{b}"
                if not (gate_ok.get((arm, "d16", jb)) and gate_ok.get((arm, "d54", jb))):
                    detail.append(f"blk{b}=SKIP(gate)")
                    continue
                lo, nlo = block_stat(arm, "d16", jb, is_split, stat)
                hi, nhi = block_stat(arm, "d54", jb, is_split, stat)
                if lo is None or hi is None:
                    detail.append(f"blk{b}=SKIP(n<100)")
                    continue
                r = lo / hi
                rs.append(r)
                detail.append(f"blk{b}={r:.3f}(n{nlo}/{nhi})")
            if len(rs) >= 2:
                m, sd = st.fmean(rs), st.stdev(rs)
                t = TCRIT.get(len(rs), 2.0)
                h = t * sd / math.sqrt(len(rs))
                ci = f"[{m-h:.3f},{m+h:.3f}]"
            elif len(rs) == 1:
                m, sd, ci = rs[0], float("nan"), "n/a (n_indep=1)"
            else:
                m, sd, ci = float("nan"), float("nan"), "n/a"
            print(f"{stat:>5} | {', '.join(detail):<60} | {m:7.3f}+-{sd:6.3f} | {ci:>18} | {len(rs)}")

        print(f"\n[2b] NEGATIVE CONTROL, arm={arm}: r = UNSPLIT108(d16)/UNSPLIT108(d54) "
              f"(should be ~1; both cells run at 108 SM here)")
        print(f"{'stat':>5} | {'per-block r':<60} | {'mean+-sd':>16} | {'t95 CI':>18} | n_indep")
        for stat in ("p95", "p50", "mean"):
            rs = []
            detail = []
            for b in range(1, args.blocks + 1):
                jb = f"{args.slurm_job}_blk{b}"
                lo, nlo = block_stat(arm, "d16", jb, is_un, stat)
                hi, nhi = block_stat(arm, "d54", jb, is_un, stat)
                if lo is None or hi is None:
                    detail.append(f"blk{b}=SKIP(n<100)")
                    continue
                r = lo / hi
                rs.append(r)
                detail.append(f"blk{b}={r:.3f}(n{nlo}/{nhi})")
            if len(rs) >= 2:
                m, sd = st.fmean(rs), st.stdev(rs)
                t = TCRIT.get(len(rs), 2.0)
                h = t * sd / math.sqrt(len(rs))
                ci = f"[{m-h:.3f},{m+h:.3f}]"
            elif len(rs) == 1:
                m, sd, ci = rs[0], float("nan"), "n/a (n_indep=1)"
            else:
                m, sd, ci = float("nan"), float("nan"), "n/a"
            print(f"{stat:>5} | {', '.join(detail):<60} | {m:7.3f}+-{sd:6.3f} | {ci:>18} | {len(rs)}")


if __name__ == "__main__":
    main()
