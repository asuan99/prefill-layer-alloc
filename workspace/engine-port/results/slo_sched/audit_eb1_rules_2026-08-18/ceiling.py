#!/usr/bin/env python3
"""E-B1 audit: the ITL CEILING is rate-invariant because the decode batch is capped.

`max_running_requests=48` (all 28 G16 boots).  The server log gives, per decode
step, (#running-req B, gen throughput tok/s G).  Per-request inter-token latency
at that instant is  ITL(B) = 1000 * B / G  ms.  Because B <= 48 ALWAYS, the
achievable ITL is bounded by ITL(48): raising the arrival rate past the point
where B saturates at 48 CANNOT raise ITL further -- it only lengthens the queue
(TTFT).  So the ITL axis has a hard, rate-invariant ceiling per arm.

Read-only: parses the campaign's own srv.log files.
"""
import json, re, statistics
from pathlib import Path
from collections import defaultdict

SLO = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched")
ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
JOBS = [("blk1", "884336"), ("blk2", "884410"), ("blk3", "884411"), ("blk4", "884412")]
RX = re.compile(r"Decode batch, #running-req: (\d+),.*?gen throughput \(token/s\): ([0-9.]+), #queue-req: (\d+)")


def main():
    out = {"note": "ITL(B) = 1000*B/gen_throughput ms, from srv.log decode-batch lines",
           "max_running_requests": 48, "per_arm": {}}
    for arm in ARMS:
        by_b = defaultdict(list)
        bmax_seen = []
        for blk, job in JOBS:
            p = SLO / f"g16_{blk}_{arm}_boot1_{job}_srv.log"
            bs = []
            for line in p.open(errors="ignore"):
                m = RX.search(line)
                if not m:
                    continue
                b, g, q = int(m.group(1)), float(m.group(2)), int(m.group(3))
                if g <= 0:
                    continue
                by_b[b].append(1000.0 * b / g)
                bs.append(b)
            bmax_seen.append(max(bs) if bs else 0)
        rows = {}
        for b in sorted(by_b):
            v = by_b[b]
            rows[b] = {"n": len(v), "median_itl_ms": statistics.median(v)}
        # ceiling = median ITL over steps with B in the top bin (46..48)
        top = [x for b, v in by_b.items() if b >= 46 for x in v]
        out["per_arm"][arm] = {
            "max_B_observed_per_block": bmax_seen,
            "ceiling_itl_ms_B46_48_median": statistics.median(top) if top else None,
            "ceiling_n_steps": len(top),
            "itl_by_B": {str(b): rows[b] for b in sorted(rows) if b in (1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 46, 47, 48)},
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
