#!/usr/bin/env python3
"""One-line per-cell console summary for `lambda0.sbatch` (rev3).

Kept as a file rather than an inline `python3 -c` so the sbatch does not need
nested quoting, and so the fields it prints stay tied to the analyzer's schema.

Prints, in this order, the four things the rev2 audit showed a reader needs to
see together: the ratio, the CEILING that ratio was compared against, the
measured DRAIN that produced the gap, and the guards.

usage: lambda0_cellprint.py <cell.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(d) -> str:
    k = d.get("kappa_pred")
    lp = d.get("drain_pred_s")
    return (
        "  ach/off_nom=%.4f  ach/off_REAL=%.4f  (ceiling kappa_pred=%s, "
        "slack vs 0.95 = %s)\n"
        "  span=%.1fs drain=%.2fs (pred %s, model_ok=%s)  lowside=%s\n"
        "  Ebar=%.4f(ok %s)  range_ratio=%s(ok %s)  in_exact=%s out_exact=%s\n"
        "  ach=%.4f req/s  TTFTp50=%.2fs p99=%.2fs  ITLp50=%.1fms"
        % (d["achieved_over_offered"], d["achieved_over_realized"],
           "SAT" if k is None else "%.4f" % k,
           "n/a" if d["achieved_over_realized"] != d["achieved_over_realized"]
           else "%+.4f" % (d["achieved_over_realized"] - 0.95),
           d["span_s"], d["drain_s"],
           "SAT" if lp is None else "%.2f" % lp,
           d["drain_model_ok"], d["low_side_candidate"],
           d["ebar"], d["ebar_guard_ok"],
           d["random_range_ratio"], d["range_ratio_ok"],
           d["input_len_exact"], d["output_len_exact"],
           d["achieved_rate"], d["ttft_p50_s"], d["ttft_p99_s"],
           1000 * d["itl_p50_s"]))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    print(fmt(json.loads(Path(sys.argv[1]).read_text())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
