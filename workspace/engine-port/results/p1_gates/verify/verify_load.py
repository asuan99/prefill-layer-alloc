"""Independent re-load + canonical scoring of the p1_opint artifacts (873944/873945).

Written for the C-1/C-2/C-3 verification of claims-auditor side findings.  Scoring
predicates come from the CANONICAL module ``pdmux_eval.analyze`` (RequestResult.passes,
percentile, summarize_requests); nothing about the predicate is re-implemented here.
Only the file->cell layout (one JSON object per RATE, one file per rep/arm) is local,
because ``load_bench_serving_rounds`` sums durations across lines, which would merge
the four rate cells of a file into one.

Aggregation units are declared explicitly (project methodology gate: unit confusion
has recurred 6 times):
  * scoring unit    = REQUEST (a request is good iff ttft<=3000ms AND p95 of its own
                      token ITLs <= 60ms)
  * goodput unit    = CELL = (model, arm, rate, rep): good_requests / cell_duration_s
  * replication unit= REP (5 per cell), paired by rep index across arms (same seed)
No pooling of requests across reps is used for any interval.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/benchmarks")
from pdmux_eval.analyze import RequestResult, percentile, summarize_requests  # noqa: E402

RAW = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_opint")
MODELS = {"zamba2-27b": "Zamba2-2.7B", "granite-40-h-micro-base": "Granite-4.0-h-micro"}
ARMS = ("plain", "agnostic")
REPS = (1, 2, 3, 4, 5)
TTFT_SLO_MS = 3000.0
ITL_SLO_MS = 60.0

FNAME = re.compile(r"p1op_(?P<model>.+?)_(?P<arm>plain|agnostic)_rep(?P<rep>\d)_(?P<job>\d+)\.jsonl$")


def load_cells():
    """-> {(model, arm, rate, rep): dict(requests=[RequestResult], duration=float, raw=record)}"""
    cells = {}
    for path in sorted(RAW.glob("p1op_*_rep?_*.jsonl")):
        m = FNAME.search(path.name)
        if not m:
            continue
        model, arm, rep = m["model"], m["arm"], int(m["rep"])
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rate = float(rec["request_rate"])
            ttfts = rec["ttfts"]
            itls = rec["itls"]
            outs = rec["output_lens"]
            reqs = [
                RequestResult(
                    request_id=f"{arm}#rep{rep}#r{rate:g}#{i}",
                    ttft_ms=1000.0 * float(ttfts[i]),
                    token_itl_ms=tuple(1000.0 * float(v) for v in (itls[i] or ())),
                    completion_s=math.nan,
                )
                for i in range(len(ttfts))
            ]
            cells[(model, arm, rate, rep)] = {
                "requests": reqs,
                "duration": float(rec["duration"]),
                "output_lens": list(outs),
                "input_lens": list(rec["input_lens"]),
                "completed": int(rec["completed"]),
                "errors": rec.get("errors") or [],
                "file": path.name,
            }
    return cells


def cell_goodput(cell):
    s = summarize_requests(
        cell["requests"], 0.0, cell["duration"], TTFT_SLO_MS, ITL_SLO_MS
    )
    return s


if __name__ == "__main__":
    cells = load_cells()
    print("cells loaded:", len(cells))
    keys = sorted({(k[0], k[2]) for k in cells})
    for model, rate in keys:
        for arm in ARMS:
            vals = []
            for rep in REPS:
                c = cells[(model, arm, rate, rep)]
                vals.append(cell_goodput(c)["slo_goodput_req_s"])
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            print(f"{model:26s} r{rate:g} {arm:9s} goodput {mean:7.4f} +- {sd:6.4f}  "
                  + " ".join(f"{v:6.3f}" for v in vals))
